"""Per-scale outward enclosures and exact integer outputs; no new risk theorem."""
from fractions import Fraction
import math
import numpy as np
from flint import arb, ctx

from c7016.algorithm import ball, validate_profiles
from paper_reference import NumericalCertificationError, certified_integer, exact_inputs
from parameter_search import algorithm as alg


def scale_intervals(profiles, plan):
    """Column 0 is D; column j+1 is the corrected polynomial at scale j.

    Same conservative IEEE-754 dot-product enclosure as the frozen implementation.
    Every scale is retained, so ablation and winner attribution use the same data.
    """
    if plan.n > 2**53:
        raise ValueError('Batched scoring requires n<=2^53')
    validate_profiles(profiles, plan.degree)
    D = profiles.observed.astype(float)
    if not plan.scales:
        return D[:, None], D[:, None]
    lo, hi = alg.compiled(plan.C, plan.L0, plan.n, plan.bits)
    x = profiles.low[:, :plan.degree].astype(float)
    down = lambda v: np.nextafter(v, -np.inf)
    up = lambda v: np.nextafter(v, np.inf)
    g = (4*plan.degree+32)*2.**-52
    absolute = x @ np.maximum(np.abs(lo), np.abs(hi)).T
    error = up(up(g*up(absolute/(1-g)))+1e-280)
    a = down(down(x@lo.T-error)+D[:, None])
    b = up(up(x@hi.T+error)+D[:, None])
    delta = np.array([float(s.delta) for s in plan.scales])
    a = down(down(a-delta*plan.n)/(1+delta))
    b = up(up(b-delta*plan.n)/(1+delta))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise NumericalCertificationError('Nonfinite enclosure')
    return np.column_stack((D, a)), np.column_stack((D, b))


def columns(plan, mode):
    if mode == 'count':
        return [0]
    if mode == 'single':
        return [0, 1] if plan.scales else [0]
    if mode == 'no_count':
        if not plan.scales:
            raise ValueError('No-polynomial branch: no_count is undefined')
        return list(range(1, len(plan.scales)+1))
    if mode != 'full':
        raise ValueError('Unknown ablation')
    return list(range(len(plan.scales)+1))


def arb_value(D, low, plan, mode='full', bits=1024):
    with ctx.workprec(bits):
        _, scales = alg.geometry(plan.C, plan.L0, plan.n, bits)
        values = [arb(int(D))]
        for s in scales:
            raw = arb(int(D))
            for f, c in zip(low[:s.k], s.corrections):
                if f:
                    raw += int(f)*c
            values.append((raw-ball(s.delta)*plan.n)/(1+ball(s.delta)))
        chosen = columns(plan, mode)
        score = values[chosen[0]]
        for i in chosen[1:]:
            score = score.max(values[i])
        return score


def compare(profiles, plan, lo, hi, threshold, mode):
    result = lo > threshold
    uncertain = np.flatnonzero((lo <= threshold) & (hi > threshold))
    for i in uncertain:
        for bits in [1024, 2048, 4096, 8192]:
            value = arb_value(profiles.observed[i], profiles.low[i], plan, mode, bits)
            if value > threshold:
                result[i] = True
                break
            if value <= threshold:
                result[i] = False
                break
        else:
            raise NumericalCertificationError('Unresolved event; no guessed label')
    return result, len(uncertain)


def outcomes(profiles, plan, pop, a, b, mode='full'):
    selected = columns(plan, mode)
    lo, hi = a[:, selected].max(axis=1), b[:, selected].max(axis=1)
    truth, tail, eff = pop.truth(plan.n, str(plan.epsilon))
    reject, refinements = compare(profiles, plan, lo, hi, plan.n, mode)
    over = np.zeros(len(lo), bool)
    if pop.support < plan.n:
        over, k = compare(profiles, plan, lo, hi, pop.support, mode)
        refinements += k
    short = np.zeros(len(lo), bool)
    target = min(eff, plan.n)
    if target > 1:
        enough, k = compare(profiles, plan, lo, hi, target-1, mode)
        short = ~enough
        refinements += k
    lower_lo = np.clip(np.ceil(lo), 1, plan.n).astype(np.int64)
    lower_hi = np.clip(np.ceil(hi), 1, plan.n).astype(np.int64)
    binary = reject if truth == 'Yes' else ~reject if truth == 'No' else np.zeros(len(lo), bool)
    return dict(lo=lo, hi=hi, reject=reject, over=over, short=short,
                lower_lo=lower_lo, lower_hi=lower_hi, binary=binary,
                joint=binary | over | short), refinements


def exact_output(D, low, plan, mode='full', precisions=(512, 1024, 2048, 4096, 8192)):
    """Unique clipped integer plus binary decision, otherwise explicit failure."""
    for bits in precisions:
        score = arb_value(D, low, plan, mode, bits)
        with ctx.workprec(bits):
            if score > plan.n:
                return dict(decision='No', lower=plan.n, bits=bits, score=score.str(65))
            if score <= plan.n:
                if score <= 1:
                    bound = 1
                else:
                    try:
                        bound = certified_integer(score, 'ceil')
                    except NumericalCertificationError:
                        continue
                return dict(decision='Yes', lower=min(plan.n, max(1, bound)),
                            bits=bits, score=score.str(65))
    raise NumericalCertificationError('Unique integer unresolved; no midpoint substitution')


def test_fingerprint(fp, n, epsilon, *, mean, L0=14):
    """Full interface for certified C=70, L0 in {14,16}; mean is predeclared.

    Passing the planned real Arb mean is required. A fingerprint alone does not
    certify iid sampling; M is recorded separately and never used as the mean.
    """
    n, eps = exact_inputs(n, epsilon)
    if L0 not in (14, 16):
        raise ValueError('Public interface is limited to the two certified pairs')
    plan = alg.parameters('70', L0, n, eps)
    if not isinstance(mean, arb) or not mean.overlaps(plan.m) or mean.rad() > plan.m.rad()*16:
        raise ValueError('Supply the certified predeclared real Poisson mean')
    low = [fp.frequencies.get(i, 0) for i in range(1, plan.degree+1)]
    output = exact_output(fp.observed, low, plan)
    return dict(**output, C=70, L0=L0, n=n, epsilon=str(eps),
                mean_interval=plan.m.str(65), M=fp.total, observed=fp.observed,
                branch=plan.branch, numerical_output_certified=True,
                theoretical_failure_bound=.0009 if L0 == 14 else .000057,
                sampling_requirement='iid Poisson at the declared mean; not inferred from counts')


def closed_form_weights(A, B, k):
    """Independent positive expansion of T_k(1+z), without Chebyshev recurrence.

    T_k(1+z)=sum_r k*(k+r-1)!*2^r/((k-r)!*(2r)!)*z^r.
    Differentiating the positive expansion gives the alternating coefficients.
    """
    z, beta = 2*A/(B-A), 2/(B-A)
    c = [ball(Fraction(k*math.factorial(k+r-1)*2**r,
                       math.factorial(k-r)*math.factorial(2*r))) for r in range(k+1)]
    normalizer = sum((c[r]*z**r for r in range(k+1)), arb(0))
    result = [arb(0)]
    for ell in range(1, k+1):
        coeff = (-beta)**ell*sum((c[r]*math.comb(r, ell)*z**(r-ell)
                                  for r in range(ell, k+1)), arb(0))
        result.append(1-coeff*math.factorial(ell)/normalizer)
    return tuple(result)
