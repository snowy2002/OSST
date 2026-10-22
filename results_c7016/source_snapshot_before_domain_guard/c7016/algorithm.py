"""The real-mean (C,L0)=(70,16) construction, without degree truncation.

Arb certifies parameters and weights. Batched statistics use outward binary64
enclosures with a conservative dot-product roundoff bound; unresolved events
are recomputed with Arb. This certifies arithmetic on supplied counts, not the
sampling design or the binary64 pseudorandom generator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from flint import arb, ctx, fmpq

from benchmark_core import Profiles
from guaranteed import directed_float
from paper_reference import (NumericalCertificationError, certified_integer,
                             exact_inputs, reference_weights)
from support_size import Fingerprint

C0, L0 = 70, 16
FAILURE = Fraction(282612, 10**6)


def ball(x):
    if isinstance(x, arb):
        return x
    q = x if isinstance(x, Fraction) else Fraction(x)
    return arb(fmpq(q.numerator, q.denominator))


@dataclass(frozen=True)
class Scale:
    j: int
    A: arb
    B: arb
    k: int
    delta: Fraction
    rho: arb
    corrections: tuple


@dataclass(frozen=True)
class Plan:
    n: int
    epsilon: Fraction
    L: arb
    m: arb
    scales: tuple[Scale, ...]
    bits: int

    @property
    def branch(self):
        return 'chebyshev' if self.scales else 'observed_count'

    @property
    def degree(self):
        return max((s.k for s in self.scales), default=0)

    def as_dict(self):
        return dict(algorithm='joint_constants_70_16', C0=C0, L0=L0,
                    n=self.n, epsilon=str(self.epsilon), branch=self.branch,
                    mean_expression=f'70*{self.n}/(({self.epsilon})*(1+ln({self.n})))',
                    mean_interval=self.m.str(65), mean_for_binary64_rng=float(self.m),
                    mean_rounding='real expression; no integer ceiling',
                    L_interval=self.L.str(65), precision_bits=self.bits,
                    max_degree=self.degree,
                    scales=[dict(j=s.j, k=s.k, delta=str(s.delta),
                                 A=s.A.str(65), B=s.B.str(65), rho=s.rho.str(65))
                            for s in self.scales])


@lru_cache(maxsize=512)
def geometry(n: int, bits: int):
    with ctx.workprec(bits):
        L = arb(n).log()+1
        if L < L0:
            return L, ()
        if not L >= L0:
            raise NumericalCertificationError('Unresolved branch boundary')
        J = certified_integer((256*L*L/C0).log()/arb(2).log(), 'floor')
        result = []
        for j in range(J+1):
            # These dimensionless parameters cancel epsilon exactly.
            A, B = C0*arb(2)**j/(8*L), 64*L
            delta = Fraction(1, 2**(j+2))
            k = certified_integer((B/A).sqrt()*ball(2/delta).log(), 'ceil')
            if k > 4096:
                raise ValueError('Full degree exceeds resource limit 4096; not truncated')
            rho = 1/((B+A)/(B-A)).chebyshev_t(k)
            if not (0 < A and A <= B/2 and (B*(B-A)).sqrt() >= k
                    and rho <= ball(delta*delta/2)):
                raise NumericalCertificationError('Uncertified theorem parameter condition')
            weights = reference_weights(A, B, k)
            result.append(Scale(j, A, B, k, delta, rho, tuple(w-1 for w in weights[1:])))
        return L, tuple(result)


def parameters(n, epsilon, *, bits=512):
    n, ep = exact_inputs(n, epsilon)
    with ctx.workprec(bits):
        L, scales = geometry(n, bits)
        m = C0*arb(n)/(ball(ep)*L)
        if scales and not scales[0].B/m <= 1:
            raise NumericalCertificationError('Polynomial endpoint exceeds one')
        return Plan(n, ep, L, m, scales, bits)


@lru_cache(maxsize=512)
def compiled(n, bits=512):
    """Certified binary64 lower/upper endpoints for every correction."""
    with ctx.workprec(bits):
        _, scales = geometry(n, bits)
        K = max((s.k for s in scales), default=0)
        lo = np.zeros((len(scales), K)); hi = lo.copy()
        for s in scales:
            for ell, value in enumerate(s.corrections):
                lo[s.j, ell] = directed_float(value, False)
                hi[s.j, ell] = directed_float(value, True)
        return lo, hi


def validate_profiles(profiles, degree):
    D, low = np.asarray(profiles.observed), np.asarray(profiles.low)
    if (D.ndim != 1 or low.ndim != 2 or len(low) != len(D)
            or low.shape[1] < degree or D.dtype.kind not in 'iu' or low.dtype.kind not in 'iu'
            or np.any(D < 0) or np.any(D > 2**53) or np.any(low < 0)
            or np.any(low > 2**53) or np.any(low.sum(axis=1, dtype=object) > D)):
        raise ValueError('Require exact nonnegative counts <=2^53, and sum(F_low)<=D')


def score_intervals(profiles: Profiles, plan: Plan):
    """Return enclosing scores for full and scale-zero-only statistics.

    Assumes standard IEEE-754 binary64 arithmetic without overflow. For K-term
    dot products the usual gamma_K bound is dominated by g=(4K+32)*2^-52.
    Positive absolute sums also receive rounding compensation. nextafter then
    encloses each addition, subtraction and division. A tiny additive allowance
    dominates all possible subnormal absolute roundoff in these bounded dots.
    """
    validate_profiles(profiles, plan.degree)
    D = profiles.observed.astype(np.float64)
    if not plan.scales:
        return D.copy(), D.copy(), D.copy(), D.copy()
    lower_w, upper_w = compiled(plan.n, plan.bits)
    x = profiles.low[:, :plan.degree].astype(np.float64)
    down = lambda v: np.nextafter(v, -np.inf)
    up = lambda v: np.nextafter(v, np.inf)
    g = (4*plan.degree+32)*2.**-52
    absolute = x @ np.maximum(np.abs(lower_w), np.abs(upper_w)).T
    absolute_upper = up(absolute/(1-g))
    error = up(up(g*absolute_upper)+1e-280)
    raw_lo = down(down((x @ lower_w.T)-error)+D[:, None])
    raw_hi = up(up((x @ upper_w.T)+error)+D[:, None])
    delta = np.array([float(s.delta) for s in plan.scales])
    adjusted_lo = down(down(raw_lo-delta*plan.n)/(1+delta))
    adjusted_hi = up(up(raw_hi-delta*plan.n)/(1+delta))
    if not np.all(np.isfinite(adjusted_lo)) or not np.all(np.isfinite(adjusted_hi)):
        raise NumericalCertificationError('Nonfinite interval; no numerical output returned')
    return (np.maximum(D, adjusted_lo.max(axis=1)),
            np.maximum(D, adjusted_hi.max(axis=1)),
            np.maximum(D, adjusted_lo[:, 0]), np.maximum(D, adjusted_hi[:, 0]))


def arb_score(observed, low, n, *, bits=768, single=False):
    """Full arbitrary-precision formula on sufficient statistics; no fake M."""
    with ctx.workprec(bits):
        _, scales = geometry(n, bits)
        score = arb(int(observed))
        if single:
            scales = scales[:1]
        for scale in scales:
            raw = arb(int(observed))
            for count, correction in zip(low[:scale.k], scale.corrections):
                if count:
                    raw += int(count)*correction
            adjusted = (raw-ball(scale.delta)*n)/(1+ball(scale.delta))
            score = score.max(adjusted)
        return score


def compare_scores(profiles, n, lower, upper, threshold, *, single=False):
    """Certify score>threshold; refine only unresolved threshold comparisons."""
    result = lower > threshold
    uncertain = np.flatnonzero((lower <= threshold) & (upper > threshold))
    for i in uncertain:
        for bits in [768, 1536, 3072, 6144]:
            value = arb_score(profiles.observed[i], profiles.low[i], n, bits=bits, single=single)
            if value > threshold:
                result[i] = True
                break
            if value <= threshold:
                result[i] = False
                break
        else:
            raise NumericalCertificationError('Threshold remains unresolved; no guessed decision')
    return result, len(uncertain)


def test_fingerprint(fp: Fingerprint, n, epsilon):
    """Public exact-output path for a full fingerprint and predeclared n/epsilon.

    Required Poisson mean is the expression returned by parameters(). The total
    M in a fingerprint is an observation, never substituted for that mean.
    """
    n, ep = exact_inputs(n, epsilon)
    for bits in [512, 1024, 2048, 4096, 8192]:
        plan = parameters(n, ep, bits=bits)
        low = [fp.frequencies.get(ell, 0) for ell in range(1, plan.degree+1)]
        score = arb_score(fp.observed, low, n, bits=bits)
        with ctx.workprec(bits):
            try:
                if score > n:
                    decision, bound = 'No', n
                elif score <= n:
                    decision = 'Yes'
                    bound = max(1, min(n, certified_integer(score, 'ceil')))
                else:
                    raise NumericalCertificationError('Unresolved decision')
                return dict(**plan.as_dict(), M=fp.total, observed=fp.observed,
                            decision=decision, support_lower_bound=bound,
                            score_interval=score.str(65), numeric_output_certified=True,
                            model_requirement='Independent iid Poisson sampling at the stated real mean')
            except NumericalCertificationError:
                continue
    raise NumericalCertificationError('Could not certify an exact integer output')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'test', 'demo'])
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--epsilon', default='0.1')
    parser.add_argument('--fingerprint', type=Path)
    parser.add_argument('--seed', type=int, default=7016001)
    args = parser.parse_args(argv)
    plan = parameters(args.n, args.epsilon)
    if args.action == 'plan':
        result = plan.as_dict()
    elif args.action == 'test':
        if args.fingerprint is None:
            parser.error('--fingerprint is required')
        fp = Fingerprint({int(k): v for k, v in json.loads(args.fingerprint.read_text()).items()})
        result = test_fingerprint(fp, args.n, args.epsilon)
    else:
        if args.n > 5_000_000:
            parser.error('Explicit category-count demo is limited to n<=5,000,000')
        rng = np.random.default_rng(args.seed)
        counts = rng.poisson(float(plan.m)/args.n, size=args.n)
        result = test_fingerprint(Fingerprint.from_counts(counts), args.n, args.epsilon)
        result['sampling_note'] = 'Synthetic iid uniform population; binary64 Poisson RNG'
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
