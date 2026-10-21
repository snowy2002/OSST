"""Certified polynomial families, fair sequential ablations, and trial summaries.

The FH-shaped practical family changes constants and re-certifies its null bound.
It is deliberately not labelled as the original FH theorem implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
from flint import arb, ctx
from scipy.stats import beta

from benchmark_core import Population, Profiles, headtail, rank_blocks, uniform
from guaranteed import (NoCertificate, Look, Plan, ball, directed_float,
                        certified_count_budget, validation_cutoff)
from guaranteed_experiments import batch_no, simulate_trajectory, validation_draws
from paper_reference import reference_weights
from support_size import make_scales, chebyshev_weights


BASE = dict(b_factor=.5, max_degree=12, weight_cap=20., single_j=0,
            validation_factor=12., validation_gate=.75, density=1,
            fh_a=.125, fh_b=.5, fh_k=.5)
FACTORS = (.5, 1., 2., 4., 8.)
FAMILIES = ("multi", "single", "fh_certified")


def signature(config):
    return tuple(sorted({**BASE, **config}.items()))


@dataclass(frozen=True)
class CertifiedScale:
    certificate: NoCertificate
    rho_upper: float
    margin_upper: float
    A: float
    B: float


def _compile(n, m, A, B, k, j, alpha, multiplicity, cap):
    """All analytical assumptions and endpoints are verified using Arb."""
    if not (0 < A and A < B and B <= m):
        return None
    while k >= 1:
        if not (B*(B-A)).sqrt() >= k:
            k -= 1
            continue
        weights = reference_weights(A, B, k)
        if all(w.abs_upper() <= ball(cap) for w in weights):
            break
        k -= 1
    if k < 1:
        return None
    rho = 1/((B+A)/(B-A)).chebyshev_t(k)
    width = max([arb(1)]+[w.upper() for w in weights])-min([arb(0)]+[w.lower() for w in weights])
    margin = width*(arb(n)/2*(ball(multiplicity)/ball(alpha)).log()).sqrt()
    threshold = directed_float(n*(1+rho)+margin, True)
    corrections = tuple(directed_float(w-1, False) for w in weights[1:])
    rational = [Fraction.from_float(x) for x in (*corrections, threshold)]
    denominator = max(x.denominator for x in rational)
    ints = [x.numerator*(denominator//x.denominator) for x in rational]
    cert = NoCertificate(j, k, tuple(ints[:-1]), denominator, ints[-1], corrections,
                         threshold, rho.str(60), width.upper().str(60))
    return CertifiedScale(cert, directed_float(rho, True), directed_float(margin, True), float(A), float(B))


@lru_cache(maxsize=16000)
def compile_scales(n, eps, m, family="multi", config_key=(), alpha="0.05", looks=1):
    cfg = {**BASE, **dict(config_key)}
    ep = Fraction(str(eps))
    with ctx.workprec(256):
        specs = []
        if family in ("multi", "single"):
            props = make_scales(n, float(ep), m, b_factor=cfg["b_factor"],
                                max_degree=cfg["max_degree"], weight_cap=cfg["weight_cap"])
            if family == "single":
                props = [s for s in props if s.j == cfg["single_j"]]
            B = (ball(cfg["b_factor"])*(arb(n).log()+1)).min(arb(m))
            specs = [(ball(Fraction(m, 8*n)*ep*2**s.j), B, s.k, s.j) for s in props]
        elif family == "fh_certified":
            loge = (1/ball(ep)).log()
            A = m*ball(cfg["fh_a"])*ball(ep)*loge/n
            B = (m*ball(cfg["fh_b"])*ball(ep)*(arb(n).log()+1)**2/(n*loge)).min(arb(m))
            k = min(cfg["max_degree"], max(1, math.ceil(cfg["fh_k"]*(1+math.log(n)))))
            specs = [(A, B, k, -1)]
        else:
            raise ValueError(family)
        # Spending on proposed scales is conservative if some cannot be certified.
        mult = max(1, len(specs))*looks
        result = [_compile(n, m, A, B, k, j, Fraction(str(alpha)), mult, cfg["weight_cap"])
                  for A, B, k, j in specs]
        return tuple(s for s in result if s is not None)


def scale_rejections(fp, scales, n):
    return [batch_no(fp, Look(1, (s.certificate,), 0, -1, "0"), n) & (fp.observed <= n)
            for s in scales]


def certified_reject(fp, scales, n):
    return batch_no(fp, Look(1, tuple(s.certificate for s in scales), 0, -1, "0"), n)


def certified_lower(fp, scales, n):
    """Integer floors of conservative rational bounds; no unsafe ceil rounding.

    Flooring costs <1 category and preserves coverage. Clipping gives output in
    [1,n]. Effective-support power is measured, not claimed by this certificate.
    """
    result = np.minimum(n, np.maximum(1, fp.observed)).astype(np.int64)
    for s in scales:
        c = s.certificate
        margin = Fraction.from_float(s.margin_upper)
        divisor = 1+Fraction.from_float(s.rho_upper)
        # Common-denominator exact comparison evaluates floor((Z_lower-u_upper)/(1+rho_upper)).
        den = max(c.denominator, margin.denominator)
        raw = [int(D)*den + sum(a*int(f)*(den//c.denominator)
                  for a, f in zip(c.coefficients, row)) - margin.numerator*(den//margin.denominator)
               for D, row in zip(fp.observed, fp.low[:, :c.degree])]
        values = np.array([max(1, min(n, (v*divisor.denominator)//(den*divisor.numerator)))
                           for v in raw], dtype=np.int64)
        result = np.maximum(result, values)
    return result


@lru_cache(maxsize=8000)
def fh_native_weights(n, eps, m, config_key):
    cfg = {**BASE, **dict(config_key)}
    e = float(eps)
    A = m*cfg["fh_a"]*e*math.log(1/e)/n
    B = m*cfg["fh_b"]*e*(1+math.log(n))**2/(n*math.log(1/e))
    k = min(cfg["max_degree"], max(1, math.ceil(cfg["fh_k"]*(1+math.log(n)))))
    if not 0 < A < B:
        return ()
    return chebyshev_weights(A, B, k, precision=120)


def fh_native_reject(fp, n, eps, m, config):
    """Tuned FH-shaped statistic and native threshold; no theorem claim."""
    w = fh_native_weights(n, str(eps), m, signature(config))
    score = fp.observed.astype(float)
    if w:
        score = score + fp.low[:, :len(w)-1] @ (np.asarray(w[1:])-1)
    return score >= (1+float(eps)/2)*n


@lru_cache(maxsize=4000)
def make_seq_plan(n, eps, delta="0.05", family="multi", validation=True, config_key=()):
    cfg = {**BASE, **dict(config_key)}
    ep, de = Fraction(str(eps)), Fraction(str(delta))
    if not 0 < de < Fraction(1, 2):
        raise ValueError("delta must lie in (0,1/2)")
    cap, bound = certified_count_budget(n, ep, de/2 if validation else de)
    baseline, basebound = certified_count_budget(n, ep, de)
    factors = np.geomspace(.5, 8, 1+4*int(cfg["density"]))
    means = sorted({max(1, math.ceil(float(c)*n/(float(ep)*(1+math.log(n))))) for c in factors})
    means = [m for m in means if 5*m < 4*cap]
    H = len(means)
    r, k, vbound = 0, -1, "0"
    if validation and H:
        r = min(math.ceil(cfg["validation_factor"]*math.log(float(2*H/de))/float(ep)), cap//(5*H))
        k, vbound = validation_cutoff(r, ep, de/(2*H))
        if k < 0:
            r = 0
    looks = []
    for m in means:
        scales = () if family == "none" else compile_scales(n, str(ep), m, family, signature(cfg), str(de), H)
        looks.append(Look(m, tuple(s.certificate for s in scales), r, k, vbound))
    return Plan(n, ep, de, cfg, tuple(looks), cap, bound, baseline, basebound)


def decide(plan, profiles, outside, use_poly=True, use_validation=True, omit_j=None):
    T = next(iter(profiles.values())).trials
    active = np.ones(T, bool)
    reject = np.zeros(T, bool)
    train = np.zeros(T, np.int64)
    validation = np.zeros(T, np.int64)
    reasons = np.full(T, "", dtype="U12")
    scale_ids = np.full(T, -999, np.int64)
    for look in plan.looks:
        fp = profiles[look.m]
        direct = active & (fp.observed > plan.n)
        active[direct] = False
        reject[direct] = True
        train[direct] = look.m
        reasons[direct] = "observed"
        if use_poly:
            # Iteration order gives a deterministic first winning scale.
            for c in look.certificates:
                if c.j == omit_j:
                    continue
                hit = active & batch_no(fp, replace(look, certificates=(c,)), plan.n)
                active[hit] = False
                reject[hit] = True
                train[hit] = look.m
                reasons[hit] = "polynomial"
                scale_ids[hit] = c.j
        if use_validation and look.validation_r:
            gate = active & (fp.low[:, 0] <= int(look.m*plan.epsilon*Fraction(str(plan.config["validation_gate"]))))
            validation[gate] += look.validation_r
            yes = gate & (outside[look.m] <= look.validation_k)
            active[yes] = False
            train[yes] = look.m
            reasons[yes] = "validation"
    reject[active] = profiles[plan.final_m].observed[active] > plan.n
    train[active] = plan.final_m
    reasons[active] = "fallback"
    return dict(reject=reject, cost=train+validation, training=train, validation=validation,
                reason=reasons, scale=scale_ids)


def cp_upper(k, N, alpha=.05):
    return 1. if k == N else float(beta.isf(alpha, k+1, N-k))


def trial_summary(reject, cost, truth, baseline=None):
    T = len(reject)
    errors = None if truth == "Gap" else int(np.sum(reject if truth == "Yes" else ~reject))
    result = dict(trials=T, rejects=int(np.sum(reject)), reject_rate=float(np.mean(reject)),
                  errors=errors, error_rate=None if errors is None else errors/T,
                  error_upper95=None if errors is None else cp_upper(errors, T))
    if cost is not None:
        cost = np.asarray(cost)
        result.update(mean_cost=float(cost.mean()), cost_se=float(cost.std(ddof=1)/math.sqrt(T)),
                      cost_median=float(np.median(cost)), cost_p90=float(np.quantile(cost,.9)),
                      cost_p99=float(np.quantile(cost,.99)), max_cost=float(cost.max()))
    if baseline is not None:
        diff = cost-baseline
        se = float(diff.std(ddof=1)/math.sqrt(T))
        result.update(relative_cost=float(cost.mean()/baseline.mean()), paired_difference=float(diff.mean()),
                      paired_difference_se=se, paired_difference_low95=float(diff.mean()-1.96*se),
                      paired_difference_high95=float(diff.mean()+1.96*se))
    return result


def near_boundary(n, eps, ratio=1., tail_factor=100, exponent=None):
    """A known distribution; truth is recomputed after floating normalization."""
    target = float(eps)*ratio
    tail = uniform(max(2*n, int(tail_factor*n))) if exponent is None else rank_blocks(max(2*n,int(tail_factor*n)), exponent)
    def mix(t):
        return Population(np.r_[1, tail.multiplicities], np.r_[1-t, t*tail.probabilities])
    lo, hi = 1e-15, .999999999999
    if mix(hi).tail_mass(n) < target:
        raise ValueError("Unattainable boundary")
    for _ in range(80):
        mid = (lo+hi)/2
        if mix(mid).tail_mass(n) < target:
            lo = mid
        else:
            hi = mid
    p = mix(hi)
    # At equality use the upper representable endpoint and record its exact margin.
    for _ in range(10):
        if p.tail_mass(n) >= target:
            return p
        hi = np.nextafter(hi, 1.)
        p = mix(hi)
    raise ArithmeticError("Boundary could not be represented on requested side")


def formal_cases(n, eps):
    return [("uniform_null", uniform(n)), ("head90_null", headtail(.9,n-1)),
            ("rare_null", headtail(.999,n-1)), ("power_null",rank_blocks(n,1.43)),
            ("uniform_boundary",uniform(math.ceil(n/(1-eps))+1)),
            ("uniform_clear",uniform(2*n)),
            ("exact_diffuse_boundary",near_boundary(n,eps)),
            ("near_multilevel_boundary",near_boundary(n,eps,1.0001,100,.91)),
            ("clear_tail",near_boundary(n,eps,2,100)),
            ("gap_near",near_boundary(n,eps,.999,100)),
            ("gap_half",near_boundary(n,eps,.5,100))]
