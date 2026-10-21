#!/usr/bin/env python3
"""Outward-rounded endpoint certificates for the accompanying joint-constant proof.

This is deterministic interval arithmetic, not a Monte Carlo experiment or a
proof-assistant kernel. Analytic reductions are in joint_constants.tex.
Run from the project root: .venv/bin/python artifacts/joint_constants/certify_constants.py
"""
from pathlib import Path
from math import factorial, comb
from fractions import Fraction
import hashlib
import json
from flint import arb, ctx

ctx.prec = 256
HERE = Path(__file__).resolve().parent
LN2, PI = arb(2).log(), arb.pi()
CSTAR = 144 * LN2**2
CHECKS = 0


def check(condition, label):
    global CHECKS
    assert condition, label
    CHECKS += 1


def amin(*xs):
    best = xs[0]
    for x in xs[1:]:
        if x < best:
            best = x
        elif not best < x:
            best = best.union(x)
    return best


def amax(*xs):
    best = xs[0]
    for x in xs[1:]:
        if x > best:
            best = x
        elif not best > x:
            best = best.union(x)
    return best


def coefficient_bound(t, w):
    """Only scale zero: rho <= 1/32; split coefficients at ell=8."""
    candidates = [(t + 2*w/17).exp()]
    for ell in range(1, 8):
        candidates.append((4*t)**ell * factorial(ell) / factorial(2*ell)
                          * (2*w/(2*ell+1)).exp())
    return 1 + amax(*candidates)/32


def margins(j):
    delta = arb(1) / 2**(j+2)
    return (2-delta/2)**2, (14-delta/2-8*delta**2)**2


def probabilities(vs, ws, heavy, large_occupancy):
    upper = sum((v/(v+margins(j)[0]) for j, v in enumerate(vs)), arb(0))
    lower_finite = amax(heavy, *(v/(v+margins(j)[1]) for j, v in enumerate(vs)))
    lower_general = amax(heavy, large_occupancy,
                        *(2*w/(2*w+margins(j)[1]) for j, w in enumerate(ws)))
    joint = amax(upper+lower_finite, lower_general)
    return dict(upper=upper, lower_finite=lower_finite,
                lower_general=lower_general, joint=joint)


def slab(C, a, b):
    """Enclose all real L in [a,b], including scale/degree jumps."""
    a, b = arb(a), arb(b)
    # Include every possibly active scale. No floating-point floor is used.
    ratio = 256*b*b/C
    jmax = 0
    while not arb(2)**(jmax+1) > ratio:
        jmax += 1
    check(arb(2)**(jmax+1) > ratio, 'scale coverage')
    vs, ws = [], []
    for j in range(jmax+1):
        q = (j+3)*LN2
        alpha = (512/C/2**j).sqrt()*q
        k = alpha*b+1
        v = amin(arb(1)/2, C*2**j/(512*a*a))
        t = k*k/(64*a*(1-v))
        w = k*k*v/(1-v)
        coarse = 1+2*(PI*t).sqrt()*t.exp()
        refined = 1+2*(PI*t).sqrt()*(t-2*q+C*2**j/(8*a)).exp()
        candidates = [coarse, refined]
        theta = k/(8*C*2**j).sqrt()
        if theta < 1:
            candidates.append(1/(1-theta))
        integ = amin(*candidates)
        F, V = integ, integ**2/4
        if j == 0:
            coeff = coefficient_bound(t, w)
            F = amin(F, coeff)
            V = amin(V, coeff**2)
        scale = (1-a).exp()*2**(2*j+4)
        vs.append(V*scale)
        ws.append(F*F*scale)
    heavy = (-31*a-1).exp()+(-C).exp()
    out = probabilities(vs, ws, heavy, 2*(1-a).exp())
    out['a'], out['b'] = a, b
    return out


def half_line(C, is_star):
    """Analytic envelopes on the entire half-line L >= 32."""
    H = arb(32)
    ws = []
    for j in range(8):
        q = (j+3)*LN2
        # Cancel log(2)^2 symbolically for the exact endpoint C=CSTAR.
        t = arb((j+3)**2)/(18*2**j) if is_star else 8*q*q/(C*2**j)
        alpha = (64*t).sqrt()
        v = C*2**j/(512*H*H)
        check(v < arb(1)/2, 'low scale v < 1/2')
        d = (alpha*H+1)**2/(64*H*(1-v))-t*H
        th = t*H+d
        if j == 0:
            if is_star:
                check(2*t == 1, 'exact symbolic endpoint exponent')
            else:
                check(2*t < 1, 'scale-zero exponential decay')
            # For CSTAR, 2*t=1 by the displayed exact rational formula.
            W = (alpha+1/H)**2*C/(512*(1-v))
            F = coefficient_bound(th, W)
            check(H >= 14, 'small coefficient polynomials decrease')
        else:
            check(2*t < 1, 'positive decay rate')
            check(H*(1-2*t) > 1, 'all expanded envelope terms decrease')
            z = amin(d, d-2*q+C*2**j/(8*H))
            F = 1+2*(PI*th).sqrt()*(t*H+z).exp()
        ws.append(F*F*(1-H).exp()*2**(2*j+4))
    alpha8 = (512/C/256).sqrt()*11*LN2
    beta2 = (alpha8+1/H)**2/32
    check(H*(1-2*beta2) > 5, 'infinite scale tail envelope decreases')
    Ft = 1+2*(PI*beta2*H).sqrt()*(beta2*H).exp()
    tail = Ft*Ft*(1-H).exp()*arb(4)/3*(1024*H*H/C)**2
    # Conservative: V <= F^2 for low scales and all omitted scales.
    upper = sum((w/(w+margins(j)[0]) for j, w in enumerate(ws)), arb(0))+tail
    finite = amax(*(w/(w+margins(j)[1]) for j, w in enumerate(ws)), tail/81,
                  (-31*H-1).exp()+(-C).exp())
    general = amax(*(2*w/(2*w+margins(j)[1]) for j, w in enumerate(ws)),
                   2*tail/81, 2*(1-H).exp(), (-31*H-1).exp()+(-C).exp())
    return dict(upper=upper, lower_finite=finite, lower_general=general,
                joint=amax(upper+finite, general), tail=tail, low_ws=ws)


def counting(C, H):
    # n < 100 gives L < 6; n >= 100 and L < H gives x > C/H.
    check(1+arb(100).log() < 6, 'count branch split')
    def g(x):
        check(x > 1, 'Poisson Chernoff x > 1')
        return x-1-x.log()
    error = amax((-g(C/6)).exp(), (-100*g(C/H)).exp())
    check(error < arb(1)/1000, 'uniform count failure < .001')
    return error


def serialize(value):
    if isinstance(value, arb):
        return str(value)
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize(v) for v in value]
    return value


def main():
    reports = []
    for label, C, is_star in [('144log(2)^2', CSTAR, True), ('70', arb(70), False)]:
        check(C > 69, 'C > 69')
        check(C < 71, 'C < 71')
        # Parameter validity on L >= 15; both expressions decrease thereafter.
        check(256*15**2/C > 1, 'J nonnegative')
        check(64*15**2/(C*arb(14).exp()) < 1, 'b < 1')
        check((512/C).sqrt()*3*LN2+arb(1)/15 < 6, 'degree <= 6L')
        check(64/arb(2).sqrt() > 6, 'expectation condition')
        slabs = [slab(C, arb(i)/10, arb(i+1)/10) for i in range(150, 320)]
        tail = half_line(C, is_star)
        for H in [15, 16, 17, 18, 19, 20]:
            chosen = slabs[(H-15)*10:]
            cerror = counting(C, arb(H))
            maximum = {key: amax(cerror, tail[key], *(x[key] for x in chosen))
                       for key in ['upper', 'lower_finite', 'lower_general', 'joint']}
            target = Fraction(2, 5) if H == 15 else Fraction(1, 3)
            check(maximum['joint'] < arb(target.numerator)/target.denominator,
                  f'{label}, L0={H}: joint failure target')
            check(maximum['upper'] < arb(target.numerator)/target.denominator,
                  'binary null error target')
            check(maximum['lower_general'] < arb(target.numerator)/target.denominator,
                  'binary alternative error target')
            if label == '70' and H == 20:
                check(maximum['joint'] < arb(8)/100, 'recommended single-run success >= .92')
            reports.append(dict(C=label, L0=H, maximum=maximum, count=cerror))
        (HERE/f'slabs_{"star" if is_star else "70"}.json').write_text(
            json.dumps(serialize(dict(C=C, slabs=slabs, half_line=tail)), indent=2)+'\n')
    # Exact binomial arithmetic for amplification examples.
    amplification = []
    for success, repeats in [(Fraction(2, 3), 47), (Fraction(3, 5), 133),
                             (Fraction(23, 25), 5)]:
        failure = sum((Fraction(comb(repeats, k))*success**(repeats-k)*(1-success)**k
                       for k in range((repeats+1)//2, repeats+1)), Fraction())
        check(failure < Fraction(1, 100), 'amplification below one percent')
        amplification.append(dict(success=str(success), repeats=repeats,
                                  exact_failure=str(failure), decimal=float(failure)))
    payload = dict(passed=True, arithmetic='Arb 256-bit outward-rounded real balls; exact Fraction binomial tails',
                   C_star=CSTAR, checks=CHECKS, reports=reports, amplification=amplification,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='Endpoint certificates plus the analytic reductions in joint_constants.tex; not a formal proof-assistant verification.')
    (HERE/'certificates.json').write_text(json.dumps(serialize(payload), indent=2)+'\n')
    print(f'PASS: {CHECKS} checks; 340 complete L-intervals; two analytic half-lines.')
    for report in reports:
        print(report['C'], report['L0'], 'joint failure <=', report['maximum']['joint'])


if __name__ == '__main__':
    main()
