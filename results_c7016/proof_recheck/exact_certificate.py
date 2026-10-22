#!/usr/bin/env python3
"""Recompute every publication certificate with exact rational enclosures.

Run: python3 exact_certificate.py
Dependencies: Python standard library and rational_interval.py only.
No randomized tests, floating-point libraries, or previous certificate input.
"""
from fractions import Fraction as Q
from math import factorial, comb
from pathlib import Path
import hashlib
import json
import time
from rational_interval import I, imin, imax, log_two, pi_interval, decimal_ceiling, BITS

ROOT = Path(__file__).resolve().parent
LOG2, PI = log_two(), pi_interval()
CSTAR = 144*LOG2**2
CHECKS = []


def require(condition, description):
    if not condition:
        raise ArithmeticError(description)
    CHECKS.append(description)


def coefficient_majorant(t, w):
    candidates = [(t+2*w/17).exp()]
    for ell in range(1, 8):
        candidates.append((4*t)**ell*factorial(ell)/factorial(2*ell)
                          *(2*w/(2*ell+1)).exp())
    return 1+imax(*candidates)/32


def margins(j):
    delta = I(Q(1, 2**(j+2)))
    return (2-delta/2)**2, (14-delta/2-8*delta**2)**2


def risk(vs, ws, L, C):
    heavy = (-31*L-1).exp()+(-C).exp()
    upper = sum((v/(v+margins(j)[0]) for j, v in enumerate(vs)), I(0))
    finite = imax(heavy, *(v/(v+margins(j)[1]) for j, v in enumerate(vs)))
    general = imax(heavy, 2*(1-L).exp(),
                   *(2*w/(2*w+margins(j)[1]) for j, w in enumerate(ws)))
    return dict(upper=upper, lower_finite=finite, lower_general=general,
                joint=imax(upper+finite, general))


def complete_slab(C, numerator):
    ell, u = I(Q(numerator, 10)), I(Q(numerator+1, 10))
    ratio = 256*u*u/C
    jmax = 0
    while not I(2**(jmax+1)) > ratio:
        jmax += 1
    require(I(2**(jmax+1)) > ratio, f'scale coverage slab {numerator}')
    vs, ws = [], []
    for j in range(jmax+1):
        q = (j+3)*LOG2
        alpha = (512/C/2**j).sqrt()*q
        K = alpha*u+1
        v = imin(I(Q(1, 2)), C*2**j/(512*ell*ell))
        T = K*K/(64*ell*(1-v))
        W = K*K*v/(1-v)
        A = C*2**j/(8*ell)
        prefactor = 2*(PI*T).sqrt()
        candidates = [1+prefactor*T.exp(), 1+prefactor*(T-2*q+A).exp()]
        theta = K/(8*C*2**j).sqrt()
        if theta < 1:
            candidates.append(1/(1-theta))
        integral = imin(*candidates)
        F, V = integral, integral**2/4
        if j == 0:
            coeff = coefficient_majorant(T, W)
            F, V = imin(F, coeff), imin(V, coeff**2)
        weight = (1-ell).exp()*2**(2*j+4)
        ws.append(F*F*weight)
        vs.append(V*weight)
    out = risk(vs, ws, ell, C)
    out.update(slab_numerator=numerator, largest_possible_scale=jmax)
    out['rational_inputs'] = dict(v_upper=[v.upper_fraction() for v in vs],
                                 w_upper=[w.upper_fraction() for w in ws],
                                 heavy_upper=((-31*ell-1).exp()+(-C).exp()).upper_fraction(),
                                 occupancy_upper=(2*(1-ell).exp()).upper_fraction())
    return out


def infinite_tail(C, symbolic_star):
    H = I(32)
    ws = []
    for j in range(8):
        q = (j+3)*LOG2
        t = I(Q((j+3)**2, 18*2**j)) if symbolic_star else 8*q*q/(C*2**j)
        alpha = (64*t).sqrt()
        v = C*2**j/(512*H*H)
        require(v < I(Q(1, 2)), f'half-line shift bound j={j}')
        # Equivalent to the endpoint difference; this expression avoids
        # subtracting two nearly equal interval values.
        xi = C*2**j/512
        d = (2*alpha+(1+alpha*alpha*xi)/H)/(64*(1-v))
        th = t*H+d
        if j == 0:
            if symbolic_star:
                require((2*t).exact_equal(1), 'symbolic cancellation gives 2*t0=1')
            else:
                require(2*t < 1, 'integer-C scale-zero decay')
            W = (alpha+1/H)**2*C/(512*(1-v))
            F = coefficient_majorant(th, W)
            require(H >= 14, 'all seven polynomial terms decrease')
        else:
            require(H*(1-2*t) > 1, f'low-scale derivative sign j={j}')
            z = imin(d, d-2*q+C*2**j/(8*H))
            F = 1+2*(PI*th).sqrt()*(t*H+z).exp()
        ws.append(F*F*(1-H).exp()*2**(2*j+4))
    alpha8 = (512/C/256).sqrt()*11*LOG2
    beta2 = (alpha8+1/H)**2/32
    require(H*(1-2*beta2) > 5, 'all omitted-scale derivatives negative')
    Ft = 1+2*(PI*beta2*H).sqrt()*(beta2*H).exp()
    omitted = I(Q(4, 3))*(1024*H*H/C)**2*(1-H).exp()*Ft*Ft
    upper = sum((w/(w+margins(j)[0]) for j, w in enumerate(ws)), I(0))+omitted
    heavy = (-31*H-1).exp()+(-C).exp()
    finite = imax(heavy, omitted/81,
                  *(w/(w+margins(j)[1]) for j, w in enumerate(ws)))
    general = imax(heavy, 2*(1-H).exp(), 2*omitted/81,
                   *(2*w/(2*w+margins(j)[1]) for j, w in enumerate(ws)))
    return dict(upper=upper, lower_finite=finite, lower_general=general,
                joint=imax(upper+finite, general), omitted_scale_sum=omitted,
                rational_inputs=dict(w_upper=[w.upper_fraction() for w in ws],
                                     omitted_upper=omitted.upper_fraction(),
                                     heavy_upper=heavy.upper_fraction(),
                                     occupancy_upper=(2*(1-H).exp()).upper_fraction()))


def taylor_lower(x, degree):
    return sum((x**k/Q(factorial(k)) for k in range(degree+1)), Q())


def elementary_checks():
    require(LOG2 > I(Q(693, 1000)), 'log2 > 693/1000')
    require(LOG2 < I(Q(694, 1000)), 'log2 < 694/1000')
    require(CSTAR > 69 and CSTAR < 70, '69 < Cstar < 70')
    # Four exact Taylor inequalities suffice for the entire counting branch.
    require(taylor_lower(Q(5), 8) > 100, 'exp(5)>100')
    require(taylor_lower(Q(5, 2), 8) > Q(23, 2), 'exp(5/2)>23/2')
    require(taylor_lower(Q(29, 20), 5) > Q(69, 20), 'exp(29/20)>69/20')
    require(taylor_lower(Q(8), 12) > 1000, 'exp(8)>1000')
    require(Q(512, 69) < Q(121, 16), 'sqrt(512/69)<11/4')
    require(Q(11, 4)*3*Q(7, 10)+Q(1, 15) < 6, 'degree < 6L')
    require(64**2 > 2*6**2, 'expectation condition')
    require(Q(64*15**2, 69*1000) < 1, 'b<1 using exp(14)>1000')
    require(Q(256*15**2, 70) > 1, 'J>=0')
    require(69*(2*Q(693, 1000)-1) > 24, 'Poisson cap exponent >24')


def encode(x):
    if isinstance(x, I):
        return x.text()
    if isinstance(x, Q):
        return str(x)
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [encode(v) for v in x]
    return x


def main():
    start = time.monotonic()
    elementary_checks()
    results, all_slabs, all_tails, bands = [], {}, {}, []
    # Rational upper bounds advertised in the publication.
    targets = {
        'star': {15: Q(392089, 10**6), 16: Q(317480, 10**6),
                 20: Q(87747, 10**6)},
        '70': {15: Q(376264, 10**6), 16: Q(282612, 10**6),
               19: Q(90185, 10**6), 20: Q(72224, 10**6)}
    }
    for name, C, symbolic in [('star', CSTAR, True), ('70', I(70), False)]:
        rows = []
        for i in range(150, 320):
            rows.append(complete_slab(C, i))
            if (i+1) % 50 == 0:
                print(f'{name}: certified complete slabs through L={Q(i+1,10)}', flush=True)
        tail = infinite_tail(C, symbolic)
        all_slabs[name], all_tails[name] = rows, tail
        for left in range(15, 32):
            row = {'C': name, 'L_left': left, 'L_right': left+1}
            for key in ['upper', 'lower_finite', 'lower_general', 'joint']:
                row[key] = imax(*(r[key] for r in rows[(left-15)*10:(left-14)*10]))
            bands.append(row)
        for L0, target in targets[name].items():
            selected = rows[(L0-15)*10:]
            peak = {key: imax(I(Q(1, 1000)), tail[key], *(r[key] for r in selected))
                    for key in ['upper', 'lower_finite', 'lower_general', 'joint']}
            for key in ['upper', 'lower_general', 'joint']:
                require(peak[key] < I(target), f'{name}, L0={L0}, {key} < {target}')
            require(target < (Q(2, 5) if L0 == 15 else Q(1, 3)), 'target probability')
            convenient_failure = (Q(2, 5) if L0 == 15 else Q(1, 3))
            if name == 'star' and L0 == 20:
                convenient_failure = Q(9, 100)
            if name == '70' and L0 == 19:
                convenient_failure = Q(1, 10)
            if name == '70' and L0 == 20:
                convenient_failure = Q(2, 25)
            require(target+Q(1, 10**9) < convenient_failure,
                    f'{name}, L0={L0}: capped-run success guarantee')
            if name == '70' and L0 == 20:
                require(target < Q(2, 25), 'single-run failure < 0.08')
            results.append(dict(C=name, L0=L0, target=target, computed=peak))
    amplification = []
    for q, repeats in [(Q(2, 3), 47), (Q(3, 5), 133), (Q(23, 25), 5)]:
        failure = sum((Q(comb(repeats, k))*(1-q)**k*q**(repeats-k)
                       for k in range((repeats+1)//2, repeats+1)), Q())
        require(failure < Q(1, 100), f'exact binomial amplification {q}, {repeats}')
        amplification.append(dict(success=q, repeats=repeats, failure=failure))
    payload = dict(passed=True, arithmetic='exact integers and rationals; dyadic outward enclosures',
                   bits=BITS, checks=len(CHECKS), check_descriptions=CHECKS,
                   C_star=CSTAR, log_two=LOG2, pi=PI, results=results, bands=bands,
                   half_lines=all_tails, complete_slabs=all_slabs, amplification=amplification,
                   source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in ['exact_certificate.py', 'rational_interval.py']},
                   scope='Finite exact-arithmetic obligations of the accompanying analytic proof; not a Lean/Coq kernel proof.')
    (ROOT/'exact_certificate.json').write_text(json.dumps(encode(payload), indent=2)+'\n')
    tex_rows = []
    for left in range(15, 32):
        pair = [b for b in bands if b['L_left'] == left]
        vals = [decimal_ceiling(b['joint'].upper_fraction()) for b in pair]
        tex_rows.append(f'$[{left},{left+1}]$ & {vals[0]} & {vals[1]} \\\\')
    vals = [decimal_ceiling(all_tails[name]['joint'].upper_fraction()) for name in ['star', '70']]
    tex_rows.append(f'$[32,\\infty)$ & {vals[0]} & {vals[1]} \\\\')
    (ROOT/'certificate_table.tex').write_text('\n'.join(tex_rows)+'\n')
    print(f'PASS: {len(CHECKS)} exact obligations; 340 complete slabs; two analytic half-lines.')
    for result in results:
        print(result['C'], result['L0'], decimal_ceiling(result['computed']['joint'].upper_fraction()))
    # Timing is operational metadata, never an input to a mathematical check.
    print(f'Elapsed seconds: {time.monotonic()-start:.1f}')


if __name__ == '__main__':
    main()
