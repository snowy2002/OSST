#!/usr/bin/env python3
"""Exact-polynomial cross-checks and a 512-bit replay of the certificates."""
from fractions import Fraction as Q
from math import comb, factorial
from pathlib import Path
import json
import hashlib
import certify_constants as cert
from flint import arb, ctx

HERE = Path(__file__).resolve().parent
ctx.prec = 512
cert.LN2 = arb(2).log()
cert.PI = arb.pi()
cert.CSTAR = 144*cert.LN2**2
checks = 0


def check(x, message):
    global checks
    assert x, message
    checks += 1


def aa(x):
    x = Q(x)
    return arb(x.numerator)/x.denominator


def add(a, b, scale=1):
    out = a+[Q(0)]*max(0, len(b)-len(a))
    for i, v in enumerate(b):
        out[i] += scale*v
    return out


def shifted_chebyshev(k, v, B):
    """Independent exact recurrence for T_k(1+2v+2s/B)."""
    tm, tn = [Q(1)], [1+2*v, 2/B]
    if k == 0:
        return tm
    for _ in range(1, k):
        twice_z_tn = add([2*(1+2*v)*x for x in tn], [Q(0)]+[4*x/B for x in tn])
        tm, tn = tn, add(twice_z_tn, tm, -1)
    return tn


def main():
    fixture_count = 0
    # Includes degree-one edge cases, coefficients near the cutoff, and shifts
    # with a/b arbitrarily close to the allowed 1/2.
    for k in [1, 2, 3, 4, 7, 8, 9, 16, 31, 64]:
        at_one = shifted_chebyshev(k, Q(0), Q(1))
        for ell in range(1, k+1):
            product = k*k
            for r in range(1, ell):
                product *= k*k-r*r
            formula = Q(4**ell*product, factorial(2*ell))
            check(at_one[ell] == formula, 'exact Chebyshev coefficient identity')
        for v in [Q(1, 10000), Q(1, 10), Q(1)]:
            for B in [Q(7, 3), Q(64), Q(2048)]:
                raw = shifted_chebyshev(k, v, B)
                rho = 1/raw[0]
                weights = [x*factorial(i)*rho for i, x in enumerate(raw)]
                fs = [Q(0)]+[1+(-1)**(i+1)*weights[i] for i in range(1, len(weights))]+[Q(1)]
                total = sum(weights, Q())
                check(max(abs(x) for x in fs) <= total, 'sup-norm bound')
                check(max(fs)-min(fs) <= total, 'range bound')
                t, w = aa(Q(k*k)/B), aa(k*k*v)
                h = t.sqrt()
                coarse = 1+2*(cert.PI*t).sqrt()*t.exp()
                A = aa(B*v)
                refined = 1+2*(cert.PI*t).sqrt()*(t-2*k*aa(v).sqrt()+A).exp()
                check(aa(total) <= coarse, 'integral coarse majorant')
                check(aa(total) <= refined, 'integral refined majorant')
                theta = aa(k)/(aa(B)*(aa(v*(1+v))).sqrt())
                if theta < 1:
                    check(aa(total) <= 1/(1-theta), 'derivative majorant')
                for ell in range(1, k+1):
                    bell = (4*t)**ell*factorial(ell)/factorial(2*ell)
                    majorant = aa(rho)*bell*(2*w/(2*ell+1)).exp()
                    check(aa(weights[ell]) <= majorant, 'individual coefficient majorant')
                    check(bell <= t.exp(), 'factorial envelope')
                fixture_count += 1
    # The original 256-bit saved balls must overlap an independently computed
    # 512-bit result for every displayed error bound and every complete slab.
    for label, C, is_star in [('star', cert.CSTAR, True), ('70', arb(70), False)]:
        old = json.loads((HERE/f'slabs_{label}.json').read_text())
        for i, previous in enumerate(old['slabs'], start=150):
            fresh = cert.slab(C, arb(i)/10, arb(i+1)/10)
            for key in ['upper', 'lower_finite', 'lower_general', 'joint']:
                check(fresh[key].overlaps(arb(previous[key])), '256/512-bit slab overlap')
        fresh = cert.half_line(C, is_star)
        for key in ['upper', 'lower_finite', 'lower_general', 'joint', 'tail']:
            check(fresh[key].overlaps(arb(old['half_line'][key])), '256/512-bit tail overlap')
    # Explicitly check margins and midpoint-product inequality independently.
    for j in range(50):
        delta = Q(1, 2**(j+2)); rho = delta*delta/2
        margin = 15*delta-16*rho*delta-rho-delta
        check(margin == delta*(14-delta/2-8*delta*delta), 'exact power margin')
        check(margin >= Q(107, 8)*delta, 'uniform positive margin')
    result = dict(passed=True, checks=checks, rational_polynomial_fixtures=fixture_count,
                  precision_replay_bits=512, replayed_slabs=340,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Independent exact recurrence and coefficient checks; higher-precision replay shares the analytic certificate formulas. Not formal verification.')
    (HERE/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS: {checks} checks; {fixture_count} exact polynomial fixtures; 340 interval replays at 512 bits.')


if __name__ == '__main__':
    main()
