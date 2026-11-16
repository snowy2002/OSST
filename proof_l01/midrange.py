"""Arb certificate for all upper tails on 7 <= L <= 14, C=70.

For every scale choose t=4/(delta*N), where the exact integer N is no larger
than exp(L-1) throughout a slab. Certify E exp(t(f(Poi(lambda))-1)) <= 1+t*delta
for every lambda>=0. Markov then gives exp(-4) per scale, at most 10 scales.

The finite MGF condition follows from a globally nonnegative polynomial:
  (t*delta) sum_{l=0}^k lambda^l/l! - sum_{l=0}^k (exp(t*y_l)-1) lambda^l/l!.
We certify its nonnegativity after mapping lambda=x/(1-x), using Bernstein
subdivision with outward Arb intervals. The positive Chebyshev expansion
covers whole L slabs and all possible ceiling degrees, not only grid points.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flint import arb, ctx, fmpq


def ball(value):
    q = Fraction(value)
    return arb(fmpq(q.numerator, q.denominator))


def hull(lo, hi):
    return arb.union(ball(lo), ball(hi))


def exact_integer(value, operation):
    result = getattr(value, operation)().unique_fmpz()
    if result is None:
        raise ArithmeticError(f'Integer endpoint ambiguous: {value}')
    return int(result)


@lru_cache(maxsize=None)
def positive_coefficients(k):
    return tuple(ball(Fraction(k * 2**r * math.factorial(k+r-1),
                               math.factorial(k-r) * math.factorial(2*r)))
                 for r in range(k+1))


def correction_envelopes(left, right, j, k):
    L = hull(left, right)
    A, B = 70*2**j/(8*L), 64*L
    if not 0 < A < B:
        raise ArithmeticError('Nonpositive affine interval')
    z, beta = 2*A/(B-A), 2/(B-A)
    c = positive_coefficients(k)
    powers = [z**r for r in range(k+1)]
    denominator = sum((c[r]*powers[r] for r in range(k+1)), arb(0))
    if not denominator > 0:
        raise ArithmeticError('Normalization has no certified positive lower bound')
    corrections = [arb(-1)]
    for ell in range(1, k+1):
        derivative = sum((c[r]*math.comb(r, ell)*powers[r-ell]
                          for r in range(ell, k+1)), arb(0))
        u = beta**ell*math.factorial(ell)*derivative/denominator
        if not u > 0:
            raise ArithmeticError('Positive coefficient not certified')
        corrections.append((-1)**(ell+1)*u)
    return corrections


def subdivide(coefficients):
    """Bernstein coefficients on the two equal halves, via de Casteljau."""
    level = list(coefficients)
    left, right = [level[0]], [level[-1]]
    while len(level) > 1:
        level = [(level[i]+level[i+1])/2 for i in range(len(level)-1)]
        left.append(level[0])
        right.append(level[-1])
    return tuple(left), tuple(reversed(right))


def certify_nonnegative(coefficients, max_nodes=10000, max_depth=40):
    """True proves a polynomial nonnegative everywhere on [0,1]."""
    pending = deque([(tuple(coefficients), 0)])
    nodes = 0
    deepest = 0
    while pending:
        values, depth = pending.popleft()
        nodes += 1
        deepest = max(deepest, depth)
        if all(value >= 0 for value in values):
            continue
        if values[0] < 0 or values[-1] < 0:
            return False, nodes, deepest, 'negative endpoint'
        if depth >= max_depth or nodes >= max_nodes:
            return False, nodes, deepest, 'subdivision resource limit'
        a, b = subdivide(values)
        pending.append((a, depth+1))
        pending.append((b, depth+1))
    return True, nodes, deepest, None


def certify_scale(left, right, j, k, N):
    delta = Fraction(1, 2**(j+2))
    t = ball(Fraction(4, 1)/(delta*N))
    eta = ball(Fraction(4, N))
    y = correction_envelopes(left, right, j, k)
    a_upper = [(t*v).expm1().upper() for v in y]
    # q_l/binom(k,l) = (eta-a_l)*(k-l)!/k! are the Bernstein
    # coefficients of (1-x)^k Q(x/(1-x)).
    coefficients = [(eta-a_upper[ell]) * ball(Fraction(math.factorial(k-ell),
                                                      math.factorial(k)))
                    for ell in range(k+1)]
    ok, nodes, depth, reason = certify_nonnegative(coefficients)
    return dict(j=j, k=k, N=N, t=str(Fraction(4, 1)/(delta*N)),
                eta=str(Fraction(4, N)), passed=ok, nodes=nodes,
                depth=depth, reason=reason)


def certify_slab(left, right):
    N = exact_integer((ball(left)-1).exp(), 'floor')
    jmax = exact_integer((256*ball(right)**2/70).log()/arb(2).log(), 'floor')
    rows = []
    for j in range(jmax+1):
        alpha = (ball(Fraction(512, 70*2**j))).sqrt()*(j+3)*arb(2).log()
        lo = exact_integer(alpha*ball(left), 'ceil')
        hi = exact_integer(alpha*ball(right), 'ceil')
        for k in range(lo, hi+1):
            row = certify_scale(left, right, j, k, N)
            rows.append(row)
            if not row['passed']:
                return dict(left=str(left), right=str(right), passed=False, scales=rows)
    return dict(left=str(left), right=str(right), passed=True, scales=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denominator', type=int, default=100)
    parser.add_argument('--left', type=int, default=7)
    parser.add_argument('--right', type=int, default=14)
    parser.add_argument('--bits', type=int, default=256)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results_proof_l01/midrange_certificate.json')
    args = parser.parse_args()
    if not (7 <= args.left < args.right <= 14):
        raise ValueError('This certificate is restricted to subintervals of [7,14]')
    if args.denominator < 1 or args.bits < 128:
        raise ValueError('Positive subdivision denominator and at least 128 bits required')
    start = time.monotonic()
    slabs = []
    with ctx.workprec(args.bits):
        for q in range(args.left*args.denominator, args.right*args.denominator):
            slab = certify_slab(Fraction(q, args.denominator),
                                Fraction(q+1, args.denominator))
            slabs.append(slab)
            if not slab['passed']:
                print('FAILED', json.dumps(slab), flush=True)
                break
            if len(slabs) % 25 == 0:
                print(f'Certified through L={slab["right"]}; '
                      f'{time.monotonic()-start:.1f}s', flush=True)
        risk = 10*arb(-4).exp()
        assert all(row['j'] <= 9 for slab in slabs for row in slab['scales'])
        payload = dict(
            passed=all(s['passed'] for s in slabs) and len(slabs)==
                   (args.right-args.left)*args.denominator,
            scope=f'Upper-tail union for C=70, real mean, {args.left}<=L<={args.right}, arbitrary support S<=n and arbitrary probabilities',
            C0=70, L0=1, L_interval=[args.left, args.right],
            subdivision_denominator=args.denominator, maximum_number_of_scales=10,
            arithmetic=f'Arb outward intervals at {args.bits} bits',
            analytic_obligation='Positive Chebyshev coefficient envelopes; exp truncation lower bound; Bernstein convex hull subdivision; Markov with t*delta*N=4',
            risk_upper_interval=risk.str(65),
            check_risk_below_032=bool(risk < ball(Fraction(32,100))),
            parameter_slabs=slabs,
            coefficient_cases=sum(len(s['scales']) for s in slabs),
            elapsed_seconds=time.monotonic()-start,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('passed','coefficient_cases',
                                          'risk_upper_interval','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
