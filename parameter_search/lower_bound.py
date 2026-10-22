"""Exact sign certificate for the asymptotic necessary sampling constant.

The mathematical reduction to this scalar inequality is in proof.tex.
This program proves the scalar brackets, not the CLT argument by itself.
"""
from fractions import Fraction as Q
from math import comb
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/joint_constants_submission'))
from rational_interval import I,log_two


def require(condition):
    if not condition:
        raise ArithmeticError('Necessary-bound certificate failed')


def asinh_small(x):
    if not (x>0 and x<1):raise ValueError('Alternating series requires 0<x<1')
    # Alternating binomial series: even final index gives an upper partial
    # sum, odd final index a lower partial sum. Interval arithmetic encloses
    # each partial sum for every x in its input interval.
    even=I(0)
    for k in range(81):
        even+=(-1)**k*Q(comb(2*k,k),4**k*(2*k+1))*x**(2*k+1)
    odd=even-Q(comb(162,81),4**81*163)*x**163
    return I.enclose(odd.lower_fraction(),even.upper_fraction())


def gamma(C):
    x=(I(Q(9,2))*log_two()**2/I(C)).sqrt()
    return 32*(x*asinh_small(x)-(1+x*x).sqrt()+1)


def main():
    lo,hi=Q('69.00627138'),Q('69.00627139')
    values={str(c):gamma(c) for c in [Q(69),lo,hi,Q(70)]}
    require(values['69']>I(Q(1,2)))
    require(values[str(lo)]>I(Q(1,2)) and values[str(hi)]<I(Q(1,2)))
    require(values['70']<I(Q(1,2)))
    # Independent transcendental implementation, much higher precision.
    from flint import arb,ctx,fmpq
    with ctx.workprec(512):
        for name,v in values.items():
            c=Q(name);C=arb(fmpq(c.numerator,c.denominator))
            alpha=(512/C).sqrt()*3*arb(2).log()
            g=alpha*(alpha/32).asinh()-(1024+alpha**2).sqrt()+32
            lower,upper=v.lower_fraction(),v.upper_fraction()
            require(g>=arb(fmpq(lower.numerator,lower.denominator)))
            require(g<=arb(fmpq(upper.numerator,upper.denominator)))
    sources=['parameter_search/lower_bound.py',
             'artifacts/joint_constants_submission/rational_interval.py']
    payload=dict(passed=True,necessary_C_lower=str(lo),necessary_C_upper=str(hi),
                 gamma_brackets={s:v.text() for s,v in values.items()},
                 meaning='For every C<Ccrit and fixed finite L0>1, the unchanged single-run tester cannot have uniform error <1%. For 1<=C<Ccrit a null sequence has liminf false rejection >=1/2; C<=1 fails in the counting branch. Fixed odd majority repetition cannot repair either uniform obstruction.',
                 precision='Exact rational alternating series and 160-bit dyadic bounds; independently checked with 512-bit Arb.',
                 source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (ROOT/'results_parameter_search/lower_bound_certificate.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS: 69.00627138 < Ccrit < 69.00627139; C=69 is excluded by the accompanying asymptotic proof.')


if __name__=='__main__':main()
