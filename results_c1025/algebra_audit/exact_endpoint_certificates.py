#!/usr/bin/env python3
"""Exact-rational certificates for sufficient constants, no FLINT or floats.

The analytic implications (Taylor lower bounds, monotonicity, e in (2,3))
are proved/explained in the audit report; this is not a formal proof checker.
"""
from fractions import Fraction as Q
from math import factorial
from pathlib import Path
import hashlib
import json


def exp_partial(x, degree):
    return sum((x**k/Q(factorial(k)) for k in range(degree+1)),Q(0))


def main():
    checks=[]
    def certify(name,lhs,rhs):
        assert lhs>rhs,(name,lhs,rhs)
        checks.append(dict(name=name,lhs=str(lhs),relation=">",rhs=str(rhs)))
    # For positive x, e^x exceeds every finite Taylor sum.
    certify("ln(51.25)<4",exp_partial(Q(4),12),Q(205,4))
    certify("ln(100)<5",exp_partial(Q(5),12),Q(100))
    certify("ln(2)<7/10",exp_partial(Q(7,10),4),Q(2))
    certify("sqrt(2)>707/500",Q(2),Q(707,500)**2)
    certify("k/L<3/2",Q(3,2),3*Q(7,10)/Q(707,500)+Q(1,1024))
    certify("small-n exponent>5",Q(1025,20)-1-4,Q(5))
    certify("large-n exponent>5",Q(10**8,2*1024*1025),Q(5))
    # At L=1024, using e<3 and e>2 gives:
    # (128e/3)*1024^4*e^-736 < 128*1024^4/2^736 = 2^-689.
    certify("simultaneous concentration<1/100",Q(1,100),Q(1,2**689))
    certify("large-occupancy bound<1/100",Q(1,100),Q(1,2**1022))
    certify("heavy top-r bound<1/100",Q(1,100),Q(1,2**1023))
    certify("b<1 at L=1024",Q(1),Q(1024**2,16*2**1023))
    certify("explicit tail excess<0.012",Q(12,1000),Q(1,1023)+(1+Q(1,1023))/100)
    certify("unrounded counterexample>0.06",(1-Q(12,1000))/16,Q(6,100))
    certify("rounded counterexample>0.01",(1-Q(12,1000)-Q(2525,10000))/16,Q(1,100))
    # Derivative signs on the entire half-line L=1024+x, x>=0.
    for label,coeffs in [("b derivative",[2-1024,-1]),
                          ("concentration derivative",[Q(4)-Q(23,32)*1024,-Q(23,32)])]:
        assert all(c<0 for c in coeffs)
        checks.append(dict(name=label,polynomial_coefficients=[str(c) for c in coeffs],
                           conclusion="strictly negative for every x>=0"))
    output=dict(passed=True,arithmetic="Python arbitrary-precision integers and Fraction only",
                checks=checks,total_checks=len(checks),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitation="Analytic reductions are supplied by the accompanying proof, not a proof-assistant kernel")
    Path(__file__).with_name('exact_endpoint_certificates.json').write_text(json.dumps(output,indent=2)+"\n")
    print(f"PASS: {len(checks)} exact-rational endpoint/sign certificates; no floating point or FLINT")


if __name__=='__main__':main()
