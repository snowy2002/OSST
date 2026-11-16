"""Independent 512-bit audit of all continuous midrange MGF obligations.

This uses a different interval parameterization and Horner composition for
positive Chebyshev coefficients, then the [0,R] Bernstein + shifted-tail
verifier in mgf.py. It does not use midrange.py's projective transformation or
de Casteljau implementation. All original parameter slabs and ceiling branches
are checked against independently reconstructed sets.
"""
from __future__ import annotations

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

from flint import arb, ctx
from proof_l01.mgf import ball, certify_mgf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, operation):
    result = getattr(value, operation)().unique_fmpz()
    if result is None:
        raise ArithmeticError('Nonunique integer endpoint')
    return int(result)


@lru_cache(maxsize=None)
def exact_positive_coefficients(k):
    # c_0=1; c_(r+1)/c_r=(k^2-r^2)/((r+1)(2r+1)).
    coefficients = [Fraction(1)]
    for r in range(k):
        coefficients.append(coefficients[-1]*Fraction(k*k-r*r, (r+1)*(2*r+1)))
    return tuple(coefficients)


def independent_y_envelopes(left, right, j, k):
    L = ball(left).union(ball(right))
    ratio = ball(Fraction(70*2**j, 512))/L**2
    if not 0 < ratio < 1:
        raise ArithmeticError('Invalid scaled endpoint ratio')
    z, beta = 2*ratio/(1-ratio), 1/(32*L*(1-ratio))
    c = exact_positive_coefficients(k)
    # Positive Horner composition T_k(1+z+beta*x).
    poly = [ball(c[-1])]
    for coefficient in reversed(c[:-1]):
        following = [arb(0) for _ in range(len(poly)+1)]
        for ell, value in enumerate(poly):
            following[ell] += z*value
            following[ell+1] += beta*value
        following[0] += ball(coefficient)
        poly = following
    if not poly[0] > 0:
        raise ArithmeticError('Nonpositive normalizer')
    ys = [arb(-1)]
    for ell in range(1, k+1):
        u = math.factorial(ell)*poly[ell]/poly[0]
        if not u > 0:
            raise ArithmeticError('Nonpositive coefficient magnitude')
        ys.append((-1)**(ell+1)*u)
    return ys


def main():
    started = time.monotonic()
    source = ROOT/'results_proof_l01/midrange_certificate.json'
    original = json.loads(source.read_text())
    if not original['passed']:
        raise ArithmeticError('Original certificate did not pass')
    if digest(ROOT/'proof_l01/midrange.py') != original['source_sha256']:
        raise ArithmeticError('Original source hash mismatch')
    slabs = original['parameter_slabs']
    if len(slabs) != 700 or original['coefficient_cases'] != 6597:
        raise ArithmeticError('Unexpected coverage size')
    audited = []
    obligations = 0
    with ctx.workprec(512):
        for index, slab in enumerate(slabs):
            left, right = Fraction(slab['left']), Fraction(slab['right'])
            if (left, right) != (Fraction(700+index, 100), Fraction(701+index, 100)):
                raise ArithmeticError('Slabs do not exactly partition [7,14]')
            N = integer((ball(left)-1).exp(), 'floor')
            if not N <= (ball(left)-1).exp():
                raise ArithmeticError('N exceeds the lower real n endpoint')
            J = integer((ball(Fraction(256,70))*ball(right)**2).log()/arb(2).log(), 'floor')
            if not 0 <= J <= 9:
                raise ArithmeticError('Number of scales exceeds the union bound')
            expected = set()
            for j in range(J+1):
                alpha = ball(Fraction(512, 70*2**j)).sqrt() * ((j+3)*arb(2).log())
                first = integer(alpha*ball(left), 'ceil')
                last = integer(alpha*ball(right), 'ceil')
                expected.update((j, k) for k in range(first, last+1))
            actual = {(row['j'], row['k']) for row in slab['scales']}
            if actual != expected or len(actual) != len(slab['scales']):
                raise ArithmeticError('Ceiling branches incomplete or duplicated')
            checked = []
            for row in slab['scales']:
                j, k = row['j'], row['k']
                t, eta = Fraction(row['t']), Fraction(row['eta'])
                delta = Fraction(1, 2**(j+2))
                if row['N'] != N or t*delta != eta or eta*N != 4:
                    raise ArithmeticError('Wrong Chernoff parameters')
                ys = independent_y_envelopes(left, right, j, k)
                cert = certify_mgf(ys, t, eta, bits=512, R=8, degree=max(k,48))
                if not cert['passed']:
                    cert = certify_mgf(ys, t, eta, bits=512, R=16, degree=max(k,72))
                if not cert['passed']:
                    raise ArithmeticError((str(left), j, k, cert))
                checked.append(dict(j=j, k=k, N=N, t=str(t), eta=str(eta), certificate=cert))
                obligations += 1
            audited.append(dict(left=str(left), right=str(right), scales=checked))
            if (index+1) % 50 == 0:
                print(f'Independently audited {index+1}/700 slabs, {obligations} cases, '
                      f'{time.monotonic()-started:.1f}s', flush=True)
        risk = 10*arb(-4).exp()
        assert risk < ball(Fraction(32,100))
        result = dict(
            passed=True, C0=70, L0=1, L_interval=[7,14],
            parameter_slabs=700, coefficient_cases=obligations,
            integer_n_lower_endpoint=403, maximum_number_of_scales=10,
            arithmetic='Independent 512-bit outward Arb calculations',
            independent_components=[
                'Coefficient magnitudes via rational coefficient recurrence and positive Horner composition',
                'Scaled ratio A/B=70*2^j/(512*L^2)',
                'Exact rational slab partition and complete floor/ceil branch reconstruction',
                'MGF certificate on [0,R] by power-to-Bernstein conversion',
                'MGF certificate on [R,infinity) by nonnegative shifted polynomial coefficients',
            ],
            risk_upper_interval=risk.str(65), slabs=audited,
            elapsed_seconds=time.monotonic()-started,
            original_certificate_sha256=digest(source),
            source_sha256={name:digest(ROOT/name) for name in (
                'proof_l01/audit_midrange.py', 'proof_l01/mgf.py')},
        )
    output = ROOT/'results_proof_l01/midrange_independent_audit.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in (
        'passed','parameter_slabs','coefficient_cases','risk_upper_interval','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
