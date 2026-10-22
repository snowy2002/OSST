#!/usr/bin/env python3
"""Independent exact-Fraction audit of primitives, identities and certificate risks.

This complements, and does not replace, the analytic proofs in the manuscript.
Uses only the Python standard library and the bundled arithmetic module.
"""
from fractions import Fraction as Q
from math import factorial
from pathlib import Path
import hashlib
import json
import rational_interval as ri

ROOT = Path(__file__).resolve().parent
COUNTS = {}


def require(condition, category):
    if not condition:
        raise ArithmeticError(category)
    COUNTS[category] = COUNTS.get(category, 0)+1


def contains(interval, lower, upper):
    return interval.lower_fraction() <= lower <= upper <= interval.upper_fraction()


def corners(interval):
    return [interval.lower_fraction(), interval.upper_fraction()]


def reference_exp(x):
    """Independent unreduced 400-term Taylor enclosure, not the engine algorithm."""
    if x < 0:
        lower, upper = reference_exp(-x)
        return 1/upper, 1/lower
    n = 400
    term = total = Q(1)
    for k in range(1, n+1):
        term *= x/k
        total += term
    tail = (term*x/(n+1))/(1-x/(n+2))
    return total, total+tail


def check_arithmetic():
    inputs = [ri.I.enclose(a, b) for a, b in
              [(Q(-7, 3), Q(-1, 4)), (Q(-1, 2), Q(1, 3)), (Q(0), Q(0)),
               (Q(1, 7), Q(5, 3)), (Q(1, 2**200), Q(1, 2**180)),
               (Q(10**25), Q(10**25)+Q(1, 3)), (Q(1, 3), Q(1, 3))]]
    for a in inputs:
        al, ah = corners(a)
        require(contains(-a, -ah, -al), 'negation enclosure')
        for exponent in range(8):
            values = [x**exponent for x in [al, ah]]
            if al <= 0 <= ah:
                values.append(Q(0)**exponent)
            require(contains(a**exponent, min(values), max(values)), 'power enclosure')
        if al >= 0:
            s = a.sqrt()
            require(s.lower_fraction()**2 <= al, 'sqrt lower witness')
            require(s.upper_fraction()**2 >= ah, 'sqrt upper witness')
        for b in inputs:
            bl, bh = corners(b)
            require(contains(a+b, al+bl, ah+bh), 'addition enclosure')
            require(contains(a-b, al-bh, ah-bl), 'subtraction enclosure')
            products = [x*y for x in [al, ah] for y in [bl, bh]]
            require(contains(a*b, min(products), max(products)), 'multiplication enclosure')
            if not bl <= 0 <= bh:
                quotients = [x/y for x in [al, ah] for y in [bl, bh]]
                require(contains(a/b, min(quotients), max(quotients)), 'division enclosure')
            require(contains(ri.imin(a,b), min(al,bl), min(ah,bh)), 'minimum enclosure')
            require(contains(ri.imax(a,b), max(al,bl), max(ah,bh)), 'maximum enclosure')
    for x in [Q(-40), Q(-7,3), Q(-1,8), Q(0), Q(1,8), Q(1), Q(5,2), Q(8), Q(40)]:
        low, high = reference_exp(x)
        require(contains(ri.I(x).exp(), low, high), 'independent Taylor containment')
    z = Q(1,3)
    low = 2*sum((z**(2*k+1)/(2*k+1) for k in range(120)), Q())
    high = low+2*z**241/(241*(1-z*z))
    require(contains(ri.log_two(), low, high), 'independent log2 containment')
    def atan90(z):
        low = sum(((-1)**k*z**(2*k+1)/(2*k+1) for k in range(90)), Q())
        return low, low+z**181/181
    a, b = atan90(Q(1,5)), atan90(Q(1,239))
    require(contains(ri.pi_interval(), 16*a[0]-4*b[1], 16*a[1]-4*b[0]), 'independent pi containment')
    # Negative controls: unsafe operations must fail, even with python -O.
    for operation in [lambda:ri.I(0).reciprocal(), lambda:ri.I(-1).sqrt(),
                      lambda:ri.I.raw(2,1), lambda:ri.I(0.1)]:
        try:
            operation()
        except (ValueError, TypeError, ZeroDivisionError):
            require(True, 'negative control rejected')
        else:
            raise ArithmeticError('negative control unexpectedly accepted')
    require(ri.decimal_ceiling(Q(-1,10**7)) == '0.000000', 'decimal upward rounding')
    require(ri.decimal_ceiling(Q(-13,10)) == '-1.300000', 'decimal upward rounding')


def poly_add(a, b, factor=1):
    result = a+[Q(0)]*max(0,len(b)-len(a))
    for i, x in enumerate(b):
        result[i] += factor*x
    return result


def chebyshev(k, v, B):
    old, current = [Q(1)], [1+2*v,2/B]
    for _ in range(1,k):
        twice_z_current = poly_add([2*(1+2*v)*x for x in current],
                                   [Q(0)]+[4*x/B for x in current])
        old, current = current, poly_add(twice_z_current,old,-1)
    return current


def check_polynomials():
    for k in [1,2,3,4,7,8,9,16,31,64]:
        coefficients = chebyshev(k,Q(0),Q(1))
        for ell in range(1,k+1):
            product = k*k
            for r in range(1,ell):
                product *= k*k-r*r
            require(coefficients[ell] == Q(4**ell*product,factorial(2*ell)),
                    'exact coefficient recurrence identity')
        for v in [Q(1,10000),Q(1,10),Q(1)]:
            for B in [Q(7,3),Q(64),Q(2048)]:
                raw = chebyshev(k,v,B)
                weights = [c*factorial(ell)/raw[0] for ell,c in enumerate(raw)]
                fs = [Q(0),Q(1)]+[1+(-1)**(ell+1)*weights[ell] for ell in range(1,k+1)]
                require(max(abs(f) for f in fs) <= sum(weights), 'exact weight norm bound')
                require(max(fs)-min(fs) <= sum(weights), 'exact weight range bound')
    for j in range(50):
        d = Q(1,2**(j+2)); rho = d*d/2
        margin = 15*d-16*rho*d-rho-d
        require(margin == d*(14-d/2-8*d*d), 'power margin identity')


def margins(j):
    delta = Q(1,2**(j+2))
    return (2-delta/2)**2, (14-delta/2-8*delta**2)**2


def fraction_risk(row, half_line=False):
    data = row['rational_inputs']
    ws = [Q(s) for s in data['w_upper']]
    vs = ws if half_line else [Q(s) for s in data['v_upper']]
    require(len(vs)==len(ws), 'risk vector alignment')
    h, o = Q(data['heavy_upper']), Q(data['occupancy_upper'])
    upper = sum((v/(v+margins(j)[0]) for j,v in enumerate(vs)), Q())
    finite = max(h, *(v/(v+margins(j)[1]) for j,v in enumerate(vs)))
    general = max(h,o,*(2*w/(2*w+margins(j)[1]) for j,w in enumerate(ws)))
    if half_line:
        omitted = Q(data['omitted_upper'])
        upper += omitted
        finite = max(finite,omitted/81)
        general = max(general,2*omitted/81)
    result = dict(upper=upper,lower_finite=finite,lower_general=general,
                  joint=max(upper+finite,general))
    for key,value in result.items():
        require(value <= Q(row[key]['upper']), 'independent Fraction risk aggregation')
    return result


def check_certificate():
    certificate = json.loads((ROOT/'exact_certificate.json').read_text())
    for name, expected in certificate['source_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, 'certificate source hash')
    manuscript = (ROOT/'submission_proof.tex').read_text()
    for line in (ROOT/'certificate_table.tex').read_text().splitlines():
        require(manuscript.count(line)==1, 'manuscript table matches exact certificate')
    for name in ['star','70']:
        rows = certificate['complete_slabs'][name]
        require([r['slab_numerator'] for r in rows]==list(range(150,320)), 'complete interval coverage')
        fraction_rows = [fraction_risk(r) for r in rows]
        tail = fraction_risk(certificate['half_lines'][name],half_line=True)
        for result in [r for r in certificate['results'] if r['C']==name]:
            target = Q(result['target']); start = (result['L0']-15)*10
            for key in ['upper','lower_general','joint']:
                worst = max(Q(1,1000),tail[key],*(r[key] for r in fraction_rows[start:]))
                require(worst < target, 'independent theorem threshold')
    return certificate


def main():
    check_arithmetic()
    check_polynomials()
    certificate = check_certificate()
    result = dict(passed=True,checks=sum(COUNTS.values()),categories=COUNTS,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  certificate_sha256=hashlib.sha256((ROOT/'exact_certificate.json').read_bytes()).hexdigest(),
                  certificate_obligations=certificate['checks'],
                  scope='Exact primitive/identity checks plus independent rational risk aggregation. Not a foundational proof-assistant formalization.')
    (ROOT/'audit_exact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {result["checks"]} exact audit checks; {certificate["checks"]} certificate obligations.')


if __name__=='__main__':
    main()
