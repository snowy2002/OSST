"""Independent, optimization-free replay of small-n upper-tail certificates.

Reconstructs all parameters and strict thresholds. The common-event bound is
checked by direct sums of nonnegative monomials on every saved interval,
instead of the generator's Arb polynomial evaluation. The Poisson heavy-count
tail is recomputed through the regularized incomplete gamma function.
"""
from __future__ import annotations

from fractions import Fraction as Q
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
from parameter_search.algorithm import geometry
from paper_reference import certified_integer
from proof_l01 import mgf

OUT = ROOT/'results_proof_l01'
LOWER_BUDGET = Q(1, 10**20)
BITS = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ball(q):
    if isinstance(q, arb):
        return q
    q = Q(q)
    return arb(fmpq(q.numerator, q.denominator))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_optimization(*args, **kwargs):
    raise ArithmeticError('Optimization is forbidden in independent replay')


def split_scales(scales, cut):
    require(cut is None or (type(cut) is int and 0 <= cut <= len(scales)), 'Invalid cut')
    good, bad = [], []
    for scale in scales:
        require(all(c > 0 if ell % 2 else c < 0
                    for ell, c in enumerate(scale.corrections, 1)),
                f'Alternating sign failure j={scale.j}')
        accepted = (scale.j >= cut) if cut is not None else all(
            scale.corrections[ell-1] <= ball(scale.delta)
            for ell in range(3, scale.k+1, 2))
        (good if accepted else bad).append(scale)
        if accepted:
            require(scale.corrections[0] >= ball(scale.delta), 'Singleton monotonicity failure')
    return good, bad


def validate_coverage(document):
    require(document.get('rigorous') is True and document.get('C') == 70
            and document.get('L0') == 1, 'Wrong certificate type or constants')
    rows = document.get('rows')
    require(isinstance(rows, list) and [r.get('n') for r in rows] == list(range(1,74)),
            'n coverage must be exactly 1..73, in order and without duplicates')
    for row in rows:
        n = row['n']
        require(type(row.get('anchor')) is int and row['anchor'] >= 2
                and row['anchor'] % 2 == 0, f'n={n}: anchor must be even and >=2')
        _, scales = geometry('70', 1, n, BITS)
        good, bad = split_scales(scales, row.get('cut'))
        require(row.get('good_scales') == [s.j for s in good], f'n={n}: wrong common scale group')
        require(row.get('bad_scales') == [s.j for s in bad], f'n={n}: wrong MGF scale group')
        stored = row.get('low_scale_certificate')
        require(isinstance(stored, list) and [m.get('j') for m in stored] == [s.j for s in bad],
                f'n={n}: low scales incomplete, duplicated or unordered')
        for scale, m in zip(bad, stored):
            require(m.get('k') == scale.k and Q(m['delta']) == scale.delta,
                    f'n={n}, j={scale.j}: k/delta mismatch')
            require(type(m['t']) is str and type(m['eta']) is str
                    and Q(m['t']) > 0 and Q(m['eta']) > 0, 'Invalid MGF rational parameters')
            require(m.get('passed') is True, 'MGF certificate did not pass')
        common = row.get('common_certificates')
        require(isinstance(common, list) and [c.get('S') for c in common] == list(range(1,n+1)),
                f'n={n}: S coverage must be exactly 1..n')
        for c in common:
            S = c['S']
            require(isinstance(c.get('z'), list) and len(c['z']) == S
                    and all(type(z) is int for z in c['z']), f'n={n},S={S}: wrong threshold vector')
            require(type(c.get('y')) is str and 0 < Q(c['y']) < 1, 'Invalid y')
            require(type(c.get('target')) is str and 0 < Q(c['target']) < Q(1,3), 'Invalid target')
            require(c.get('passed') is True, 'Common certificate did not pass')
    return rows


def leaf_partition(common):
    require(common.get('zero_event') is not True, 'Nonzero partition required')
    R = Q(common['R'])
    require(R > 1, 'Tail split must exceed one')
    leaves = common.get('leaves')
    require(isinstance(leaves, list) and leaves, 'Missing common-event intervals')
    intervals = sorted((Q(a), Q(b)) for a,b in leaves)
    cursor = Q(1)
    for a,b in intervals:
        require(a == cursor and b > a, 'Interval coverage has a gap, overlap or duplicate')
        cursor = b
    require(cursor == R, 'Interval coverage does not end at the tail split')
    return intervals, R


def negative_controls(document):
    outcomes = []
    for test in ('delete_one_n', 'delete_one_S', 'delete_one_scale'):
        changed = dict(document)
        changed['rows'] = list(document['rows'])
        if test == 'delete_one_n':
            del changed['rows'][0]
        else:
            index = 0 if test == 'delete_one_S' else next(
                i for i,r in enumerate(changed['rows']) if r['low_scale_certificate'])
            row = dict(changed['rows'][index])
            changed['rows'][index] = row
            key = 'common_certificates' if test == 'delete_one_S' else 'low_scale_certificate'
            row[key] = list(row[key])[1:]
        try:
            validate_coverage(changed)
        except ValueError as exc:
            outcomes.append(dict(test=test, rejected=True, reason=str(exc)))
        else:
            raise ArithmeticError(f'Negative control accepted: {test}')
    c = next(c for r in document['rows'] for c in r['common_certificates']
             if not c.get('zero_event') and c.get('leaves'))
    changed = dict(c, leaves=c['leaves'][1:])
    try:
        leaf_partition(changed)
    except ValueError as exc:
        outcomes.append(dict(test='delete_one_leaf', rejected=True, reason=str(exc)))
    else:
        raise ArithmeticError('Missing interval leaf accepted')
    return outcomes


def regenerate_z(n, S, good, anchor):
    envelopes = []
    for scale in good:
        q = arb(0)
        qh = arb(0)
        for ell in range(3, scale.k+1, 2):
            q = q.max(scale.corrections[ell-1])
        for ell in range(anchor+1, scale.k+1, 2):
            qh = qh.max(scale.corrections[ell-1])
        u1 = scale.corrections[0]
        require(u1 >= q, 'Threshold envelope is not nondecreasing')
        denominator = 1+q
        slope = (u1-q)/denominator
        intercept = ((S-1)*q + qh - ball(scale.delta)*(S+n))/denominator
        envelopes.append((slope, intercept))
    z = []
    for s in range(S):
        if not envelopes:
            z.append(-1)
            continue
        values = [s*slope+intercept for slope,intercept in envelopes]
        value = values[0]
        for v in values[1:]:
            value = value.max(v)
        z.append(certified_integer(value, 'ceil')-1)
    require(all(a<=b for a,b in zip(z,z[1:])), 'Threshold vector is not monotone')
    return z


@lru_cache(maxsize=262144)
def exp_negative(value):
    return (-ball(value)).exp()


def powers(value, degree):
    result = [arb(1)]
    for _ in range(degree):
        result.append(result[-1]*value)
    return result


def check_common(common, z):
    r = common['S']-1
    y, target = ball(Q(common['y'])), ball(Q(common['target']))
    terms = [(s, math.comb(r,s)*y**(-z[s])) for s in range(r+1) if z[s]>=0]
    if common.get('zero_event') is True:
        require(not terms and common.get('leaves') == [], 'Invalid zero-event shortcut')
        return 0, arb(0)
    require(bool(terms), 'Zero event has unnecessary nonzero certificate')
    intervals, R = leaf_partition(common)
    tail_powers = powers(ball(R)*exp_negative(R), r)
    tail = sum((coefficient*tail_powers[s] for s,coefficient in terms), arb(0))
    require(tail < target, 'Tail bound does not meet target')
    maximum = tail
    for a,b in intervals:
        singleton = ball(a)*exp_negative(a)
        other = 1-(1-y+ball(b))*exp_negative(b)
        require(singleton > 0 and other > 0 and other < 1, 'Invalid endpoint probability')
        single_powers, other_powers = powers(singleton,r), powers(other,r)
        # Independent direct positive sum: no arb_poly, Horner polynomial,
        # or division by h is used for this common-event verification.
        bound = sum((coefficient*single_powers[s]*other_powers[r-s]
                     for s,coefficient in terms), arb(0))
        require(bound < target, f'Endpoint bound failed on [{a},{b}]')
        maximum = maximum.max(bound)
    return len(intervals), maximum


def main():
    start = time.perf_counter()
    path = OUT/'small_certificates.json'
    original_hash = sha(path)
    document = json.loads(path.read_text())
    sources = {name:sha(ROOT/name) for name in (
        'proof_l01/verify_small.py', 'proof_l01/mgf.py',
        'parameter_search/algorithm.py', 'paper_reference.py', 'c7016/algorithm.py',
        'proof_l01/homogenization.txt')}
    mgf.minimize_scalar = no_optimization
    mgf.numerical_scale = no_optimization
    common_count = low_count = leaf_count = 0
    audited = []
    with ctx.workprec(BITS):
        rows = validate_coverage(document)
        negative = negative_controls(document)
        maximum_total = arb(0)
        for row in rows:
            n, anchor = row['n'], row['anchor']
            L, scales = geometry('70',1,n,BITS)
            good, bad = split_scales(scales,row.get('cut'))
            low_parameters = []
            for scale, stored in zip(bad,row['low_scale_certificate']):
                t, eta = Q(stored['t']), Q(stored['eta'])
                certificate = mgf.certify_mgf([arb(-1)]+list(scale.corrections), t, eta,
                                             bits=BITS, R=stored['R'], degree=stored['degree'])
                require(certificate['passed'], f'MGF replay failed n={n},j={scale.j}')
                low_parameters.append(((1+ball(eta)).log(), ball(t)*ball(scale.delta)))
                low_count += 1
            row_max = arb(0)
            support_rows = []
            for common in row['common_certificates']:
                S = common['S']
                z = regenerate_z(n,S,good,anchor)
                require(z == common['z'], f'Strict threshold mismatch n={n},S={S}')
                leaves, endpoint_max = check_common(common,z)
                leaf_count += leaves
                low = sum(((S*logG-td*(n+S)).exp() for logG,td in low_parameters),arb(0))
                heavy_mean = 70*arb(n)/(S*L)
                heavy = heavy_mean.gamma_upper(anchor,regularized=1)
                total = low+ball(Q(common['target']))+heavy
                require(total < ball(Q(1,3)-LOWER_BUDGET),
                        f'Joint probability budget failed n={n},S={S}')
                require(total.overlaps(arb(common['total_bound'])), 'Saved total inconsistent')
                row_max = row_max.max(total)
                support_rows.append(dict(S=S, anchor=anchor, passed=True,
                                         common_interval_leaves=leaves,
                                         common_endpoint_max=endpoint_max.str(55),
                                         heavy_cdf=heavy.str(55), total_upper=total.str(65)))
                common_count += 1
            require(row_max.overlaps(arb(row['union_bound_arb'])), 'Saved row bound inconsistent')
            maximum_total = maximum_total.max(row_max)
            audited.append(dict(n=n, support_count=len(support_rows), passed=True,
                                maximum_upper_bound=row_max.str(65), supports=support_rows))
            print(f'AUDIT n={n}/73, S cases={common_count}, leaves={leaf_count}, '
                  f'max={float(maximum_total):.12f}, {time.perf_counter()-start:.1f}s',flush=True)
        require(maximum_total+ball(LOWER_BUDGET) < ball(Q(331,1000)),
                'Final small-n bound is not below .331')
        result = dict(
            status='PASS', C=70,L0=1,n_range=[1,73],n_count=len(audited),
            support_cases=common_count, pair_count=common_count, low_scale_mgf_cases=low_count,
            common_interval_leaves=leaf_count, arithmetic=f'{BITS}-bit outward Arb',
            optimizer_called=False, optimization_calls_disabled=True,
            anchor_condition='Even integer K>=2; actual choices rechecked per row',
            threshold_reconstruction='Original S, r=S-1, q and qh, strict ceil-1, complete good/bad partition',
            common_verification='Exact leaf partition; independent direct sums of positive monomials at interval endpoints and tail',
            heavy_verification='Regularized upper incomplete gamma = Poisson CDF(K-1)',
            lower_deviation_budget=str(LOWER_BUDGET),
            maximum_upper_tail=maximum_total.str(65),
            maximum_union_bound_arb=maximum_total.str(65),
            maximum_with_lower_budget=(maximum_total+ball(LOWER_BUDGET)).str(65),
            upper_plus_reserved_lower_strictly_below='331/1000',
            negative_controls=negative,rows=audited,
            certificate_sha256=original_hash,source_sha256=sources,
            elapsed_seconds=time.perf_counter()-start,
        )
    require(original_hash == sha(path),'Certificate changed during audit')
    require(all(digest == sha(ROOT/name) for name,digest in sources.items()),
            'A verification source changed during audit')
    (OUT/'small_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in (
        'status','n_count','support_cases','low_scale_mgf_cases',
        'common_interval_leaves','maximum_upper_tail','elapsed_seconds')},indent=2),flush=True)


if __name__ == '__main__':
    main()
