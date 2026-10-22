"""Supplementary proof audit; does not overwrite the frozen certificates.

Run: .venv/bin/python -m parameter_search.review_proof
Checks omitted by the earlier independent numerical audit: complete domains,
scale/degree coverage, monotonicity premises, counting branch, and aggregation.
Also checks the critical root via a different, positive atanh series.
This is an arithmetic audit, not a machine-checked proof of the probability theory.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import math

from flint import arb, ctx, fmpq
from parameter_search import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_proof_review'
COUNTS = {}


def check(condition, category):
    if not bool(condition):
        raise ArithmeticError(f'Proof review failed: {category}')
    COUNTS[category] = COUNTS.get(category, 0) + 1


def ball(value):
    value = Q(str(value))
    return arb(fmpq(value.numerator, value.denominator))


def alternate_gamma(C):
    """Use asinh(x)=2 atanh(x/(1+sqrt(1+x*x))), with positive terms.

    The tail after terms 0..N-1 is <=
      2*z**(2*N+1)/((2*N+1)*(1-z*z)).
    This is separate from lower_bound.py's alternating asinh implementation.
    """
    x = (ball('9/2') * arb(2).log()**2 / ball(C)).sqrt()
    z = x / (1 + (1 + x*x).sqrt())
    check(0 < z and z < 1, 'positive_series_domain')
    N = 80
    partial = 2 * sum((z**(2*k+1)/(2*k+1) for k in range(N)), arb(0))
    remainder = 2*z**(2*N+1)/((2*N+1)*(1-z*z))
    lower = 32*(x*partial - (1+x*x).sqrt() + 1)
    upper = lower + 32*x*remainder
    return lower, upper


def structural_audit(data):
    check(set(data['complete_slabs']) == {'star', '70'}, 'constant_set')
    check(set(data['half_lines']) == {'star', '70'}, 'constant_set')
    result_keys = [(r['C'], r['L0']) for r in data['results']]
    expected = {(c, l) for c in ['star', '70'] for l in [14, 15, 16, 18]}
    check(len(result_keys) == len(expected) and set(result_keys) == expected,
          'result_set_no_duplicates')
    for name, rows in data['complete_slabs'].items():
        C = 144*arb(2).log()**2 if name == 'star' else arb(70)
        intervals = [(Q(r['left']), Q(r['right'])) for r in rows]
        check(intervals == [(Q(i, 10), Q(i+1, 10)) for i in range(140, 400)],
              'continuous_domain_no_gaps_or_duplicates')
        tail = data['half_lines'][name]
        check(tail['H'] == 40, 'half_line_meets_slabs')
        check(69 < C and C <= 70, 'parameter_lemma')
        check(256*arb(14)**2/C > 1, 'parameter_lemma')
        check(64*arb(14)**2/(C*arb(13).exp()) < 1, 'parameter_lemma')
        check((512/C).sqrt()*3*arb(2).log()+ball('1/14') < 6,
              'parameter_lemma')
        for row in rows:
            left, right = ball(row['left']), ball(row['right'])
            # Length defines the saved scale set; prove it includes all active scales.
            length = len(row['inputs']['v'])
            check(length == len(row['inputs']['w']) == len(row['inputs']['integral']),
                  'scale_vector_lengths')
            check(arb(2)**length > 256*right**2/C, 'all_active_scales_included')
            check(arb(2)**(length-1) <= 256*right**2/C, 'last_scale_not_accidentally_extra')
            for j in range(length):
                delta = ball(Q(1, 2**(j+2)))
                check(2-delta/2 > 1 and 14-delta/2-8*delta**2 > 9,
                      'tail_margin_constants')
                # k(L)=ceil(alpha*L) <= alpha*right+1, including integer crossings.
                alpha = (512/C/2**j).sqrt()*(j+3)*arb(2).log()
                check(alpha > 0 and left <= right, 'degree_ceiling_envelope')
        H = arb(40)
        for j in range(8):
            # For star, use the exact cancellation, including 2*t_0=1.
            t = ball(Q((j+3)**2, 18*2**j)) if name == 'star' else (
                8*((j+3)*arb(2).log())**2/(C*2**j))
            check(C*2**j/(512*H**2) < ball('1/2'), 'half_line_shift')
            if j == 0:
                check(t == ball('1/2') if name == 'star' else 2*t < 1,
                      'scale_zero_decay')
                check(H >= 14, 'scale_zero_polynomial_decay')
            else:
                check(H*(1-2*t) > 1, 'low_scale_decay')
        beta2 = ((512/C/256).sqrt()*11*arb(2).log()+1/H)**2/32
        check(H*(1-2*beta2) > 5, 'omitted_scale_decay')


def counting_and_aggregation(data):
    rows_out = []
    for result in data['results']:
        name, L0 = result['C'], result['L0']
        C = 144*arb(2).log()**2 if name == 'star' else arb(70)
        saved_count = Q(result['counting']['upper'])
        count_bounds = []
        for i in range(4, 4*L0):
            left, right = ball(Q(i, 4)), ball(Q(i+1, 4))
            x = C/right
            check(x > 1, 'counting_rate_positive')
            bound = (-(left-1).exp()*(x-1)**2/(2*x)).exp()
            check(bound <= ball(saved_count), 'counting_recomputed_bound')
            count_bounds.append(bound)
        check(ball(saved_count) < ball('1/10000000000'), 'counting_target')
        selected = [r for r in data['complete_slabs'][name] if Q(r['left']) >= L0]
        check(len(selected) == (40-L0)*10, 'selected_domain')
        for key in ['upper', 'lower_finite', 'lower_general', 'joint']:
            exact_max = max([saved_count, Q(data['half_lines'][name][key]['upper'])] +
                            [Q(r[key]['upper']) for r in selected])
            check(exact_max == Q(result['computed'][key]['upper']), 'exact_aggregation')
        joint = Q(result['computed']['joint']['upper'])
        check(joint < Q(result['target']) < Q('1/100'), 'advertised_target')
        rows_out.append(dict(C=name, L0=L0, counting_upper=float(saved_count),
                             joint_upper=float(joint), target=result['target']))
    return rows_out


def root_audit():
    roots = {}
    for C, sign in [('69', 1), ('69.00627138', 1), ('69.00627139', -1), ('70', -1)]:
        lower, upper = alternate_gamma(C)
        check(lower > ball('1/2') if sign == 1 else upper < ball('1/2'),
              'independent_critical_root_sign')
        roots[C] = dict(lower=str(lower), upper=str(upper))
    return roots


def main():
    data = json.loads((ROOT/'results_parameter_search/certificate.json').read_text())
    with ctx.workprec(512):
        structural_audit(data)
        results = counting_and_aggregation(data)
        roots = root_audit()
        # Reuse the pre-existing separate Arb recomputation of all row formulas.
        audit.COUNT = 0
        check(audit.certificate_audit() == 520, 'existing_arb_formula_audit')
    OUT.mkdir(exist_ok=True)
    sources = ['parameter_search/review_proof.py', 'parameter_search/audit.py',
               'results_parameter_search/proof.tex',
               'results_parameter_search/certificate.json',
               'results_parameter_search/lower_bound_certificate.json',
               'artifacts/joint_constants_submission/submission_proof.tex']
    payload = dict(passed=True, supplementary_checks=sum(COUNTS.values()),
                   categories=COUNTS, existing_arb_formula_checks=audit.COUNT,
                   results=results, independent_gamma_brackets=roots,
                   source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest()
                                  for s in sources},
                   scope='Arithmetic and coverage audit. Analytic inequalities and '
                         'asymptotic probability arguments are reviewed in review.tex; '
                         'no claim of proof-assistant formal verification.')
    (OUT/'audit.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k: payload[k] for k in ['passed', 'supplementary_checks',
                     'existing_arb_formula_checks', 'results']}, indent=2))


if __name__ == '__main__':
    main()
