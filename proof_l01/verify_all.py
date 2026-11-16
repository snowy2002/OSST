"""Replay and aggregate the complete C=70,L0=1 sufficient-constant proof.

Default: recompute every numerical obligation, without numerical optimizers.
--aggregate-only: assemble existing audited certificates (no new replay claim).
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from flint import arb, ctx, fmpq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_proof_l01'


def ball(value):
    v=Fraction(value)
    return arb(fmpq(v.numerator,v.denominator))


def require(ok,message):
    if not bool(ok):
        raise ArithmeticError(message)


def read(name):
    return json.loads((OUT/name).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basic_and_large_audit():
    from real_l01.theory_review import parameter_checks,lower_tail_checks
    from parameter_search import audit,review_proof
    with ctx.workprec(512):
        basic=dict(parameters=parameter_checks(),lower=lower_tail_checks())
        require(arb(basic['lower']['certified_failure_upper_bound'])<ball('1.657e-27'),
                'Small-range lower-tail target')
        old=json.loads((ROOT/'results_parameter_search/certificate.json').read_text())
        review_proof.COUNTS.clear()
        review_proof.structural_audit(old)
        rows=review_proof.counting_and_aggregation(old)
        audit.COUNT=0
        slabs=audit.certificate_audit()
        large=dict(passed=True,independent_arithmetic='512-bit Arb',
                   finite_slabs_checked=slabs,formula_checks=audit.COUNT,
                   coverage_checks=review_proof.COUNTS,results=rows,
                   old_certificate_sha256=digest(ROOT/'results_parameter_search/certificate.json'))
    basic['source_sha256']={p:digest(ROOT/p) for p in
        ['real_l01/theory_review.py','parameter_search/algorithm.py','proof_l01/verify_all.py']}
    large['source_sha256']={p:digest(ROOT/p) for p in
        ['parameter_search/audit.py','parameter_search/review_proof.py']}
    (OUT/'basic_certificate.json').write_text(json.dumps(basic,indent=2)+'\n')
    (OUT/'large_range_independent_audit.json').write_text(json.dumps(large,indent=2)+'\n')


def aggregate(full_replay):
    small=read('small_certificates.json')
    require([r['n'] for r in small['rows']]==list(range(1,74)), 'Exact small-n coverage')
    require(small['all_passed'] and all(r['passed'] for r in small['rows']), 'Small certificates passed')
    for r in small['rows']:
        require([p['S'] for p in r['common_certificates']]==list(range(1,r['n']+1)), 'Every support size')
    small_audit=read('small_audit.json')
    require(small_audit.get('status')=='PASS' or small_audit.get('passed') is True,'Independent small audit')
    bridge=read('n74_audit.json')
    require(bridge['status']=='PASS' and bridge['n']==74 and bridge['scale_count']==7,'Bridge audit')
    finite=read('mgf_certificates.json')
    require([r['n'] for r in finite['rows']]==list(range(75,404)),'Exact finite coverage')
    finite_audit=read('finite_audit.json')
    require(finite_audit['status']=='PASS' and finite_audit['n_count']==329 and
            finite_audit['scale_count']==2570,'Independent finite audit')
    require(finite_audit['certificate_sha256']==digest(OUT/'mgf_certificates.json'),'Finite certificate identity')
    mid=read('midrange_certificate.json')
    mid_audit=read('midrange_independent_audit.json')
    require(mid['passed'] and mid_audit['passed'],'Independent middle-range proof')
    require(mid_audit['parameter_slabs']==700 and mid_audit['coefficient_cases']==6597 and
            mid_audit['maximum_number_of_scales']==10,'Middle range coverage counts')
    require(mid_audit['original_certificate_sha256']==digest(OUT/'midrange_certificate.json'),'Middle certificate identity')
    expected=[(str(Fraction(i,100)),str(Fraction(i+1,100))) for i in range(700,1400)]
    require([(s['left'],s['right']) for s in mid['parameter_slabs']]==expected,'No continuous L gaps')
    large=read('large_range.json')
    require(large['passed'] and large['C']==70 and large['half_line']['H']==40,'Large range and infinite tail')
    require([(r['left'],r['right']) for r in large['closed_slabs']]==
            [(str(Fraction(i,10)),str(Fraction(i+1,10))) for i in range(140,400)],'Large continuous coverage')
    require(read('large_range_independent_audit.json')['passed'],'Independent large audit')
    with ctx.workprec(512):
        lower=arb(read('basic_certificate.json')['lower']['certified_failure_upper_bound'])
        U_small=arb(0)
        for r in small['rows']:
            U_small=U_small.max(arb(r['union_bound_arb']))
        U_bridge=arb(bridge['union_bound_arb'])
        U_finite=arb(finite_audit['maximum_union_bound_arb'])
        U_middle=arb(mid_audit['risk_upper_interval'])
        B_large=ball(large['bounds']['joint']['upper'])
        require(U_small<ball('.331'),'Small upper target')
        require(U_bridge<ball('.301518'),'Bridge upper target')
        require(U_finite<ball('.291304'),'Finite upper target')
        require(U_middle<ball('.183157'),'Continuous upper target')
        require(B_large<ball('.0009'),'Large joint target')
        require(1+arb(403).log()<7<1+arb(404).log(),'Integer/continuous bridge')
        U=U_small.max(U_bridge).max(U_finite).max(U_middle)
        joint=(U+lower).max(B_large)
        require(joint<ball('.331')<ball(Fraction(1,3)),'Full simultaneous guarantee')
        summary=dict(status='PASS',C0=70,L0=1,poisson_mean='70*n/(epsilon*(1+log(n))) (real)',
                     scope='Every positive integer n, every epsilon in (0,1), every finite or countably supported fixed distribution under independent Poisson sampling',
                     guarantee='Binary promise and clipped effective-support interval simultaneously correct with probability > 0.669 > 2/3',
                     failure_bound_rational='331/1000',
                     computed_uniform_joint_bound_arb=joint.str(65),
                     upper_bounds={'n_1_73':U_small.str(65),'n_74':U_bridge.str(65),
                                   'n_75_403':U_finite.str(65),'L_7_14':U_middle.str(65)},
                     small_lower_bound_arb=lower.str(65),large_joint_bound_arb=B_large.str(65),
                     finite_n_support_pairs=2701,finite_n_75_403_scale_cases=2570,
                     continuous_midrange_cases=6597,continuous_midrange_slabs=700,
                     fresh_full_replay_this_invocation=full_replay,
                     optimizer_used_in_replay=False,
                     optimal_real_constant_claim=False,
                     limitations=['Computer-assisted proof, not proof-assistant formalization',
                                  'The model is iid Poisson sampling; field sampling validity is not implied',
                                  'This sufficient-constant proof does not show practical sample-efficiency superiority'])
    names=['small_certificates.json','small_audit.json','n74_certificate.json','n74_audit.json',
           'mgf_certificates.json','finite_audit.json','midrange_certificate.json',
           'midrange_independent_audit.json','large_range.json','large_range_independent_audit.json',
           'basic_certificate.json']
    summary['artifact_sha256']={name:digest(OUT/name) for name in names}
    summary['proof_source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'proof_l01').glob('*.py'))}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'certified_bounds.tex').write_text('\\newcommand{\\SmallUpperDisplay}{0.331}\n\\newcommand{\\BridgeUpperDisplay}{0.301518}\n')
    print(json.dumps({k:summary[k] for k in ['status','failure_bound_rational',
                     'computed_uniform_joint_bound_arb','fresh_full_replay_this_invocation']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not args.aggregate_only:
        for module in ['proof_l01.verify_small','proof_l01.bridge74','proof_l01.verify_finite',
                       'proof_l01.audit_midrange','proof_l01.large_range']:
            print('REPLAY',module,flush=True)
            subprocess.run([sys.executable,'-m',module],cwd=ROOT,check=True)
    basic_and_large_audit()
    aggregate(not args.aggregate_only)


if __name__=='__main__':
    main()
