#!/usr/bin/env python3
"""Independent, deliberately scoped checks of the supplied L0=1 summary.

This certifies scalar/parameter inequalities, not the unprovided new theorem.
Run: .venv/bin/python real_l01/theory_review.py
The optional search is a diagnostic, never an analytic proof of equation (8).
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flint import arb, ctx, fmpq
from parameter_search.lower_bound import gamma
from parameter_search.algorithm import geometry


def ball(q):
    q = Fraction(q)
    return arb(fmpq(q.numerator, q.denominator))


def H(c):
    alpha = 3 * arb(2).log() * (512 / ball(c)).sqrt()
    return alpha * (alpha / 32).asinh() - (1024 + alpha**2).sqrt() + 32


def check(ok, message):
    if not ok:
        raise ArithmeticError(message)


def scalar_checks():
    result = {}
    for c in ('69.0062713812', '69.0062713813', '70'):
        rigorous_rational = gamma(Fraction(c))
        h = H(c)
        check(h >= ball(rigorous_rational.lower_fraction()), 'Arb/rational lower bound')
        check(h <= ball(rigorous_rational.upper_fraction()), 'Arb/rational upper bound')
        sign = 1 if c == '69.0062713812' else -1
        check(sign * (2 * h - 1) > 0, 'Root endpoint sign')
        result[c] = {
            'H_arb_512_bits': h.str(70),
            '2H_minus_1_arb_512_bits': (2*h-1).str(70),
            'H_independent_integer_intervals': rigorous_rational.text(),
        }
    alpha = 3 * arb(2).log() * (arb(512)/70).sqrt()
    t = (1024+alpha**2).sqrt()-32
    phi = ((alpha+t)*(alpha+t).log() - (alpha-t)*(alpha-t).log()
           - t*(64*t).log()-t)
    check((phi-H('70')).contains(0), 'Independent maximizer algebra agreement')
    return {
        'passed': True,
        'Cstar_open_bracket': ['69.0062713812', '69.0062713813'],
        'uniqueness_reason': 'Hprime(alpha)=asinh(alpha/32)>0, alpha(C) strictly decreases; endpoint limits are infinity and zero.',
        'evaluations': result,
        'C70_alpha': alpha.str(60),
        'C70_t_maximizer': t.str(60),
        'C70_2H_strictly_below_1': True,
        'meaning': 'The scalar asymptotic coefficient exponent is certified; this alone does not establish uniform finite-n success or attainment of the optimal constant.',
    }


def lower_tail_checks():
    # If Delta_r >= epsilon, discoveries before r+1 have probability >= eps.
    # Hence P(D<=r) <= P(Poi(C*n/L)<=r) <= P(Poi(C*n/L)<=n).
    # For C>=69, 1<=L<=32, n=exp(L-1), Chernoff yields
    # exp[-n*(C/L-1-log(C/L))]. Each slab uses lower bounds for n and C/L.
    upper = arb(0)
    for i in range(31*32):
        left = ball(Fraction(1)+Fraction(i,32))
        right = ball(Fraction(1)+Fraction(i+1,32))
        rate = 69/right
        check(rate > 1, 'Poisson Chernoff domain')
        bound = (-((left-1).exp())*(rate-1-rate.log())).exp()
        upper = upper.max(bound)
    check(upper < ball(Fraction(2,10**12)), 'Claimed small-L lower-tail threshold')
    return {
        'passed': True,
        'C_range': '[69,infinity)', 'L_range': '[1,32]',
        'continuous_slab_count': 31*32,
        'certified_failure_upper_bound': upper.str(65),
        'method': 'Poisson-process discovery coupling plus Chernoff, with 512-bit Arb endpoint bounds on 992 closed slabs.',
        'limitation': 'This controls only the lower deviation. Upper deviation and their joint probability need separate proof.',
    }


def parameter_checks():
    # J >= 0, and its definition enforces A_j/B <= 1/2.
    check(arb(256)/70 > 1, 'At least one active scale for all L>=1')
    alpha = 3*arb(2).log()*(arb(512)/70).sqrt()
    ratio_bound = (alpha+1)/(64/arb(2).sqrt())
    check(ratio_bound < 1, 'k <= sqrt(B(B-A)) for all L>=1')
    # (j+3)*2^(-j/2) decreases for j>=0, so k <= alpha*L+1.
    # rho <= 2 exp[-2k sqrt(A/B)] <= delta^2/2 follows analytically.
    finite_count = 0
    for n in range(1,404):
        _, scales = geometry('70',1,n,512)
        finite_count += len(scales)
    b_exceptions = []
    for n in range(1,20):
        threshold = 70*arb(n)/(64*(1+arb(n).log())**2)
        if threshold < 1:
            b_exceptions.append({'n': n, 'epsilon_above_which_b_exceeds_1': threshold.str(45)})
    return {
        'passed': True,
        'all_L_ge_1_geometry_argument': 'J=floor(log2(256 L^2/70))>=0; 0<A_j<=B/2; k_j<=alpha0(70)*L+1; sqrt(B(B-A_j))>=64 L/sqrt(2); rho_j<=delta_j^2/2.',
        'all_L_ge_1_degree_condition_ratio_upper_bound': ratio_bound.str(50),
        'additional_finite_Arb_checks': {'n_range': [1,403], 'scale_count': finite_count},
        'old_b_guard': {
            'b_formula': '64 epsilon L^2/(70 n)',
            'exceptions_for_n_1_to_19': b_exceptions,
            'analytic_review': 'Original paper Lemma 2.3 states b<=1, but its Appendix A proof separately covers x in [0,a], [a,b], and [b,infinity), without using b<=1. The identity, expectation bounds, and coefficient bound therefore extend to arbitrary 0<a<b under the stated remaining hypotheses. This is an extension of that lemma, not a proof of all success-probability claims.',
        },
    }


def diagnostic_homogenization(cases):
    import numpy as np
    from scipy.optimize import differential_evolution, minimize_scalar
    rng = np.random.default_rng(23)

    def value(lambdas, y, phi):
        p = np.array([1.])
        for lam in lambdas:
            p = np.convolve(p, [1-(1-y+lam)*np.exp(-lam), lam*np.exp(-lam)])
        return float(np.dot(p,phi))

    max_gap = 0.
    counterexamples = []
    for r in [2,3,4,5]:
        for case in range(cases):
            y = rng.random()
            phi = np.cumsum(rng.exponential(1,size=r+1)); phi /= phi[-1]
            if case < r:
                phi = np.array([float(i > case) for i in range(r+1)])
            f = lambda ls: -value(ls,y,phi)
            opt = differential_evolution(f, [(1,30)]*r, seed=case,
                                         popsize=7, maxiter=90, tol=1e-9, polish=True)
            grid = np.linspace(1,30,300)
            values = np.array([-f([x]*r) for x in grid])
            best = max(float(phi[0]), float(values.max()))
            # Refine every detected local maximum, including boundary cells:
            # the diagonal objective need not be globally unimodal.
            for j in range(len(grid)):
                if ((j == 0 or values[j] >= values[j-1]) and
                    (j == len(grid)-1 or values[j] >= values[j+1])):
                    diag = minimize_scalar(lambda x:f([x]*r),
                         bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),
                         method='bounded', options={'xatol':1e-12})
                    best = max(best, -diag.fun)
            gap = -opt.fun-best
            max_gap = max(max_gap, gap)
            if gap > 1e-7:
                counterexamples.append({'r':r,'y':float(y),'phi':phi.tolist(),
                  'lambdas':opt.x.tolist(),'value':-opt.fun,'diagonal_best':best})
    return {'status':'diagnostic_only', 'cases':4*cases, 'seed':23,
        'r_values':[2,3,4,5], 'search_box':[1,30],
        'largest_off_diagonal_numerical_improvement':max_gap,
        'candidate_counterexamples':counterexamples,
        'analytic_partial_check': 'Conditional on all other variables, the objective is A+exp(-lambda)*[(y-1)A+lambda(B-A)] with B>=A>=0. Its finite maximizer is lambda=1+(1-y)A/(B-A)>=1. This justifies excluding lambda<1 but does not imply equality of all means.',
        'limitation':'Neither global numerical optimization nor absence of counterexamples proves equation (8); no analytic proof or original verifier of this lemma was supplied locally.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diagnostic-cases',type=int,default=0,
                    help='Optional numerical cases for each r=2,3,4,5; not a proof.')
    ap.add_argument('--output', type=Path,
                    default=ROOT/'results_real_l01_20261104/theory_review.json')
    args = ap.parse_args()
    with ctx.workprec(512):
        result = {
            'review_date':'2026-11-04',
            'requested_parameters':{'C0':70,'L0':1,'m':'70*n/(epsilon*(1+log(n)))','poisson_mean_is_real':True},
            'new_proof_material':{
                'status':'user_pasted_summary_only',
                'expected_missing_local_files':['optimal_constant_L0_one.pdf','optimal_constant_L0_one.tex','L0_1_C0_optimal_certificate.zip','verify_all.py'],
                'search_scope':'Filename-targeted rg --files over /home/snowy; none of the named new artifacts found. Existing older proof packages are different artifacts.',
            },
            'scalar_certificate':scalar_checks(),
            'lower_tail_certificate':lower_tail_checks(),
            'parameter_certificate':parameter_checks(),
            'remaining_theory_gaps':[
                'The supplied sufficient C interval [Cstar,69.00628] and finite certificate interval [69.00627,69.00628] do not contain C=70. Increasing C does not establish stochastic monotonicity because degrees/scales change discretely.',
                'The new small-n MGF majorants, joint-scale envelopes and interval certificates cannot be independently rerun without their missing source/certificate files.',
                'Equation (8), the uniform 0.053 coefficient prefactor, critical-constant attainment, and the triangular-array necessity proof require the full analytic derivations; their statements alone are not sufficient evidence.',
            ],
            'existing_certified_overlap':{
                'source':'results_proof_review/audit.json',
                'C':70,'L_at_least':14,'n_at_least':442414,
                'reason':'With L>=14, both L0=1 and L0=14 select exactly the same multiscale estimator and real Poisson mean.',
                'existing_joint_failure_bound':0.0008794011635031739,
                'status':'Previously audited artifact; this script checks source existence but does not rerun the entire earlier certificate.',
            },
            'overall_status':'partial_independent_theory_validation; full_uniform_L0_1_success_not_certified',
            'interpretation':'C0=70,L0=1 is a well-defined experimentally testable parameterization with certified scalar margin and geometry. Empirical real-data success is not a substitute for the missing finite-range upper-tail proof.',
        }
        if args.diagnostic_cases:
            result['equation8_exploratory_search'] = diagnostic_homogenization(args.diagnostic_cases)
    sources = ['real_l01/theory_review.py','parameter_search/lower_bound.py',
               'parameter_search/algorithm.py',
               'artifacts/joint_constants_submission/rational_interval.py',
               'results_proof_review/audit.json','2610.18723v1.pdf']
    result['source_sha256'] = {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'status':result['overall_status'],
                     'Cstar_bracket':result['scalar_certificate']['Cstar_open_bracket'],
                     'lower_tail_bound':result['lower_tail_certificate']['certified_failure_upper_bound']},
                    ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
