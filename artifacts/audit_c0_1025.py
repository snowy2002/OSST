#!/usr/bin/env python3
"""Independent, reproducible numerical/algebraic audit of the C0=1025 proof.

This is NOT a Lean/Isabelle proof certificate and finite checks do not prove
the quantified theorem. Universal implications are justified in the proof
and audit note. No Monte Carlo sampling is used. Production constants and
historical experiment files are not modified.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import sys
import time

from flint import arb, arb_poly, ctx, fmpq, fmpq_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).with_name("c0_1025_audit")
COUNTS = Counter()
DETAILS = {}


def rat(x):
    x = Fraction(x)
    return fmpq(x.numerator, x.denominator)


def ball(x):
    return arb(rat(x))


def check(group, condition, message):
    if not condition:
        raise AssertionError(f"{group}: {message}")
    COUNTS[group] += 1


def integer(x, direction):
    result = getattr(x, direction)().unique_fmpz()
    if result is None:
        raise ArithmeticError(f"Uncertified {direction}: {x.str(30)}")
    return int(result)


def cheb_at_one_coefficients(k):
    """Exact integer coefficients of T_k(1+u), using a positive series.

    d_l = k * 2^l * (k+l-1)! / ((k-l)! (2l)!).
    This does not use the production three-term recurrence in k.
    """
    coeffs = [1]
    current = 1
    for ell in range(1, k+1):
        numerator = current*(k*k-(ell-1)**2)
        denominator = ell*(2*ell-1)
        if numerator % denominator:
            raise ArithmeticError("T_k(1+u) coefficient not integral")
        current = numerator//denominator
        coeffs.append(current)
    return coeffs


def independent_weights(A, B, k):
    """Positive Taylor expansion followed by polynomial composition.

    Dimensionless A=ma and B=mb. All coefficient operations in this
    construction are positive before the final known alternating sign.
    """
    polynomial = arb_poly(cheb_at_one_coefficients(k))(
        arb_poly([2*A/(B-A), 2/(B-A)]))
    signed = []
    factorial = 1
    magnitudes = []
    for ell in range(k+1):
        if ell:
            factorial *= ell
        v = polynomial[ell]*factorial/polynomial[0]
        magnitudes.append(v)
        signed.append(-v if ell % 2 == 0 else v)
    weights = [arb(0)] + [1+v for v in signed[1:]]
    return weights, magnitudes, polynomial


def endpoints_and_symbolic_signs():
    group = "universal_bound_endpoints_and_exact_algebra"
    c = arb(1025)
    small = c/20-1-(c/20).log()
    large = arb(10**8)/(2*1024*1025)
    check(group, arb(10**8).log()+1 < 20, "count split logarithm")
    check(group, small > arb(100).log(), "small-n count Chernoff exponent")
    check(group, large > arb(100).log(), "large-n count Chernoff exponent")
    check(group, 3*arb(2).log()/arb(2).sqrt()+arb(1)/1024 < arb(3)/2,
          "global degree bound")
    check(group, arb(64)/arb(2).sqrt() > arb(3)/2, "expectation premise")
    # Put L=1024+x, x>=0. Negative coefficients certify these signs
    # for the whole half-line, not merely a sampled grid.
    b_derivative_numerator = [Fraction(2-1024), Fraction(-1)]
    risk_derivative_numerator = [Fraction(4)-Fraction(23,32)*1024, -Fraction(23,32)]
    for name, coeffs in [("b monotonicity", b_derivative_numerator),
                         ("risk monotonicity", risk_derivative_numerator)]:
        check(group, all(v < 0 for v in coeffs), name)
    # Let d=1/4-delta>=0 and u=1-r/n>=0.
    # (14-16*delta)-2*r/n = 8+16*d+2*u, strictly positive.
    check(group, Fraction(14)-16*Fraction(1,4)-2 == 8, "alternative margin identity")
    check(group, Fraction(6)-2*Fraction(1,4) == Fraction(11,2), "diffuse numerator")
    check(group, Fraction(11,2)/Fraction(5,4) == Fraction(22,5), "diffuse corrected margin")
    # Exact numerator identity, checked as a polynomial in delta, for several
    # rational r/n; the coefficient derivation is given in the audit note.
    d = fmpq_poly([0,1])
    for u in [fmpq(0), fmpq(1,7), fmpq(1,2), fmpq(1)]:
        left = (1-d)*(u+16*d)-2*d-u*(1+d)
        right = d*(14-16*d-2*u)
        check(group, left == right, "scale correction algebra")
    for J in range(65):
        total = sum(16*4**j for j in range(J+1))
        check(group, 3*total <= 4*(16*4**J), "geometric union sum")
    risk = arb(128)/3*arb(1).exp()*arb(1024)**4*(-arb(23)/32*1024).exp()
    check(group, risk < ball("3e-306"), "uniform large-L risk")
    check(group, arb(1024)**2*arb(-1023).exp()/16 < 1, "b<=1")
    check(group, 2*arb(-1023).exp() < ball(".01"), "large occupancy error")
    heavy = arb(-31*1024-1).exp()+(-arb(1023).exp()).exp()
    check(group, heavy < ball(".01"), "heavy top-r case")
    DETAILS[group] = dict(count_small_exponent=small.str(30),
                          count_large_exponent=large.str(30),
                          count_uniform_error_upper=(-small.min(large)).exp().str(30),
                          polynomial_uniform_error_upper=risk.str(30))


def poisson_cdf(mean, r):
    term = (-mean).exp()
    total = term
    for ell in range(1,r+1):
        term = term*mean/ell
        total += term
    return total


def occupancy_cdf(probabilities, mean):
    """Full Poisson-binomial law, independent of waiting-time reasoning."""
    dp = [arb(1)]
    for p in probabilities:
        absent = (-mean*ball(p)).exp()
        present = -(-mean*ball(p)).expm1()
        nxt = [arb(0)]*(len(dp)+1)
        for d, value in enumerate(dp):
            nxt[d] += value*absent
            nxt[d+1] += value*present
        dp = nxt
    return [sum(dp[:r+1]) for r in range(len(dp))]


def exact_count_probabilities():
    group = "independent_finite_distribution_count_probabilities"
    generator = random.Random(102521)
    distributions = []
    for S in range(2,13):
        patterns = [[1]*S, [10000]+[1]*(S-1)]
        patterns += [[generator.randint(1,1000) for _ in range(S)] for _ in range(3)]
        for weights in patterns:
            weights = sorted(weights, reverse=True)
            distributions.append([Fraction(w,sum(weights)) for w in weights])
    inequalities = 0
    formal_cells = 0
    for probabilities in distributions:
        S = len(probabilities)
        for mean0 in [Fraction(1,20),Fraction(1,2),2,10,50]:
            mean = ball(mean0)
            cdfs = occupancy_cdf(probabilities,mean)
            check(group, cdfs[-1].contains(1), "occupancy law normalizes")
            for r in range(1,S):
                epsilon = sum(probabilities[r:])
                bound = poisson_cdf(mean*ball(epsilon),r)
                check(group, cdfs[r] < bound, "discovery-time bound vs independent exact law")
                inequalities += 1
        # Direct risk of the unchanged count algorithm, at C0=1025.
        for n in sorted({1,S//2,S-1}):
            epsilon = sum(probabilities[n:])
            mean = 1025*arb(n)/(ball(epsilon)*(arb(n).log()+1))
            actual = occupancy_cdf(probabilities,mean)[n]
            check(group, actual < ball(".01"), "formal C0=1025 alternative risk")
            check(group, actual < poisson_cdf(mean*ball(epsilon),n), "formal count domination")
            formal_cells += 1
    DETAILS[group] = dict(distributions=len(distributions),
                          general_bound_comparisons=inequalities,
                          formal_C1025_alternative_cells=formal_cells,
                          method="Arb Poisson-binomial dynamic programming; no sampling")


def exact_coefficients():
    group = "exact_rational_coefficients_and_expectations"
    degrees = [1,2,3,5,8,12,20,40]
    parameters = [(Fraction(1,128),Fraction(7,3)),
                  (Fraction(1,8),Fraction(16)),
                  (Fraction(1,2),Fraction(1)),
                  (Fraction(999,1000),Fraction(1))]
    # Recurrence in degree over exact rationals versus the positive finite
    # series followed by affine composition: structurally different formulas.
    identities = 0
    for A,B in parameters:
        alpha,beta = rat((B+A)/(B-A)),rat(2/(B-A))
        affine = fmpq_poly([alpha,beta])
        for k in degrees:
            prev,cur = fmpq_poly([1]),affine
            for _ in range(1,k):
                prev,cur = cur,2*affine*cur-prev
            positive = fmpq_poly(cheb_at_one_coefficients(k))(
                fmpq_poly([alpha-1,beta]))
            check(group, cur == positive, "exact polynomial coefficient identity")
            check(group, all(coef>0 for coef in positive.coeffs()), "all shifted coefficients positive")
            weights, mags, poly = independent_weights(ball(A),ball(B),k)
            factorial = 1
            for ell in range(k+1):
                if ell:
                    factorial *= ell
                exactmag = positive[ell]*factorial/positive[0]
                check(group, mags[ell].contains(exactmag), "interval contains rational coefficient")
                identities += 1
            bound = 4*(2*arb(k)**2/(ball(B)-ball(A))).exp()
            check(group, sum(mags) < bound, "sum of absolute factorial weights bound")
            # Poisson expectation from probabilities vs polynomial closed form.
            for lam in [arb(0),ball(A)/2,ball(A),(ball(A)+ball(B))/2,ball(B),2*ball(B)]:
                pmf = (-lam).exp()
                direct = -pmf
                for ell in range(1,k+1):
                    pmf = pmf*lam/ell
                    direct += (weights[ell]-1)*pmf
                direct += 1
                closed = 1-(-lam).exp()*((ball(B)+ball(A)-2*lam)/(ball(B)-ball(A))).chebyshev_t(k)/poly[0]
                check(group, direct.overlaps(closed), "Poisson expectation identity")
                check(group, (direct-closed).abs_upper() < ball("1e-300"), "expectation identity precision")
    DETAILS[group] = dict(parameter_pairs=len(parameters), degrees=degrees,
                          individual_coefficients=identities,
                          oracle="exact fmpq_poly recurrence vs positive T_k(1+u) series")


def independent_plan(n, epsilon, rounded):
    L = arb(n).log()+1
    ep = ball(epsilon)
    mean = 1025*arb(n)/(ep*L)
    if rounded:
        mean = arb(integer(mean,"ceil")) if n != 1 else arb(math.ceil(Fraction(1025)/epsilon))
    if L < 1024:
        return dict(L=L,m=mean,scales=[],J=None)
    if not L >= 1024:
        raise ArithmeticError("Undecided branch")
    # Independently derive the dimensionless quantities directly from m.
    A0 = mean*ep/(8*n)
    B = 64*L
    J = integer((B/(2*A0)).log()/arb(2).log(),"floor")
    scales=[]
    for j in range(J+1):
        A = A0*2**j
        delta = Fraction(1,2**(j+2))
        target = (B/A).sqrt()*ball(2/delta).log()
        k = integer(target,"ceil")
        rho = 1/((B+A)/(B-A)).chebyshev_t(k)
        scales.append(dict(j=j,A=A,B=B,k=k,delta=delta,rho=rho,target=target))
    return dict(L=L,m=mean,scales=scales,J=J)


def branch_and_parameters():
    group = "integer_branch_boundaries_and_parameters"
    boundary = integer(arb(1023).exp(),"floor")
    ns = [1,2,1000,10**6,10**8-1,10**8,10**8+1,
          boundary-1,boundary,boundary+1,boundary+2,10**445,10**500,10**1000]
    epsilons = [Fraction(1,10**40),Fraction(1,1000),Fraction(1,10),Fraction(1,2),Fraction(999999,10**6)]
    cells=0; scale_cells=0; largest_k=0; summaries=[]
    for n in ns:
        for epsilon in epsilons:
            for rounded in [False,True]:
                plan=independent_plan(n,epsilon,rounded)
                L=plan['L'];mean=plan['m'];scales=plan['scales']
                check(group, bool(scales)==(n>boundary), "exact integer branch side")
                if rounded:
                    target=1025*arb(n)/(ball(epsilon)*L)
                    if n==1:
                        check(group,int(mean.unique_fmpz())==math.ceil(Fraction(1025)/epsilon),"exact rational ceil n=1")
                    else:
                        check(group,mean>=target and mean-1<target,"minimal integer mean")
                for s in scales:
                    A,B,k,d=s['A'],s['B'],s['k'],ball(s['delta'])
                    check(group, A>0 and A<=B/2, "a range")
                    check(group, B/mean<1, "b range")
                    check(group, s['target']>k-1 and s['target']<=k, "degree ceiling")
                    check(group, k<arb(3)/2*L, "degree envelope")
                    check(group, (B*(B-A)).sqrt()>k, "expectation lemma condition")
                    check(group, s['rho']<=d*d/2, "rho approximation")
                    check(group, B-A>=32*L, "weight denominator")
                    check(group, (A/mean*d).overlaps(ball(epsilon)/(32*n)), "a_j delta_j identity")
                    scale_cells+=1;largest_k=max(largest_k,k)
                if scales:
                    check(group,2*scales[-1]['A']>scales[-1]['B']/2,"J includes last admissible scale")
                    check(group,ball(1/scales[-1]['delta'])<=L*L,"union allocation bound")
                cells+=1
                if epsilon==Fraction(1,10) and not rounded:
                    summaries.append(dict(n_digits=len(str(n)),branch='polynomial' if scales else 'count',
                                          L=L.str(22),J=plan['J'],degrees=[s['k'] for s in scales]))
    DETAILS[group]=dict(parameter_cells=cells,scale_cells=scale_cells,largest_degree=largest_k,
                        boundary_n=str(boundary),selected_cases=summaries)
    return boundary


def large_degree_weights_and_pipeline(boundary):
    import paper_reference as reference
    from support_size import Fingerprint
    group="large_degree_independent_weights_and_pipeline"
    old_constant=reference.C0
    record=[];weight_comparisons=0;expectations=0;pipeline_cases=0
    precision_events=[];public_calls=[]
    try:
        # A process-local test fixture. The module file is never edited.
        reference.C0=1025
        for n in [boundary+1,10**445,10**500]:
            epsilon=Fraction(1,10)
            independent=independent_plan(n,epsilon,True)
            production=reference.reference_parameters(n,epsilon,max_bits=16384)
            check(group,production.m==int(independent['m'].unique_fmpz()),"production mean agrees")
            check(group,production.J==independent['J'],"production scale count agrees")
            for s,p in zip(independent['scales'],production.scales):
                check(group,s['k']==p.k and s['delta']==p.delta,"production degrees/allocations agree")
            # All scales in the first boundary case; edge and interior scales
            # in larger cases. Keep full, untruncated degrees.
            selected=(list(range(len(production.scales))) if n==boundary+1 else
                      sorted({0,1,len(production.scales)//2,len(production.scales)-1}))
            weights_by_j={}
            for j in selected:
                s=independent['scales'][j]
                A,B,k=s['A'],s['B'],s['k']
                weights,mags,poly=independent_weights(A,B,k)
                weights_by_j[j]=weights
                # The production implementation uses a different, signed
                # three-term recurrence. Compare each coefficient at high precision.
                other=reference.reference_weights(A,B,k)
                for ell,(w,v) in enumerate(zip(weights,other)):
                    check(group,w.overlaps(v),"independent vs production coefficient")
                    check(group,(w-v).abs_upper()<ball('1e-200')*(1+abs(w)),"coefficient precision")
                    weight_comparisons+=1
                integral_bound=4*(2*arb(k)**2/(B-A)).exp()
                check(group,sum(mags)<integral_bound,"full-degree integral bound")
                uniform_bound=4*(arb(9)/64*independent['L']).exp()
                check(group,integral_bound<uniform_bound,"uniform envelope")
                max_weight=arb(1)
                for w in weights:max_weight=max_weight.max(abs(w))
                check(group,max_weight<sum(mags),"norm vs coefficient sum")
                # Direct finite PMF formula vs scalar Chebyshev formula, with
                # samples covering all three x ranges and their boundaries.
                locations=[A/100,A/2,A,2*A,(A+B)/2,B,2*B,10*B]
                for lam in locations:
                    pmf=(-lam).exp(); direct=-pmf
                    for ell in range(1,k+1):
                        pmf=pmf*lam/ell
                        direct+=(weights[ell]-1)*pmf
                    direct+=1
                    closed=1-(-lam).exp()*((B+A-2*lam)/(B-A)).chebyshev_t(k)/poly[0]
                    check(group,direct.overlaps(closed),"full-degree expectation identity")
                    lower=(1-ball(s['delta']))*(lam/A).min(1)
                    upper=1+ball(s['delta'])
                    check(group,closed>lower and closed<upper,"expectation bounds in all ranges")
                    expectations+=1
                record.append(dict(n_digits=len(str(n)),j=j,k=k,
                                   max_abs_weight=max_weight.str(24),
                                   coefficient_sum=sum(mags).str(24),
                                   integral_bound=integral_bound.str(24)))
            if n==boundary+1:
                K=max(s['k'] for s in independent['scales'])
                # The original low-degree plan's initial precision need not
                # suffice after changing C. Explicitly use a high-precision
                # plan for arithmetic comparison; also test the public
                # adaptive-precision entry point separately below.
                high_plan=reference._plan_at_precision(n,epsilon,4096)
                fixtures=[{1:n},{1:n//100},{K+1:n},{K+1:n+1},{1:n//3,2:n//5,3:n//7}]
                for freq in fixtures:
                    fp=Fingerprint(freq)
                    direct=arb(fp.observed)
                    for s in independent['scales']:
                        w=weights_by_j[s['j']]
                        z=sum(count*(w[ell] if ell<len(w) else arb(1)) for ell,count in freq.items())
                        corrected=(z-ball(s['delta'])*n)/(1+ball(s['delta']))
                        direct=direct.max(corrected)
                    actual,_=reference.reference_statistic(fp,high_plan,max_degree=10000)
                    check(group,actual.overlaps(direct),"full corrected maximum")
                    check(group,(actual-direct).abs_upper()<arb(n)*ball('1e-200'),"full corrected maximum precision")
                    # Check an actual polynomial win, not only cases where D
                    # masks the entire polynomial branch.
                    if freq=={1:n}:
                        check(group,direct>n,"fixture exercises polynomial rejection")
                    if freq=={K+1:n}:
                        check(group,direct.contains(n),"null output boundary")
                    if freq=={1:n}:
                        low_score,_=reference.reference_statistic(fp,production,max_degree=10000)
                        precision_events.append(dict(initial_bits=production.precision_bits,
                                                     high_bits=high_plan.precision_bits,
                                                     initial_relative_difference_bound=((low_score-direct).abs_upper()/n).str(24),
                                                     high_relative_difference_bound=((actual-direct).abs_upper()/n).str(24)))
                    if freq in [{1:n},{K+1:n}]:
                        public=reference.reference_test(fp,n,epsilon,m=production.m,
                                                        max_bits=16384,max_degree=10000)
                        expected='No' if freq=={1:n} else 'Yes'
                        check(group,public['decision']==expected,"public adaptive-precision decision")
                        check(group,public['support_lower_bound']==n,"public clipped lower bound")
                        public_calls.append(dict(expected=expected,returned=public['decision'],
                                                 initial_bits=production.precision_bits,
                                                 final_bits=public['precision_bits']))
                    pipeline_cases+=1
    finally:
        reference.C0=old_constant
    DETAILS[group]=dict(weight_coefficients_compared=weight_comparisons,
                       expectation_points=expectations,pipeline_fixtures=pipeline_cases,
                       precision_events=precision_events,public_adaptive_calls=public_calls,
                       full_degree_cases=record,
                       note="Deterministic formula checks, not simulations with 10^444 observations")


def explicit_counterexample(boundary):
    group="explicit_finite_counterexample_for_C1024"
    n=boundary
    K=100*n*n+n-1
    epsilon=Fraction(1,4)
    q=epsilon/(K-n+1)
    head=1-K*q
    check(group,head>q>0,"valid head and tail masses")
    check(group,head+K*q==1,"normalization")
    check(group,(K-n+1)*q==epsilon,"exact No promise")
    L=arb(n).log()+1
    check(group,L<1024 and L>1023,"counterexample count branch")
    mean=1024*arb(n)/(ball(epsilon)*L)
    tailmean=mean*ball(K*q)
    excess=tailmean-n
    check(group,excess>0 and excess<ball('.012'),"explicit tail mean n+small excess")
    error_lower=(-excess).exp()/16
    check(group,error_lower>ball('.06'),"failure above 1 percent")
    roundedmean=arb(integer(mean,'ceil'))
    rounded_excess=roundedmean*ball(K*q)-n
    check(group,(-rounded_excess).exp()/16>ball('.048'),"rounded mean counterexample")
    # Poisson central-moment identity with exact Touchard polynomials.
    # E Y^r from falling-factorial moments, r=0,...,4.
    mu=fmpq_poly([0,1])
    raw=[fmpq_poly([1]),mu,mu**2+mu,mu**3+3*mu**2+mu,
         mu**4+6*mu**3+7*mu**2+mu]
    central4=sum(math.comb(4,j)*(-mu)**(4-j)*raw[j] for j in range(5))
    check(group,central4==3*mu**2+mu,"exact Poisson fourth central moment")
    DETAILS[group]=dict(n_digits=len(str(n)),tail_categories_digits=len(str(K)),
                       K_formula="100*n^2+n-1",epsilon="1/4",q_formula="1/(400*n^2)",
                       tail_mean_minus_n=excess.str(30),
                       rigorous_error_lower_bound=error_lower.str(30),
                       rounded_mean_error_lower_bound=((-rounded_excess).exp()/16).str(30))


def negative_controls():
    group="deliberate_error_controls"
    A,B=arb(1)/8,arb(16)
    w,mags,p=independent_weights(A,B,8)
    wrong=1-p[2]/p[0] # missing 2! in the negative coefficient
    check(group,not w[2].overlaps(wrong),"detect omitted factorial")
    delta=Fraction(1,4);n=100;S=100
    z=(1+delta)*S+delta*n # extremal value allowed by null expectation + deviation
    correct=(z-delta*n)/(1+delta)
    incorrect=(z+delta*n)/(1+delta)
    check(group,correct==S and incorrect>S,"detect wrong sign in correction")
    check(group,float(Fraction(1024)+Fraction(9,4*10**219))==1024.0,
          "binary64 cannot encode the proposed tiny increment")
    DETAILS[group]=dict(controls=["missing factorial detected","wrong correction sign detected",
                                  "tiny increment lost in binary64 detected"])


def main():
    OUT.mkdir(exist_ok=True)
    start=time.perf_counter()
    with ctx.workprec(4096):
        for function in [endpoints_and_symbolic_signs,exact_count_probabilities,exact_coefficients]:
            t=time.perf_counter();function()
            print(f"PASS {function.__name__}: {time.perf_counter()-t:.2f}s",flush=True)
        boundary=branch_and_parameters()
        print("PASS branch_and_parameters",flush=True)
        large_degree_weights_and_pipeline(boundary)
        print("PASS large_degree_weights_and_pipeline",flush=True)
        explicit_counterexample(boundary)
        negative_controls()
    inputs=[Path(__file__),Path(__file__).with_name('c0_reduction_proof.tex'),
            ROOT/'paper_reference.py',ROOT/'2610.18723v1.pdf']
    result=dict(passed=True,precision_bits=4096,elapsed_seconds=time.perf_counter()-start,
                python=platform.python_version(),checks_by_group=dict(COUNTS),
                total_assertions=sum(COUNTS.values()),details=DETAILS,
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                limitations=["Not a machine-checked formal proof",
                             "Universal validity comes from the analytic proof, not finite grids",
                             "Independent formula implementations share the Python/FLINT arithmetic stack",
                             "Huge-scale tests use compressed algebra, not physical sampling",
                             "No third-party mathematical review has been performed"])
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(passed=result['passed'],assertions=result['total_assertions'],
                         elapsed_seconds=result['elapsed_seconds'],groups=result['checks_by_group']),indent=2))


if __name__=='__main__':
    main()
