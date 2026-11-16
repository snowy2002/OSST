"""Independent checks of certification, boundaries, and exact stopping costs."""
from fractions import Fraction
import math
import numpy as np
import pytest
from scipy.stats import binom
from flint import arb, ctx, fmpq

from benchmark_core import Population, Profiles, uniform, stable_rng
from guaranteed import ball
from paper_reference import reference_weights
from publication.core import (BASE, compile_scales, signature, certified_lower, near_boundary,
                              make_seq_plan, formal_cases,exact_tail_mass)
from publication.stream import fixed_cap, grouped_labels, simulate_stream_case
from publication.reference import majority_repetitions,fixed_occupancy


@pytest.mark.parametrize("family",["multi","single","fh_certified"])
def test_new_family_certificates_against_independent_arb_formula(family):
    n,m=1000,5000
    scales=compile_scales(n,".1",m,family,signature(BASE))
    assert scales
    with ctx.workprec(256):
      for s in scales:
        # A/B stored as floats are for diagnostics. Reconstruct exact design inputs.
        if family=="fh_certified":
          A=m*ball(BASE["fh_a"])*ball(Fraction(1,10))*arb(10).log()/n
          B=(m*ball(BASE["fh_b"])*ball(Fraction(1,10))*(arb(n).log()+1)**2/(n*arb(10).log())).min(arb(m))
        else:
          A=ball(Fraction(m,80*n)*2**s.certificate.j)
          B=(ball(BASE["b_factor"])*(arb(n).log()+1)).min(arb(m))
        c=s.certificate;w=reference_weights(A,B,c.degree)
        assert 0<A<B<=m and (B*(B-A)).sqrt()>=c.degree
        for a,x in zip(c.coefficients,w[1:]):assert arb(fmpq(a,c.denominator))<=x-1
        rho=1/((B+A)/(B-A)).chebyshev_t(c.degree)
        assert ball(Fraction.from_float(s.rho_upper))>=rho
        width=max([arb(1)]+[x.upper() for x in w])-min([arb(0)]+[x.lower() for x in w])
        margin=width*(arb(n)/2*(arb(len(scales))/ball(Fraction(1,20))).log()).sqrt()
        assert ball(Fraction.from_float(s.margin_upper))>=margin


def test_exact_lower_rounding_cannot_exceed_unrounded_certified_bound():
    n,m=1000,5000
    scales=compile_scales(n,".1",m,"multi",signature(BASE))
    fp=Profiles(np.array([1,600,1000,1200]),np.array([[1]+[0]*23,[300,100]+[0]*22,[0]*24,[500]+[0]*23]))
    lower=certified_lower(fp,scales,n)
    assert np.all((lower>=1)&(lower<=n))
    for i,D in enumerate(fp.observed):
      candidates=[max(1,min(n,int(D)))]
      for s in scales:
        c=s.certificate
        z=Fraction(int(D))+sum(Fraction(a*int(f),c.denominator) for a,f in zip(c.coefficients,fp.low[i]))
        x=(z-Fraction.from_float(s.margin_upper))/(1+Fraction.from_float(s.rho_upper))
        candidates.append(max(1,min(n,math.floor(x))))
      assert lower[i]==max(candidates)


@pytest.mark.parametrize("eps",[.001,.01,.1,.2])
def test_boundary_labels_and_probability_mass(eps):
    for n in [50,1000,1000000]:
      p=near_boundary(n,eps)
      assert p.tail_mass(n)>=eps
      assert exact_tail_mass(p,n)>=Fraction(str(eps))
      assert p.tail_mass(n)-eps<1e-12
      assert near_boundary(n,eps,.999).truth(n,eps)=="Gap"
      cases=formal_cases(n,eps)
      assert all(p.truth(n,eps)==("Gap" if name.startswith("gap") else "Yes" if name.endswith("null") else "No") for name,p in cases)


def test_adversarial_boundary_promise_exact_rational_all_grid_values():
    for n in [1000,10000,100000,1000000]:
      for eps in [.001,.005,.01,.02,.05,.1,.2]:
        for exponent in [None,.4,.8,1.2]:
          try:
            p=near_boundary(n,eps,1.,100,exponent)
          except ValueError:  # This shape may have insufficient top-n tail mass.
            continue
          tail=exact_tail_mass(p,n)
          assert tail>=Fraction(str(eps))
          assert float(tail-Fraction(str(eps)))<1e-12


def test_poly_only_does_not_waste_false_negative_budget():
    p=make_seq_plan(1000,".1",".05","multi",False,signature(BASE))
    assert p.final_m==p.baseline_m
    assert all(l.validation_r==0 for l in p.looks)
    q=make_seq_plan(1000,".1",".05","multi",True,signature(BASE))
    assert q.final_m>p.final_m


def test_grouped_label_marginals_and_fixed_multinomial():
    p=Population(np.array([1,2]),np.array([.5,.25]))
    labels=grouped_labels(p,100000,stable_rng(1))
    freq=np.bincount(labels,minlength=3)/len(labels)
    expected=np.repeat(p.probabilities,p.multiplicities)
    assert np.max(np.abs(freq-expected))<.008
    D=fixed_occupancy(p,3,50000,stable_rng(2))
    mean=np.sum(1-(1-expected)**3)
    assert abs(D.mean()-mean)<.015


def test_exact_immediate_count_cost_against_occupancy_dynamic_program():
    n,S,eps=3,7,.2
    cap=fixed_cap(n,eps)
    state=np.zeros(n+2);state[0]=1
    expected=0
    for t in range(cap):
      expected+=state[:n+1].sum()
      nxt=np.zeros_like(state)
      for d in range(n+1):
        nxt[d]+=state[d]*d/S;nxt[d+1]+=state[d]*(S-d)/S
      nxt[-1]+=state[-1];state=nxt
    configs={fam:dict(BASE) for fam in ["multi","single","fh_certified","fh_native_tuned"]}
    rows,outputs=simulate_stream_case(uniform(S),n,eps,4000,configs,("unit",))
    cost=outputs["count_fixed_immediate"]["cost"]
    assert abs(cost.mean()-expected)<6*cost.std(ddof=1)/math.sqrt(len(cost))
    assert np.all(outputs["count_poisson_immediate"]["cost"]<=outputs["count_checkpoint"]["cost"])
    assert np.all(outputs["count_poisson_immediate"]["reject"]==outputs["count_checkpoint"]["reject"])


def test_fixed_budget_and_reference_amplification_bounds():
    for n,eps in [(3,.2),(1000,.01)]:
      cap=fixed_cap(n,eps)
      assert binom.cdf(n,cap,eps)<=.05
    for error in [Fraction(1,3),Fraction(1,10)]:
      R,bound=majority_repetitions(error)
      assert R%2==1 and bound<=Fraction(1,20)
      assert float(bound)==pytest.approx(binom.sf(R//2,R,float(error)))
