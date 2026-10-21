"""C0=1025 follow-up: exact targets, certified plans, occupancy simulation."""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from flint import arb,ctx,fmpq
from scipy.stats import beta,binom
from benchmark_core import Population,uniform,headtail,rank_blocks,stable_rng
from publication.core import near_boundary,exact_tail_mass
from paper_reference import reference_parameters,certified_integer


def effective_support(pop,epsilon):
    weights=[Fraction.from_float(float(p)) for p in pop.probabilities]
    mass=sum(int(g)*p for g,p in zip(pop.multiplicities,weights))
    remainder=Fraction(str(epsilon))*mass; discard=0
    for g,p in zip(pop.multiplicities[::-1],weights[::-1]):
        g=int(g)
        if g*p<=remainder: discard+=g; remainder-=g*p
        else: discard+=min(g,int(remainder//p)); break
    return max(1,pop.support-discard)


def truth(pop,n,eps):
    tail=exact_tail_mass(pop,n)
    label='Yes' if pop.support<=n else 'No' if tail>=Fraction(str(eps)) else 'Gap'
    return label,tail,effective_support(pop,eps)


@lru_cache(maxsize=None)
def budget(n,epsilon,method):
    ep=Fraction(str(epsilon))
    if method=='count_chernoff_99':
        with ctx.workprec(256):
            a=arb(100).log()
            lam=n+a+(a*a+2*n*a).sqrt()
            m=certified_integer(lam/arb(fmpq(ep.numerator,ep.denominator)),'ceil')
            x=m*arb(fmpq(ep.numerator,ep.denominator))-n
            assert (-(x*x)/(2*(n+x))).exp()<arb(1)/100
        return m
    c=Fraction(method.removeprefix('C'))
    if c.denominator!=1:
        with ctx.workprec(256):
            return certified_integer(arb(fmpq(c.numerator,c.denominator))*n/
                                     (arb(fmpq(ep.numerator,ep.denominator))*(arb(n).log()+1)),'ceil')
    c=int(c)
    plan=reference_parameters(n,ep,c0=c)
    if plan.branch!='observed_count':
        raise ValueError('Occupancy shortcut must not replace the polynomial branch')
    return plan.m


def sample_path(pop,means,trials,rng):
    """Joint counts along one Poisson process, without enumerating labels.

    Within each equal-probability group, remaining unseen labels are thinned
    with probability 1-exp(-delta_m*p). This preserves exact joint occupancy
    laws in real arithmetic; NumPy evaluates probabilities in binary64.
    Does not claim to generate the realized total sample count M.
    """
    means=sorted(set(int(m) for m in means)); result=np.zeros((len(means),trials),dtype=np.int64)
    for g,p in zip(pop.multiplicities,pop.probabilities):
        seen=np.zeros(trials,dtype=np.int64); last=0
        for j,m in enumerate(means):
            q=float(-np.expm1(-float(m-last)*float(p)))
            seen+=rng.binomial(int(g)-seen,q)
            result[j]+=seen;last=m
    return {m:result[j] for j,m in enumerate(means)}


def count_decisions(observed,plan):
    """Vectorized literal observed-count branch; plan is interval certified."""
    if plan.branch!='observed_count': raise ValueError('Polynomial plan requires full reference_test')
    observed=np.asarray(observed)
    if np.any(observed<0) or np.any(observed!=np.floor(observed)):raise ValueError('Invalid observed count')
    return observed>plan.n,np.minimum(plan.n,np.maximum(1,observed))


def cp_upper(k,total,alpha=.05):
    return 1. if k==total else float(beta.isf(alpha,k+1,total-k))


def exact_occupancy_cdf(pop,m,n):
    """Independent finite PGF convolution in binary64 (not interval certified)."""
    if pop.support>20000 or n>2000: return None
    pmf=np.array([1.])
    for g,p in zip(pop.multiplicities,pop.probabilities):
        g=int(g); q=float(-np.expm1(-float(m)*p))
        block=binom.pmf(np.arange(min(g,n)+1),g,q)
        pmf=np.convolve(pmf,block)[:n+1]
    return float(np.clip(pmf.sum(),0,1))


def scenarios(n,eps):
    cases=[('uniform_null',uniform(n)),('subset_null',uniform(max(1,n//2)))]
    if n>1:
        cases.extend([('head90_null',headtail(.9,n-1)),('rare_null',headtail(1-1e-8,n-1)),
                      ('power_null',rank_blocks(n,1.4,64))])
        edges=np.unique(np.linspace(0,n,min(n,64)+1).astype(np.int64)); g=np.diff(edges)
        rng=stable_rng(1025001,'population',n,eps)
        for name,w in [('lognormal_null',np.exp(rng.normal(0,2.5,len(g)))),
                       ('gaussian_blocks_null',np.exp(-np.linspace(-3,3,len(g))**2/2)),
                       ('geometric_blocks_null',np.exp(-np.linspace(0,14,len(g))))]:
            cases.append((name,Population(g,w/(g@w))))
    cases.extend([('uniform_boundary',uniform(math.ceil(n/(1-eps))+1)),
                  ('diffuse_boundary',near_boundary(n,eps,1.,100)),
                  ('ultradiffuse_boundary',near_boundary(n,eps,1.,1000)),
                  ('multiscale_boundary',near_boundary(n,eps,1.000001,100,.91)),
                  ('clear_alternative',near_boundary(n,eps,2.,100)),
                  ('gap_near',near_boundary(n,eps,.999999,100)),
                  ('gap_half',near_boundary(n,eps,.5,100))])
    for name,pop in cases:
        label,_,_=truth(pop,n,eps)
        expected='Yes' if name.endswith('null') else 'Gap' if name.startswith('gap') else 'No'
        assert label==expected,(name,n,eps,label)
    return cases
