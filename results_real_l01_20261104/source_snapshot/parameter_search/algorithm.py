"""Parameterised, numerically enclosed implementation for comparison experiments.

Certification belongs to specific rows of certificate.json, NOT arbitrary C/L0.
The original c7016 implementation and its frozen experiment sources are unchanged.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
from flint import arb, ctx

from c7016.algorithm import Scale, ball, validate_profiles
from guaranteed import directed_float
from paper_reference import NumericalCertificationError, certified_integer, exact_inputs, reference_weights


def constant(C):
    return 144*arb(2).log()**2 if str(C)=='star' else ball(Fraction(str(C)))


@dataclass(frozen=True)
class Plan:
    C: str
    L0: int
    n: int
    epsilon: Fraction
    L: arb
    m: arb
    scales: tuple
    bits: int

    @property
    def degree(self):
        return max((s.k for s in self.scales),default=0)

    @property
    def branch(self):
        return 'chebyshev' if self.scales else 'observed_count'


@lru_cache(maxsize=1024)
def geometry(C,L0,n,bits=512):
    with ctx.workprec(bits):
        c=constant(C);L=1+arb(n).log()
        if c<=0 or L0<1:
            raise ValueError('Require C>0 and integer L0>=1')
        if L<L0:
            return L,()
        if not L>=L0:
            raise NumericalCertificationError('Branch unresolved')
        J=certified_integer((256*L*L/c).log()/arb(2).log(),'floor')
        if J<0:
            raise ValueError('No active scale at these experimental parameters')
        result=[]
        for j in range(J+1):
            A,B=c*arb(2)**j/(8*L),64*L
            delta=Fraction(1,2**(j+2))
            k=certified_integer((B/A).sqrt()*ball(2/delta).log(),'ceil')
            if k>4096:
                raise ValueError('Degree resource limit; never truncate')
            rho=1/((B+A)/(B-A)).chebyshev_t(k)
            if not (0<A and A<=B/2 and (B*(B-A)).sqrt()>=k and rho<=ball(delta**2/2)):
                raise NumericalCertificationError('Parameter condition failed')
            weights=reference_weights(A,B,k)
            result.append(Scale(j,A,B,k,delta,rho,tuple(w-1 for w in weights[1:])))
        return L,tuple(result)


def parameters(C,L0,n,epsilon,bits=512):
    n,eps=exact_inputs(n,epsilon)
    with ctx.workprec(bits):
        L,scales=geometry(str(C),int(L0),n,bits)
        m=constant(C)*n/(ball(eps)*L)
        if scales and not scales[0].B/m<=1:
            raise NumericalCertificationError('Polynomial endpoint exceeds one')
        return Plan(str(C),int(L0),n,eps,L,m,scales,bits)


@lru_cache(maxsize=1024)
def compiled(C,L0,n,bits=512):
    last=None
    for precision in [bits,2*bits,4*bits,8*bits]:
        try:
            return compile_at_precision(C,L0,n,precision)
        except ArithmeticError as exc:
            last=exc
    raise NumericalCertificationError('Unable to enclose weights after precision escalation') from last


def compile_at_precision(C,L0,n,bits):
    with ctx.workprec(bits):
        _,scales=geometry(C,L0,n,bits)
        K=max((s.k for s in scales),default=0)
        lo=np.zeros((len(scales),K));hi=lo.copy()
        for scale in scales:
            for ell,value in enumerate(scale.corrections):
                lo[scale.j,ell]=directed_float(value,False)
                hi[scale.j,ell]=directed_float(value,True)
        return lo,hi


def score_intervals(profiles,plan):
    if plan.n>2**53:
        raise ValueError('Batched enclosures require n<=2^53')
    validate_profiles(profiles,plan.degree)
    D=profiles.observed.astype(np.float64)
    if not plan.scales:
        return D.copy(),D.copy()
    low,high=compiled(plan.C,plan.L0,plan.n,plan.bits)
    x=profiles.low[:,:plan.degree].astype(np.float64)
    down=lambda v:np.nextafter(v,-np.inf)
    up=lambda v:np.nextafter(v,np.inf)
    g=(4*plan.degree+32)*2.**-52
    absolute=x@np.maximum(np.abs(low),np.abs(high)).T
    error=up(up(g*up(absolute/(1-g)))+1e-280)
    a=down(down(x@low.T-error)+D[:,None])
    b=up(up(x@high.T+error)+D[:,None])
    delta=np.array([float(s.delta) for s in plan.scales])
    a=down(down(a-delta*plan.n)/(1+delta))
    b=up(up(b-delta*plan.n)/(1+delta))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise NumericalCertificationError('Nonfinite score enclosure')
    return np.maximum(D,a.max(axis=1)),np.maximum(D,b.max(axis=1))


def arb_score(observed,low,plan,bits=1024):
    with ctx.workprec(bits):
        _,scales=geometry(plan.C,plan.L0,plan.n,bits)
        score=arb(int(observed))
        for s in scales:
            raw=arb(int(observed))
            for count,correction in zip(low[:s.k],s.corrections):
                if count:
                    raw+=int(count)*correction
            score=score.max((raw-ball(s.delta)*plan.n)/(1+ball(s.delta)))
        return score


def compare(profiles,plan,lo,hi,threshold):
    if abs(threshold)>2**53:
        raise ValueError('Threshold outside exact binary64 integer domain')
    result=lo>threshold
    uncertain=np.flatnonzero((lo<=threshold)&(hi>threshold))
    for i in uncertain:
        for bits in [1024,2048,4096,8192]:
            score=arb_score(profiles.observed[i],profiles.low[i],plan,bits)
            if score>threshold:
                result[i]=True;break
            if score<=threshold:
                result[i]=False;break
        else:
            raise NumericalCertificationError('Unresolved decision; no guess')
    return result,len(uncertain)
