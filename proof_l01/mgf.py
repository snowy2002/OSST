"""Explore and certify finite-n multiscale upper tails, retaining zero counts.

G(t,lambda)=1+sum_{ell=0}^k expm1(t*y_ell) Poi(lambda)[ell], y_0=-1.
For S in [1,n], the Chernoff exponent is affine in S, so its worst
endpoint is max{logG-t*delta*(n+1), n*logG-2*t*delta*n}.
The initial optimizer is explicitly nonrigorous; separate certificates are
required before its numerical estimates can be used in a proof.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
from fractions import Fraction
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from parameter_search.algorithm import geometry
from flint import arb, arb_poly, ctx, fmpq


def ball(value):
    if isinstance(value,arb):return value
    q=Fraction(value)
    return arb(fmpq(q.numerator,q.denominator))


def bernstein_coefficients(poly,a,b,degree):
    """Power-to-Bernstein conversion using one Arb polynomial convolution.

    beta_i=(i!/d!)*sum_{j<=i} (d-j)! q_j/(i-j)!,
    where q(u)=poly(a+(b-a)u). Every arithmetic operation is outward-rounded.
    """
    q=poly(arb_poly([ball(a),ball(b)-ball(a)]))
    fac=[math.factorial(i) for i in range(degree+1)]
    u=arb_poly([q[j]*fac[degree-j] for j in range(degree+1)])
    v=arb_poly([arb(1)/fac[j] for j in range(degree+1)])
    prod=u*v
    return [prod[i]*fac[i]/fac[degree] for i in range(degree+1)]


def certify_mgf(ys,t,eta,*,bits=512,R=8,degree=None,max_depth=18):
    """Prove G(t,lambda)<=1+eta uniformly on lambda>=0.

    Build P=eta*T_d(lambda)-sum expm1(t*y_l)*lambda^l/l!, where
    T_d is the exponential Taylor lower bound. On [0,R], adaptive Bernstein
    positivity suffices. On [R,infinity), all coefficients of P(R+x) are
    checked nonnegative. No optimizer result enters the proof unchecked.
    ys may be outward Arb intervals (including a continuous L slab).
    """
    with ctx.workprec(bits):
        t,eta=ball(t),ball(eta)
        if not (t>0 and eta>0):raise ValueError('Positive t,eta required')
        k=len(ys)-1;d=max(k,48) if degree is None else degree
        if d<k:raise ValueError('Taylor degree must cover polynomial degree')
        coeff=[eta/math.factorial(i) for i in range(d+1)]
        for ell,y in enumerate(ys):coeff[ell]-=((t*ball(y)).exp()-1)/math.factorial(ell)
        poly=arb_poly(coeff)
        shifted=poly(arb_poly([ball(R),1]))
        tail_ok=all(shifted[i]>=0 for i in range(d+1))
        if not tail_ok:
            return {'passed':False,'reason':'tail_shift_coefficient','R':R,'degree':d}
        pending=[(Fraction(0),Fraction(R),0)];leaves=[];nodes=0
        while pending:
            a,b,depth=pending.pop();nodes+=1
            bern=bernstein_coefficients(poly,a,b,d)
            if all(v>0 for v in bern):
                leaves.append([str(a),str(b)]);continue
            if depth>=max_depth:
                return {'passed':False,'reason':'bernstein_depth','interval':[str(a),str(b)],'R':R,'degree':d,'nodes':nodes}
            mid=(a+b)/2
            pending.extend([(mid,b,depth+1),(a,mid,depth+1)])
        return {'passed':True,'R':R,'degree':d,'bits':bits,'bernstein_nodes':nodes,
                'bernstein_leaves':leaves,'tail_shift_all_coefficients_nonnegative':True}


def certify_scale(n,scale,t,eta,*,bits=512):
    t,eta=Fraction(t),Fraction(eta)
    ys=[arb(-1)]+list(scale.corrections)
    certificate=certify_mgf(ys,t,eta,bits=bits)
    if not certificate['passed']:
        certificate=certify_mgf(ys,t,eta,bits=bits,R=16,degree=max(scale.k,72))
    with ctx.workprec(bits):
        logG=(1+ball(eta)).log();td=ball(t)*ball(scale.delta)
        exponent=(logG-td*(ball(n)+1)).max(ball(n)*logG-2*td*ball(n))
        bound=exponent.exp()
    certificate.update({'j':scale.j,'k':scale.k,'t':str(t),'eta':str(eta),'delta':str(scale.delta),
                        'bound_arb':bound.str(65),'bound_midpoint':float(bound)})
    return certificate,bound


def certify_rows(rows,output):
    certified=[];start=time.perf_counter()
    with ctx.workprec(512):
        for row in rows:
            n=row['n'];_,scales=geometry('70',1,n,512);certs=[];union=arb(0)
            for scale,proposal in zip(scales,row['scales']):
                t=Fraction(round(proposal['t']*10**12),10**12)
                eta=Fraction(math.ceil((proposal['eta']*1.002+1e-9)*10**12),10**12)
                cert,bound=certify_scale(n,scale,t,eta)
                if not cert['passed']:raise ArithmeticError((n,scale.j,cert))
                certs.append(cert);union+=bound
            item={'n':n,'passed':bool(union<ball(Fraction(8,25))),
                  'union_bound_arb':union.str(65),'union_bound_midpoint':float(union),'scales':certs}
            certified.append(item)
            print('CERT',n,item['union_bound_arb'],item['passed'],flush=True)
            output.write_text(json.dumps({'rigorous':True,'C':70,'L0':1,'target_upper_tail':str(Fraction(8,25)),
                'all_passed':all(x['passed'] for x in certified),'rows':certified,
                'method':'Exponential Taylor lower polynomial; outward Arb Bernstein positivity on [0,R]; nonnegative shifted coefficients on [R,infinity); Chernoff S endpoints; union bound.',
                'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')
    return certified


def numerical_scale(n, scale):
    ys=np.array([-1.]+[float(c) for c in scale.corrections])
    k=scale.k;delta=float(scale.delta)
    # Dense low-mean grid plus broad tail; optimize every observed local max.
    grid=np.unique(np.r_[0.,np.linspace(0,8,161),np.linspace(8,k+30,151)])
    ell=np.arange(k+1)
    with np.errstate(divide='ignore',invalid='ignore'):
        pmf=np.exp(-grid[:,None]+ell[None,:]*np.log(grid[:,None])-gammaln(ell+1))
    pmf[0]=0.;pmf[0,0]=1.
    def sup(t, refine=True):
        cs=np.expm1(t*ys)
        values=pmf@cs
        index=np.flatnonzero((values[1:-1]>=values[:-2])&(values[1:-1]>=values[2:]))+1
        best=max(0.,float(values.max()));arg=float(grid[np.argmax(values)])
        if refine:
            def f(lam):
                probs=np.exp(-lam+ell*math.log(lam)-gammaln(ell+1))
                return -float(probs@cs)
            for i in index:
                opt=minimize_scalar(f,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
                if -opt.fun>best:best,arg=-float(opt.fun),float(opt.x)
        return best,arg
    def objective(logt):
        t=math.exp(logt);eta,_=sup(t)
        logG=math.log1p(eta)
        return max(logG-t*delta*(n+1),n*logG-2*t*delta*n)
    # A log grid avoids trusting unimodality across changes in maximizing lambda.
    logs=np.linspace(math.log(1e-5),math.log(40/max(ys.max(),1e-3)),100)
    vals=np.array([objective(v) for v in logs])
    i=int(np.argmin(vals));a=logs[max(0,i-1)];b=logs[min(len(logs)-1,i+1)]
    opt=minimize_scalar(objective,bounds=(a,b),method='bounded',options={'xatol':1e-10})
    t=math.exp(float(opt.x));eta,lam=sup(t);exponent=objective(float(opt.x))
    return {'j':scale.j,'k':k,'delta':str(scale.delta),'t':t,'eta':eta,'maximizer_lambda':lam,
            'log_bound':exponent,'bound':math.exp(exponent),'S_endpoint':n if math.log1p(eta)>t*delta else 1}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=75);ap.add_argument('--end',type=int,default=403)
    ap.add_argument('--n',type=int,nargs='+');ap.add_argument('--certify',action='store_true');args=ap.parse_args()
    out=ROOT/'results_proof_l01';out.mkdir(exist_ok=True)
    if args.certify:
        rows=json.loads((out/'mgf_exploration.json').read_text())['rows']
        if args.n:rows=[r for r in rows if r['n'] in args.n]
        certify_rows(rows,out/'mgf_certificates.json');return
    ns=args.n or list(range(args.start,args.end+1));rows=[];start=time.perf_counter()
    for n in ns:
        _,scales=geometry('70',1,n,512)
        details=[numerical_scale(n,s) for s in scales]
        row={'n':n,'scales':details,'union_bound':sum(x['bound'] for x in details)}
        rows.append(row)
        print(n,row['union_bound'],flush=True)
        (out/'mgf_exploration.json').write_text(json.dumps({'rigorous':False,'C':70,'L0':1,'rows':rows,'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
