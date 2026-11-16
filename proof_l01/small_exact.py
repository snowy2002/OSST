"""Small-n common zero/singleton event, with analytic homogenization.

The numerical optimizer only proposes rational y and target bounds.
Separate Arb interval certificates establish the uniform lambda bound.
See homogenization.txt for the analytic reduction.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln,logsumexp
from flint import arb,arb_poly,ctx,fmpq
from parameter_search.algorithm import geometry
from paper_reference import certified_integer
from proof_l01.mgf import numerical_scale,certify_scale


def ball(q):
    if isinstance(q,arb):return q
    q=Q(q);return arb(fmpq(q.numerator,q.denominator))


def thresholds(n,scales,S=None,cut=None,anchor=2):
    good=[];bad=[]
    with ctx.workprec(512):
        for scale in scales:
            if not all((c>=0 if ell%2 else c<=0) for ell,c in enumerate(scale.corrections,1)):
                raise ArithmeticError('Alternating correction sign not certified')
            accepted=(scale.j>=cut if cut is not None else
                      all(scale.corrections[ell-1]<=ball(scale.delta)
                          for ell in range(3,scale.k+1,2)))
            if accepted:
                if not scale.corrections[0]>=ball(scale.delta):
                    raise ArithmeticError('Nonmonotone singleton envelope')
                good.append(scale)
            else:bad.append(scale)
        if S is None:return good,bad,[]
        qs=[];qhs=[]
        for scale in good:
            q=arb(0)
            for ell in range(3,scale.k+1,2):q=q.max(scale.corrections[ell-1])
            if not scale.corrections[0]>=q:raise ArithmeticError('Nonmonotone q envelope')
            qs.append(q)
            qh=arb(0)
            for ell in range(anchor+1,scale.k+1,2):qh=qh.max(scale.corrections[ell-1])
            qhs.append(qh)
        z=[]
        for s in range(S+1):
            envelopes=[((scale.corrections[0]-q)*s
                         +(q-ball(scale.delta))*S-ball(scale.delta)*n-q+qh)/(1+q)
                        for scale,q,qh in zip(good,qs,qhs)]
            if not envelopes:z.append(-1);continue
            upper=envelopes[0]
            for v in envelopes[1:]:upper=upper.max(v)
            z.append(certified_integer(upper,'ceil')-1)
        if any(a>b for a,b in zip(z,z[1:])):raise ArithmeticError('Nonmonotone z')
        return good,bad,z


def numerical_common(S,z,fixed_y=None):
    ss=np.array([s for s in range(S+1) if z[s]>=0],dtype=float)
    if not len(ss):return {'S':S,'y':Q(1,2).__str__(),'bound':0.,'lambda':1.}
    zs=np.array([z[int(s)] for s in ss]);lcomb=gammaln(S+1)-gammaln(ss+1)-gammaln(S-ss+1)
    grid=np.r_[np.linspace(1,8,141),np.linspace(8,30,51)]
    def maximum(logy,refine=True):
        y=math.exp(logy)
        def values(lams):
            lams=np.atleast_1d(lams);ex=np.exp(-lams);b=lams*ex;h=1-(1-y+lams)*ex
            return logsumexp(lcomb+np.log(b)[:,None]*ss+np.log(h)[:,None]*(S-ss)-logy*zs,axis=1)
        vals=values(grid);best=float(vals.max());arg=float(grid[vals.argmax()])
        idx=np.flatnonzero((vals[1:-1]>=vals[:-2])&(vals[1:-1]>=vals[2:]))+1
        if refine:
            for i in idx:
                opt=minimize_scalar(lambda lam:-float(values([lam])[0]),bounds=(grid[i-1],grid[i+1]),method='bounded')
                if -opt.fun>best:best,arg=-float(opt.fun),float(opt.x)
        return best,arg
    if fixed_y is not None:
        y=Q(fixed_y);logbound,lam=maximum(math.log(float(y)))
        return {'S':S,'y':str(y),'bound':math.exp(logbound),'lambda':lam}
    # The outer log-y objective is convex as a supremum of log-MGFs.
    opt=minimize_scalar(lambda a:maximum(a)[0],bounds=(-16.,-1e-7),method='bounded',options={'xatol':1e-7})
    y=Q(max(1,round(math.exp(opt.x)*10**12)),10**12)
    logbound,lam=maximum(math.log(float(y)))
    return {'S':S,'y':str(y),'bound':math.exp(logbound),'lambda':lam}


def certify_common(S,z,y,target,*,bits=512,R=24,max_depth=20):
    with ctx.workprec(bits):
        y=ball(y);target=ball(target)
        if not (y>0 and y<1 and target>0):raise ValueError('Invalid certificate parameters')
        coeff=[ball(math.comb(S,s))*y**(-z[s]) if z[s]>=0 else arb(0) for s in range(S+1)]
        if all(c==0 for c in coeff):return {'passed':True,'zero_event':True,'nodes':0,'leaves':[]}
        poly=arb_poly(coeff)
        def endpoint_bound(a,b):
            a,b=ball(a),ball(b)
            single=a*(-a).exp()
            other=1-(1-y+b)*(-b).exp()
            return other**S*poly(single/other)
        tail_single=ball(R)*(-ball(R)).exp()
        tail_bound=poly(tail_single)
        if not tail_bound<target:return {'passed':False,'reason':'tail','tail':tail_bound.str(50)}
        pending=[(Q(1),Q(R),0)];nodes=0;leaves=[]
        while pending:
            a,b,depth=pending.pop();nodes+=1
            bound=endpoint_bound(a,b)
            if bound<target:leaves.append([str(a),str(b)]);continue
            if depth>=max_depth:return {'passed':False,'reason':'depth','interval':[str(a),str(b)],'bound':bound.str(50),'target':target.str(50)}
            mid=(a+b)/2;pending.extend([(a,mid,depth+1),(mid,b,depth+1)])
        return {'passed':True,'nodes':nodes,'leaves':leaves,'R':R,'tail_bound':tail_bound.str(50),'bits':bits}


def explore(ns,output,cut=None,anchor=2):
    rows=[]
    for n in ns:
        _,scales=geometry('70',1,n,512);good,bad,z=thresholds(n,scales,cut=cut)
        mgfs=[numerical_scale(n,s) for s in bad]
        lowrisk=sum(x['bound'] for x in mgfs)
        # One common bound for every S individually; retain all proposals.
        common=[]
        _,_,largest_z=thresholds(n,scales,n,cut=cut,anchor=anchor)
        first=numerical_common(n-1,largest_z[:n])
        shared_y=first['y']
        for S in range(1,n+1):
            _,_,zS=thresholds(n,scales,S,cut=cut,anchor=anchor)
            proposal=numerical_common(S-1,zS[:S],fixed_y=shared_y)
            if proposal['bound']>max(.315-lowrisk,0):
                proposal=numerical_common(S-1,zS[:S])
            proposal['r']=S-1;proposal['S']=S
            proposal['z']=zS[:S];common.append(proposal)
        worst=max(common,key=lambda x:x['bound'])
        heavy_mean=70/(1+math.log(n))
        heavy_bound=sum(heavy_mean**ell/math.factorial(ell) for ell in range(anchor))*math.exp(-heavy_mean)
        for p in common:
            S=p['S']
            p['low_scale_bound']=sum(math.exp(S*math.log1p(m['eta'])-m['t']*float(Q(m['delta']))*(n+S)) for m in mgfs)
            heavy_mean=70*n/(S*(1+math.log(n)))
            p['heavy_bound']=sum(heavy_mean**ell/math.factorial(ell) for ell in range(anchor))*math.exp(-heavy_mean)
            p['total_bound']=p['bound']+p['low_scale_bound']+p['heavy_bound']
        row={'n':n,'cut':cut,'anchor':anchor,'good_scales':[s.j for s in good],'bad_scales':[s.j for s in bad],
             'z':z,'mgfs':mgfs,'low_scale_bound':lowrisk,'common':common,
             'heavy_bound':heavy_bound,'worst_common':worst,'union_bound':max(p['total_bound'] for p in common)}
        rows.append(row)
        output.write_text(json.dumps({'rigorous':False,'rows':rows},indent=2)+'\n')
        print(n,row['union_bound'],'low',lowrisk,'common',worst['bound'],'S',worst['S'],flush=True)
    return rows


def certify(rows,output):
    records=[]
    with ctx.workprec(512):
        for row in rows:
            n=row['n'];_,scales=geometry('70',1,n,512);good,bad,z=thresholds(n,scales,cut=row.get('cut'))
            mgfs=[];lowrisk=arb(0)
            for scale,proposal in zip(bad,row['mgfs']):
                t=Q(round(proposal['t']*10**12),10**12)
                eta=Q(math.ceil((proposal['eta']*1.002+1e-9)*10**12),10**12)
                cert,bound=certify_scale(n,scale,t,eta)
                if not cert['passed']:raise ArithmeticError((n,scale.j,cert))
                mgfs.append(cert);lowrisk+=bound
            common=[];total=arb(0)
            for proposal in row['common']:
                target=Q(math.ceil((proposal['bound']*1.002+1e-7)*10**10),10**10)
                _,_,zS=thresholds(n,scales,proposal['S'],cut=row.get('cut'),anchor=row.get('anchor',2))
                zS=zS[:proposal['S']]
                if zS!=proposal['z']:raise ArithmeticError('Threshold mismatch')
                cert=certify_common(proposal['S']-1,zS,Q(proposal['y']),target)
                if not cert['passed']:raise ArithmeticError((n,proposal['S'],cert))
                cert.update({'S':proposal['S'],'y':proposal['y'],'z':zS})
                S=proposal['S'];low_S=arb(0)
                for m in mgfs:
                    low_S+=(S*(1+ball(Q(m['eta']))).log()-ball(Q(m['t']))*ball(Q(m['delta']))*(n+S)).exp()
                heavy_mean=70*arb(n)/(S*(1+arb(n).log()))
                heavy_S=sum((heavy_mean**ell/math.factorial(ell) for ell in range(row.get('anchor',2))),arb(0))*(-heavy_mean).exp()
                total_S=low_S+ball(target)+heavy_S
                cert.update({'target':str(target),'low_scale_bound':low_S.str(50),
                             'heavy_bound':heavy_S.str(50),'total_bound':total_S.str(60)})
                total=total.max(total_S)
                common.append(cert)
            record={'n':n,'cut':row.get('cut'),'anchor':row.get('anchor',2),'passed':bool(total<ball(Q(1,3))-ball(Q(1,10**20))),
                    'z':z,'good_scales':[s.j for s in good],'bad_scales':[s.j for s in bad],
                    'low_scale_certificate':mgfs,
                    'common_certificates':common,'union_bound_arb':total.str(65),'union_bound':float(total)}
            records.append(record)
            output.write_text(json.dumps({'rigorous':True,'C':70,'L0':1,'all_passed':all(r['passed'] for r in records),'rows':records},indent=2)+'\n')
            print('CERT',n,record['union_bound'],record['passed'],flush=True)
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=1);ap.add_argument('--end',type=int,default=73)
    ap.add_argument('--n',type=int,nargs='+');ap.add_argument('--certify',action='store_true');ap.add_argument('--refine',action='store_true');args=ap.parse_args()
    out=ROOT/'results_proof_l01';out.mkdir(exist_ok=True)
    if args.refine:
        path=out/'small_exploration.json';rows=json.loads(path.read_text())['rows']
        for i,row in enumerate(rows):
            if (args.n and row['n'] not in args.n) or (not args.n and row['union_bound']<.31):continue
            n=row['n'];_,scales=geometry('70',1,n,512);base=len(row['bad_scales']);best=row
            for cut in range(max(0,base-2),min(len(scales),base+2)+1):
                if cut==base:continue
                # Exclude a common envelope already >=1 when S=n and F1=0.
                _,_,z0=thresholds(n,scales,n,cut=cut)
                if z0[0]>=0:continue
                candidate=explore([n],out/'small_refinement_last.json',cut=cut)[0]
                if candidate['union_bound']<best['union_bound']:best=candidate
            if best['union_bound']>.325:
                for anchor in [4,6]:
                    for cut in range(max(0,base-2),min(len(scales),base+1)+1):
                        _,_,z0=thresholds(n,scales,n,cut=cut,anchor=anchor)
                        if z0[0]>=0:continue
                        candidate=explore([n],out/'small_refinement_last.json',cut=cut,anchor=anchor)[0]
                        if candidate['union_bound']<best['union_bound']:best=candidate
            rows[i]=best;path.write_text(json.dumps({'rigorous':False,'rows':rows},indent=2)+'\n')
            print('BEST',n,best.get('cut'),best['union_bound'],flush=True)
    elif args.certify:
        rows=json.loads((out/'small_exploration.json').read_text())['rows']
        if args.n:rows=[r for r in rows if r['n'] in args.n]
        certify(rows,out/'small_certificates.json')
    else:explore(args.n or list(range(args.start,args.end+1)),out/'small_exploration.json')


if __name__=='__main__':main()
