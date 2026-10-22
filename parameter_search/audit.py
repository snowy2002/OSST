"""Independent Arb recomputation of certificate formulas and raw-event audit."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from flint import arb,ctx,fmpq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_parameter_search'
COUNT=0


def q(x):
    v=Fraction(str(x));return arb(fmpq(v.numerator,v.denominator))


def check(ok):
    global COUNT
    if not bool(ok):raise ArithmeticError('Independent audit failed')
    COUNT+=1


def maxb(xs):
    result=xs[0]
    for x in xs[1:]:result=result.max(x)
    return result


def minb(xs):
    result=xs[0]
    for x in xs[1:]:result=result.min(x)
    return result


def coef(t,w):
    return 1+maxb([(t+2*w/17).exp()]+[(4*t)**l*math.factorial(l)/math.factorial(2*l)*
                                         (2*w/(2*l+1)).exp() for l in range(1,8)])/32


def audit_risks(vs,ws,ints,L,C,omitted):
    ni=(1-L).exp();U=[];Rf=[];R=[]
    for j,(v,w) in enumerate(zip(vs,ws)):
        delta=q(Fraction(1,2**(j+2)));d2=(2-delta/2)**2;kappa=14-delta/2-8*delta**2
        choices=[v/(v+d2),(-d2/(2*v)).exp()]
        if ints is not None:
            B=ints[j]-1;A=1+delta**2/2+(B-1).max(0)
            choices.append((-1/(2*A*B*ni/(delta**2*d2)+2*B*ni/(3*delta))).exp())
        U.append(minb(choices))
        Rf.append(minb([v/(v+kappa**2),(-kappa**2/(2*v)).exp()]))
        alpha=q('33/32')
        R.append(minb([alpha*w/(alpha*w+kappa**2),
                      (-kappa**2/(2*alpha*w+4*kappa*(w*ni).sqrt()/3)).exp()]))
    heavy=(-31*L-1).exp()+(-C).exp()
    u=sum(U,arb(0))+omitted
    rf=maxb([heavy,omitted/81]+Rf)
    r=maxb([heavy,(-1/(2112*ni)).exp(),q('33/32')*omitted/81]+R)
    return dict(upper=u,lower_finite=rf,lower_general=r,joint=(u+rf).max(r))


def certificate_audit():
    data=json.loads((OUT/'certificate.json').read_text())
    for name,digest in data['source_sha256'].items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest)
    with ctx.workprec(512):
        for name,rows in data['complete_slabs'].items():
            C=144*arb(2).log()**2 if name=='star' else arb(70)
            for row in rows:
                left,right=q(row['left']),q(row['right'])
                vs,ws,ints=[],[],[]
                J=int((256*right**2/C).log().__truediv__(arb(2).log()).floor().unique_fmpz())
                for j in range(J+1):
                    K=(512/C/2**j).sqrt()*(j+3)*arb(2).log()*right+1
                    v=(C*2**j/(512*left**2)).min(q('1/2'))
                    t=K*K/(64*left*(1-v));w=K*K*v/(1-v)
                    candidates=[1+2*(arb.pi()*t).sqrt()*t.exp(),
                       1+2*(arb.pi()*t).sqrt()*(t-2*(j+3)*arb(2).log()+C*2**j/(8*left)).exp()]
                    theta=K/(8*C*2**j).sqrt()
                    if theta<1:candidates.append(1/(1-theta))
                    integral=minb(candidates);F=integral;V=integral**2/4
                    if j==0:
                        coefficient=coef(t,w);F=F.min(coefficient);V=V.min(coefficient**2)
                    factor=(1-left).exp()*2**(2*j+4)
                    vs.append(V*factor);ws.append(F*F*factor);ints.append(integral)
                    for field,value in [('v',vs[-1]),('w',ws[-1]),('integral',integral)]:
                        check(value<=q(row['inputs'][field][j]))
                computed=audit_risks(vs,ws,ints,left,C,arb(0))
                for field,value in computed.items():check(value<=q(row[field]['upper']))
            tail=data['half_lines'][name];H=arb(tail['H']);weights=[]
            for j in range(8):
                t=8*((j+3)*arb(2).log())**2/(C*2**j)
                alpha=(64*t).sqrt();v=C*2**j/(512*H**2)
                d=(2*alpha+(1+alpha**2*C*2**j/512)/H)/(64*(1-v))
                if j==0:
                    F=coef(t*H+d,(alpha+1/H)**2*C/(512*(1-v)))
                else:
                    z=d.min(d-2*(j+3)*arb(2).log()+C*2**j/(8*H))
                    F=1+2*(arb.pi()*(t*H+d)).sqrt()*(t*H+z).exp()
                weights.append(F**2*(1-H).exp()*2**(2*j+4))
                check(weights[-1]<=q(tail['inputs']['w'][j]))
            beta=((512/C/256).sqrt()*11*arb(2).log()+1/H)**2/32
            F=1+2*(arb.pi()*beta*H).sqrt()*(beta*H).exp()
            omitted=q('4/3')*(1024*H**2/C)**2*(1-H).exp()*F**2
            check(omitted<=q(tail['inputs']['omitted']))
            computed=audit_risks(weights,weights,None,H,C,omitted)
            for field,value in computed.items():check(value<=q(tail[field]['upper']))
        for result in data['results']:
            check(q(result['computed']['joint']['upper'])<q(result['target']))
    return len(data['complete_slabs']['star'])+len(data['complete_slabs']['70'])


def experiment_audit():
    from c7016.populations import Population
    count=0;rows=0;independent={}
    protocol=json.loads((OUT/'experiment_protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest)
    for path in sorted((OUT/'cells').glob('*.json')):
        cell=json.loads(path.read_text());rawpath=OUT/cell['raw_file']
        check(hashlib.sha256(rawpath.read_bytes()).hexdigest()==cell['raw_sha256'])
        pop=Population(tuple(cell['population']['multiplicities']),
                       tuple(Fraction(p) for p in cell['population']['probabilities_exact']))
        with np.load(rawpath) as raw:
            for row in cell['rows']:
                C=str(row['C']);method=f'{C}_{row["L0"]}';n=row['n']
                label,tail,eff=pop.truth(n,row['epsilon'])
                check((label,str(tail),eff)==(row['truth'],row['tail_exact'],row['effective']))
                D=raw[f'D_{C}'];lo=raw[f'{method}_lo'];hi=raw[f'{method}_hi']
                reject=raw[f'{method}_reject'];over=raw[f'{method}_over'];short=raw[f'{method}_short']
                check(len(D)==row['trials'])
                check(np.all(lo<=hi))
                check(np.all(reject[lo>n]) and not np.any(reject[hi<=n]))
                check(int(np.sum(over|short))==row['coverage_errors'])
                if label!='Gap':
                    errors=int(np.sum(reject if label=='Yes' else ~reject))
                    check(errors==row['errors'])
                check(int(np.sum(raw[f'{method}_lower_lo']>np.clip(D,1,n)))==row['improvements'])
                check(not np.any(over) if pop.support>=n else np.all(over[lo>pop.support]) and not np.any(over[hi<=pop.support]))
                t=min(n,eff)
                check(not np.any(short) if t==1 else not np.any(short[lo>t-1]) and np.all(short[hi<=t-1]))
                independent[(row['cell'],C)]=len(D);rows+=1
        count+=1
    summary=json.loads((OUT/'experiment_summary.json').read_text())
    check(summary['rows']==rows)
    check(summary['independent_sample_paths']==sum(independent.values()))
    return count,rows


def main():
    slabs=certificate_audit();print('certificate audit passed',COUNT,flush=True)
    cells,rows=experiment_audit()
    payload=dict(passed=True,checks=COUNT,complete_slabs=slabs,half_lines=2,
                 experiment_cells=cells,method_rows=rows,
                 scope='Independent 512-bit Arb formulas, raw hashes, exact truth and event consistency; not formal verification.')
    (OUT/'audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('ALL AUDITS PASS',COUNT,flush=True)


if __name__=='__main__':main()
