"""Exact rational certificate for the stronger concentration analysis.

Run from the repository root: python -m parameter_search.certify
No simulation or floating-point search result is an input to this verifier.
The analytic reduction is in results_parameter_search/proof.tex.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT/'artifacts/joint_constants_submission'
sys.path.insert(0, str(LEGACY))
from rational_interval import I, imin, imax, log_two, pi_interval, decimal_ceiling
from exact_certificate import coefficient_majorant, margins, encode

LOG2, PI = log_two(), pi_interval()
STAR = 144*LOG2**2
CHECKS = []


def negative_exp(x):
    # Exact underflow enclosure: e>2, hence exp(x)<2^-128 for x<=-128.
    # Avoid constructing gigantic integers for exp(-10^15).
    require(x < 0, 'negative exponential argument')
    return I.enclose(0,Q(1,2**128)) if x <= -128 else x.exp()


def require(ok, description):
    if not ok:
        raise ArithmeticError(description)
    CHECKS.append(description)


def risks(vs, ws, integrals, L, C, omitted=I(0)):
    inverse_n = (1-L).exp()
    us, fs, rs = [], [], []
    for j, (v, w) in enumerate(zip(vs, ws)):
        delta = I(Q(1,2**(j+2)))
        d2, k2 = margins(j)
        kappa = 14-delta/2-8*delta**2
        choices = [v/(v+d2), negative_exp(-d2/(2*v))]
        if integrals is not None:
            G = integrals[j]-1
            A = 1+delta**2/2+imax(I(0),G-1)
            require(G > 0, f'positive endpoint envelope L={L.text()}, j={j}')
            denominator = 2*A*G*inverse_n/(delta**2*d2)+2*G*inverse_n/(3*delta)
            choices.append(negative_exp(-1/denominator))
        us.append(imin(*choices))
        fs.append(imin(v/(v+k2), negative_exp(-k2/(2*v))))
        alpha=I(Q(33,32))
        rs.append(imin(alpha*w/(alpha*w+k2),
                       negative_exp(-k2/(2*alpha*w+4*kappa*(w*inverse_n).sqrt()/3))))
    heavy = (-31*L-1).exp()+(-C).exp()
    U = sum(us,I(0))+omitted
    Rf = imax(heavy,omitted/81,*fs)
    R = imax(heavy,negative_exp(-1/(2112*inverse_n)),I(Q(33,32))*omitted/81,*rs)
    return dict(upper=U, lower_finite=Rf, lower_general=R, joint=imax(U+Rf,R))


def slab(C, numerator, denominator=10):
    left,right=I(Q(numerator,denominator)),I(Q(numerator+1,denominator))
    ratio=256*right**2/C
    jmax=0
    while not I(2**(jmax+1))>ratio:
        jmax+=1
    require(I(2**(jmax+1))>ratio,f'complete scale coverage {numerator}/{denominator}')
    vs,ws,integrals=[],[],[]
    for j in range(jmax+1):
        q=(j+3)*LOG2
        K=(512/C/2**j).sqrt()*q*right+1
        v=imin(I(Q(1,2)),C*2**j/(512*left**2))
        T=K*K/(64*left*(1-v)); W=K*K*v/(1-v)
        A=C*2**j/(8*left); prefactor=2*(PI*T).sqrt()
        candidates=[1+prefactor*T.exp(),1+prefactor*(T-2*q+A).exp()]
        theta=K/(8*C*2**j).sqrt()
        if theta<1:
            candidates.append(1/(1-theta))
        integral=imin(*candidates)
        F,V=integral,integral**2/4
        if j==0:
            coeff=coefficient_majorant(T,W)
            F,V=imin(F,coeff),imin(V,coeff**2)
        factor=(1-left).exp()*2**(2*j+4)
        vs.append(V*factor);ws.append(F*F*factor);integrals.append(integral)
    out=risks(vs,ws,integrals,left,C)
    out.update(left=str(Q(numerator,denominator)),right=str(Q(numerator+1,denominator)),
               inputs=dict(v=[v.upper_fraction() for v in vs],w=[w.upper_fraction() for w in ws],
                           integral=[v.upper_fraction() for v in integrals]))
    return out


def half_line(C,symbolic=False,Hvalue=40):
    H=I(Hvalue); ws=[]
    for j in range(8):
        q=(j+3)*LOG2
        t=I(Q((j+3)**2,18*2**j)) if symbolic else 8*q*q/(C*2**j)
        alpha=(64*t).sqrt();xi=C*2**j/512;v=xi/H**2
        require(v<I(Q(1,2)),f'half-line shift {j}')
        d=(2*alpha+(1+alpha**2*xi)/H)/(64*(1-v));th=t*H+d
        if j==0:
            require((2*t).exact_equal(1) if symbolic else 2*t<1,'scale-zero decay')
            W=(alpha+1/H)**2*C/(512*(1-v));F=coefficient_majorant(th,W)
            require(H>=14,'polynomial envelope decay')
        else:
            require(H*(1-2*t)>1,f'low scale decay {j}')
            z=imin(d,d-2*q+C*2**j/(8*H))
            F=1+2*(PI*th).sqrt()*(t*H+z).exp()
        ws.append(F*F*(1-H).exp()*2**(2*j+4))
    beta2=((512/C/256).sqrt()*11*LOG2+1/H)**2/32
    require(H*(1-2*beta2)>5,'omitted scale decay')
    Ft=1+2*(PI*beta2*H).sqrt()*(beta2*H).exp()
    omitted=I(Q(4,3))*(1024*H*H/C)**2*(1-H).exp()*Ft**2
    out=risks(ws,ws,None,H,C,omitted)
    out.update(H=Hvalue,inputs=dict(w=[w.upper_fraction() for w in ws],
                                  omitted=omitted.upper_fraction()))
    return out


def counting(C,L0):
    # g(x)=x-1-ln x >= (x-1)^2/(2x), x>=1.
    # Cover continuous L in [1,L0], which contains all possible integer n.
    bounds=[]
    for a in range(4,4*L0):
        left,right=I(Q(a,4)),I(Q(a+1,4))
        x=C/right
        require(x>1,f'counting positive rate {a}/4')
        bounds.append(negative_exp(-((left-1).exp())*(x-1)**2/(2*x)))
    result=imax(*bounds)
    require(result<I(Q(1,10**10)),f'counting error below 1e-10 L0={L0}')
    return result


def main():
    started=time.monotonic(); allrows={};tails={};results=[]
    for name,C,symbolic in [('star',STAR,True),('70',I(70),False)]:
        # Parameter lemma, extended down to L=14.
        require(C>69 and C<=70,'C range')
        require(256*I(14)**2/C>1,'J nonnegative')
        require(64*I(14)**2/(C*I(13).exp())<1,'polynomial endpoint below one')
        require((512/C).sqrt()*3*LOG2+I(Q(1,14))<6,'degree bound')
        rows=[]
        for i in range(140,400):
            rows.append(slab(C,i))
            if (i+1)%50==0:
                print(name,'complete slabs through',Q(i+1,10),flush=True)
        tail=half_line(C,symbolic)
        allrows[name],tails[name]=rows,tail
        targets={'star':{14:Q(15,10000),15:Q(38,100000),16:Q(31,100000),18:Q(28,100000)},
                 '70':{14:Q(9,10000),15:Q(81,10**6),16:Q(57,10**6),18:Q(40,10**6)}}[name]
        for L0,target in targets.items():
            count=counting(C,L0)
            selected=rows[(L0-14)*10:]
            peak={key:imax(count,tail[key],*(r[key] for r in selected))
                  for key in ['upper','lower_finite','lower_general','joint']}
            require(peak['joint']<I(target),f'{name},{L0}: advertised bound {target}')
            require(target<I(Q(1,100)),f'{name},{L0}: one run below one percent')
            results.append(dict(C=name,L0=L0,target=str(target),computed=peak,counting=count))
    sources=['parameter_search/certify.py','artifacts/joint_constants_submission/exact_certificate.py',
             'artifacts/joint_constants_submission/rational_interval.py']
    payload=dict(passed=True,checks=len(CHECKS),check_descriptions=CHECKS,
                 arithmetic='160-bit dyadic outward intervals, exact integer/rational operations',
                 results=results,complete_slabs=allrows,half_lines=tails,
                 source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                 scope='Finite obligations of the accompanying analytic argument, not a formal proof assistant.')
    path=ROOT/'results_parameter_search/certificate.json'
    path.write_text(json.dumps(encode(payload),indent=2)+'\n')
    print('PASS',len(CHECKS),'obligations,',round(time.monotonic()-started,1),'seconds',flush=True)
    for row in results:
        print(row['C'],row['L0'],decimal_ceiling(row['computed']['joint'].upper_fraction(),9),flush=True)


if __name__=='__main__':
    main()
