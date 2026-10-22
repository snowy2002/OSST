"""Floating-point candidate search. Results here are NOT proof certificates."""
import csv
import math
from pathlib import Path

STAR = 144*math.log(2)**2


def exp(x):
    return math.exp(x) if x < 700 else math.inf


def coefficient(t, w):
    return 1+max([math.exp(t+2*w/17)]+[
        (4*t)**ell*math.factorial(ell)/math.factorial(2*ell)*math.exp(2*w/(2*ell+1))
        for ell in range(1, 8)])/32


def risks(vs, ws, integrals, L, C, omitted=0):
    n = math.exp(L-1)
    upper, finite, general = [], [], []
    for j, (v, w) in enumerate(zip(vs, ws)):
        delta = 2**(-j-2)
        d, kappa = 2-delta/2, 14-delta/2-8*delta**2
        choices = [v/(v+d*d), math.exp(-d*d/(2*v))]
        if integrals is not None:
            # Center at 1+rho_bound; endpoint Bernoulli dominates its MGF.
            G = integrals[j]-1
            A = 1+delta*delta/2+max(0, G-1)
            choices.append(math.exp(-1/(2*A*G/(n*delta**2*d*d)+2*G/(3*delta*n))))
        upper.append(min(choices))
        finite.append(min(v/(v+kappa*kappa), math.exp(-kappa*kappa/(2*v))))
        alpha = 33/32
        general.append(min(alpha*w/(alpha*w+kappa*kappa),
                           math.exp(-kappa*kappa/(2*alpha*w+4*kappa*math.sqrt(w/n)/3))))
    heavy = math.exp(-31*L-1)+math.exp(-C)
    U = sum(upper)+omitted
    Rf = max([heavy, omitted/81]+finite)
    R = max([heavy, math.exp(-n/2112), (33/32)*omitted/81]+general)
    return U, Rf, R, max(U+Rf, R)


def slab(C, left, width=.1):
    right = left+width
    vs, ws, integrals = [], [], []
    jmax = math.floor(math.log2(256*right*right/C))
    for j in range(jmax+1):
        q = (j+3)*math.log(2)
        K = math.sqrt(512/C/2**j)*q*right+1
        v = min(.5, C*2**j/(512*left*left))
        t = K*K/(64*left*(1-v)); w = K*K*v/(1-v)
        A = C*2**j/(8*left); theta = K/math.sqrt(8*C*2**j)
        vals = [1+2*math.sqrt(math.pi*t)*exp(t),
                1+2*math.sqrt(math.pi*t)*exp(t-2*q+A)]
        if theta < 1:
            vals.append(1/(1-theta))
        integral = min(vals)
        F, V = integral, integral**2/4
        if j == 0:
            f = coefficient(t,w); F, V = min(F,f), min(V,f*f)
        factor = math.exp(1-left)*2**(2*j+4)
        vs.append(V*factor); ws.append(F*F*factor); integrals.append(integral)
    return risks(vs,ws,integrals,left,C)


def tail(C,H=32):
    ws=[]
    for j in range(8):
        q=(j+3)*math.log(2); t=8*q*q/(C*2**j); a=math.sqrt(64*t)
        v=C*2**j/(512*H*H)
        if not (v<.5 and (j==0 or H*(1-2*t)>1)):
            return (math.inf,)*4
        d=(2*a+(1+a*a*C*2**j/512)/H)/(64*(1-v)); th=t*H+d
        if j==0:
            W=(a+1/H)**2*C/(512*(1-v)); F=coefficient(th,W)
        else:
            z=min(d,d-2*q+C*2**j/(8*H)); F=1+2*math.sqrt(math.pi*th)*math.exp(t*H+z)
        ws.append(F*F*math.exp(1-H)*2**(2*j+4))
    beta=(math.sqrt(512/C/256)*11*math.log(2)+1/H)**2/32
    if H*(1-2*beta)<=5:
        return (math.inf,)*4
    Ft=1+2*math.sqrt(math.pi*beta*H)*math.exp(beta*H)
    omitted=4/3*(1024*H*H/C)**2*math.exp(1-H)*Ft*Ft
    return risks(ws,ws,None,H,C,omitted)


def main():
    rows=[]
    for C in [STAR]+list(range(70,151)):
        half=tail(C,40)
        slabs=[slab(C,i/10) for i in range(100,400)]
        for L0 in range(10,31):
            peak=max([half[3]]+[r[3] for r in slabs[(L0-10)*10:]])
            repeats=next((r for r in range(1,202,2)
                          if sum(math.comb(r,k)*peak**k*(1-peak)**(r-k)
                                 for k in range((r+1)//2,r+1))<.01),None) if peak<.5 else None
            rows.append(dict(C=C,L0=L0,estimated_bound=peak,repeats=repeats,
                             total_coefficient=C*repeats if repeats else None))
        print(C,[(x['L0'],round(x['estimated_bound'],6)) for x in rows[-21:] if x['L0']<=18],flush=True)
    path=Path('results_parameter_search/exploration.csv')
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
