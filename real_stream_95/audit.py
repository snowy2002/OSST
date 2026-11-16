"""Independent saved-data checks and literal stream replay; no old file edits."""
from collections import Counter
import argparse
from fractions import Fraction
import json
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from real_stream_95.run import ROOT, OUT, sha, seed, dump, budget_plan, interval, amplification_failure
from real_l01 import algorithm as alg
from followup.core import exact_output


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=type(OUT),default=OUT)
    OUT=parser.parse_args().out
    start=time.monotonic();checks=Counter()
    def require(value, label):
        if not bool(value):
            raise ArithmeticError(label)
        checks[label]+=1
    protocol=json.loads((OUT/'protocol.json').read_text())
    table=pd.read_csv(OUT/'results.csv')
    alpha=.05/len(table);trials=protocol['trials']
    for name,digest in protocol['source_sha256'].items():
        require(sha(ROOT/name)==digest,'current source hash')
        require(sha(OUT/'source_snapshot'/name)==digest,'frozen source hash')
    require(amplification_failure()+Fraction(1,1000)<Fraction(1,20),'exact 23-copy total risk')
    # e^7>1000 proves log(1000)<7 using rational arithmetic only.
    require(sum(Fraction(7)**k/math.factorial(k) for k in range(25))>1000,'Poisson cap constant')
    moments=[]
    for r in protocol['populations']:
        N,S=r['N'],r['S'];counts=np.asarray(r['counts'],dtype=np.int64)
        pool=np.repeat(np.arange(S,dtype=np.int32),counts)
        plans=json.loads((OUT/'plans'/f'{r["id"]}.json').read_text())
        for mode in ['without_replacement','iid_with_replacement']:
            rng=np.random.default_rng(seed('direct_stream_v1',r['id'],mode,0))
            labels=rng.permutation(pool) if mode=='without_replacement' else pool[rng.integers(0,N,size=N)]
            # A literal Python loop cross-checks the batched histogram reduction.
            manual=Counter(int(x) for x in labels[:min(N,4096)])
            require(all(manual.get(i,0)==v for i,v in enumerate(np.bincount(labels[:min(N,4096)],minlength=S))),
                    'one-label-at-a-time reduction')
            with np.load(OUT/'raw'/f'{r["id"]}_{mode}.npz',allow_pickle=False) as z:
                for fidx,(fraction,B) in enumerate(zip(protocol['fractions'],plans['budgets'])):
                    census=z['census_counts'][fidx]
                    require(np.all(census.sum(axis=1)==B),'prefix sample count')
                    require(np.array_equal(census[0],np.bincount(labels[:B],minlength=S)),'first stream replay')
                    if mode=='without_replacement':
                        require(np.all(census<=counts),'no record reused without replacement')
                        if B==N:
                            require(np.all(census==counts),'full stream multiset')
                    D=np.count_nonzero(census,axis=1)
                    require(np.array_equal(D,z[f'{fidx}_observed']),'observed species output')
                    f1,f2=(census==1).sum(axis=1),(census==2).sum(axis=1)
                    require(np.array_equal(D+f1*(f1-1)/(2*(f2+1)),z[f'{fidx}_chao1_bc']), 'Chao formula')
                    expected=float(np.sum(1-hypergeom.pmf(0,N,counts,B))) if mode=='without_replacement' else float(np.sum(-np.expm1(B*np.log1p(-counts/N))))
                    moments.append(dict(population=r['id'],sampling=mode,fraction=fraction,
                        observed_mean=float(D.mean()),analytic_mean=expected,
                        standard_error=float(D.std(ddof=1)/np.sqrt(trials))))
                    for k in ['single','amplified','count95']:
                        p=plans['plans'][fidx][k]
                        require(p==budget_plan(B,23 if k=='amplified' else 1,'count' if k=='count95' else 'osst'),
                                'budget-only parameter replay')
                        if p is None:
                            continue
                        arr=z[k+'_counts'][fidx];sizes=z[k+'_sizes'][fidx];overflow=z[k+'_overflow'][fidx]
                        require(np.array_equal(overflow,sizes.sum(axis=1)>B),'overflow definition')
                        require(np.all(arr.sum(axis=-1)[~overflow]==sizes[~overflow]),'per-copy sample totals')
                        if mode=='without_replacement':
                            require(np.all(arr.sum(axis=1)<=counts),'copies share no repeated record')
                        rr=np.random.default_rng(seed('poisson_sizes_v1',r['id'],mode,0,fidx,k))
                        ms=rr.poisson(p['m'],size=p['copies'])
                        require(np.array_equal(ms,sizes[0]),'Poisson size replay')
                        if not overflow[0]:
                            pos=0
                            for j,M in enumerate(ms):
                                require(np.array_equal(arr[0,j],np.bincount(labels[pos:pos+M],minlength=S)),'disjoint chunk replay')
                                pos+=M
                        if k=='single' and not overflow[0]:
                            plan=alg.parameters('70',1,p['n'],Fraction(1,10),bits=1024)
                            v=arr[0,0];hist=np.bincount(v,minlength=plan.degree+1)
                            low=hist[1:plan.degree+1]
                            exact=exact_output(np.count_nonzero(v),low,plan,precisions=(1024,2048,4096))
                            require(exact['lower']==z[f'{fidx}_C70_L1'][0], 'high precision integer score')
                            require((exact['decision']=='No')==z[f'{fidx}_C70_L1_reject'][0], 'high precision binary decision')
                        if k=='amplified' and fidx==5 and not overflow[0]:
                            plan=alg.parameters('70',1,p['n'],Fraction(1,10),bits=1024)
                            copies=[];rejects=[]
                            for v in arr[0]:
                                hist=np.bincount(v,minlength=plan.degree+1)
                                exact=exact_output(np.count_nonzero(v),hist[1:plan.degree+1],plan,precisions=(1024,2048,4096))
                                copies.append(exact['lower']);rejects.append(exact['decision']=='No')
                            require(np.median(copies)==z[f'{fidx}_C70_L1_median23'][0],'independent median reconstruction')
                            require((sum(rejects)>=12)==z[f'{fidx}_C70_L1_median23_reject'][0],'independent majority reconstruction')
                    subset=table[(table.population==r['id'])&(table.sampling==mode)&(table.fraction==fraction)]
                    for _,row in subset.iterrows():
                        values=z[f'{fidx}_{row.method}'];valid=np.isfinite(values)
                        close=valid&(np.abs(values-S)<=.05*S)
                        require(int(valid.sum())==row.completed,'completion accounting')
                        require(int(close.sum())==row.within5_count,'5-percent tolerance reconstructed')
                        if valid.any():
                            require(np.isclose(values[valid].mean(),row['mean']),'mean output reconstructed')
                            require(np.isclose(np.abs(values[valid]-S).mean()/S,row.mape),'relative error reconstructed')
                        for a,cols in [(.05,('within5_ci95_lo','within5_ci95_hi')),(alpha,('within5_sim95_lo','within5_sim95_hi'))]:
                            ci=interval(int(close.sum()),trials,a)
                            require(all(np.isclose(ci[i],row[c]) for i,c in enumerate(cols)),'confidence intervals reconstructed')
                        if not pd.isna(row.n):
                            n=int(row.n);tail=N;eff=None
                            for j,c in enumerate(sorted(counts,reverse=True),1):
                                tail-=int(c)
                                if Fraction(tail,N)<=Fraction(1,10):eff=j;break
                            ok=valid&(values>=min(eff,n))&(values<=min(S,n))
                            require(np.isclose(ok.mean(),row.lower_interval_coverage),'coverage independently reconstructed')
                            rej=z[f'{fidx}_{row.method}_reject']
                            if S<=n:ok&=~rej
                            elif Fraction(sum(sorted(map(int,counts),reverse=True)[n:]),N)>=Fraction(1,10):ok&=rej
                            require(np.isclose(ok.mean(),row.joint_success_rate),'joint event independently reconstructed')
    pd.DataFrame(moments).to_csv(OUT/'sampling_moment_checks.csv',index=False)
    if (OUT/'reported_lower.csv').exists():
        wrapped=pd.read_csv(OUT/'reported_lower.csv')
        for (pid,mode),group in wrapped.groupby(['population','sampling']):
            with np.load(OUT/'raw'/f'{pid}_{mode}.npz') as z:
                for _,r in group.iterrows():
                    fidx=protocol['fractions'].index(r.fraction)
                    k='amplified' if r.original_method=='C70_L1_median23' else 'single'
                    D=np.count_nonzero(z[k+'_counts'][fidx].sum(axis=1),axis=1)
                    values=np.maximum(D,z[f'{fidx}_{r.original_method}'])
                    valid=np.isfinite(values);close=valid&(np.abs(values-r.S)<=.05*r.S)
                    require(int(close.sum())==r.within5_count,'report wrapper tolerance')
                    require(int(valid.sum())==r.completed,'report wrapper completion')
                    require(int(np.sum(valid&(values>D)))==r.gain_count,'report wrapper increment')
                    if valid.any():
                        require(np.isclose(values[valid].mean(),r['mean']),'report wrapper mean')
                        require(np.isclose(np.abs(values[valid]-r.S).mean()/r.S,r.mape),'report wrapper error')
        # Source-file order is directly verifiable without random sampling.
        fish=pd.read_csv(ROOT/'data/inch_lake_fish.csv')
        for _,r in pd.read_csv(OUT/'fish_source_order.csv').iterrows():
            part=fish[(fish.year==r.year)&(fish.gear==r.gear)]
            require(part.iloc[:int(r.samples)].species.nunique()==r.observed,'actual file row order')
            require(part.species.nunique()==r.S and len(part)==r.N,'actual file reference totals')
    dump(OUT/'audit.json',dict(passed=True,checks=sum(checks.values()),categories=checks,
         seconds=time.monotonic()-start,
         limitations='Arithmetic replay is not a new universal proof. One literal stream per population/model and selected high-precision outputs replayed; every saved result row checked.'))
    print('PASS',sum(checks.values()),'checks')


if __name__=='__main__':
    main()
