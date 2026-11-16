"""Post-hoc descriptive reporting wrapper; no new draws or modified algorithm.

Report max(number of distinct labels actually consumed, original lower output).
This preserves a valid lower bound but does not promise accurate total richness.
"""
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from real_stream_95.run import ROOT, OUT, dump, interval, seed, budget_plan, score


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=type(OUT),default=OUT)
    OUT=parser.parse_args().out
    protocol=json.loads((OUT/'protocol.json').read_text())
    base=pd.read_csv(OUT/'results.csv');rows=[]
    for r in protocol['populations']:
        for mode in ['without_replacement','iid_with_replacement']:
            with np.load(OUT/'raw'/f'{r["id"]}_{mode}.npz') as z:
                for fidx,f in enumerate(protocol['fractions']):
                    for method,k in [('C70_L1','single'),('C70_L14','single'),('C70_L1_median23','amplified')]:
                        row=base[(base.population==r['id'])&(base.sampling==mode)&(base.fraction==f)&(base.method==method)].iloc[0].to_dict()
                        original=z[f'{fidx}_{method}'];valid=np.isfinite(original)
                        union=np.count_nonzero(z[k+'_counts'][fidx].sum(axis=1),axis=1)
                        values=np.maximum(union,original);v=values[valid];S=r['S']
                        close=valid&(np.abs(values-S)<=.05*S)
                        row.update(method=method+'_report_maxD',original_method=method,
                            mean=float(v.mean()) if len(v) else None,
                            q025=float(np.quantile(v,.025)) if len(v) else None,
                            q975=float(np.quantile(v,.975)) if len(v) else None,
                            mape=float(np.abs(v-S).mean()/S) if len(v) else None,
                            mae=float(np.abs(v-S).mean()) if len(v) else None,
                            within5_rate=float(close.mean()),within5_count=int(close.sum()),
                            exact_rate=float(np.mean(valid&(values==S))),over_rate=float(np.mean(valid&(values>S))),
                            mean_actual_distinct=float(union[valid].mean()) if len(v) else None,
                            mean_gain_over_distinct=float(np.mean(v-union[valid])) if len(v) else None,
                            gain_count=int(np.sum(valid&(values>union))),
                            interpretation='Post-hoc report max(D_union, original clipped lower). Not original paper output or an unbiased point estimator.')
                        # The wrapper is NOT clipped at n; change the interval target.
                        row['lower_interval_coverage']=float(np.mean(valid&(values>=min(row['effective_90'],row['n']))&(values<=S))) if not pd.isna(row['n']) else None
                        for key in ['bias','rmse','median','joint_success_rate','theoretical_target','output_type']:
                            row.pop(key,None)
                        rows.append(row)
    d=pd.DataFrame(rows)
    for idx,row in d.iterrows():
        for alpha,cols in [(.05,('within5_ci95_lo','within5_ci95_hi')),(.05/len(d),('within5_sim95_lo','within5_sim95_hi'))]:
            a,b=interval(int(row.within5_count),int(row.trials),alpha)
            d.loc[idx,cols[0]]=a;d.loc[idx,cols[1]]=b
    d.to_csv(OUT/'reported_lower.csv',index=False)
    # Literal source-file order case study. It is not a known capture timeline.
    fish=pd.read_csv(ROOT/'data/inch_lake_fish.csv')
    rows=[]
    for year in [2007,2008]:
        part=fish[(fish.year==year)&(fish.gear=='seine')]
        labels,categories=pd.factorize(part.species,sort=True)
        N,S=len(labels),len(categories)
        prefix=Counter();checkpoints={max(1,int(N*f)):f for f in protocol['fractions']}
        for t,label in enumerate(labels,1):
            prefix[int(label)]+=1
            if t not in checkpoints:continue
            f1=sum(v==1 for v in prefix.values());f2=sum(v==2 for v in prefix.values())
            rows.append(dict(year=year,gear='seine',fraction=checkpoints[t],samples=t,N=N,S=S,
                observed=len(prefix),chao1_bc=len(prefix)+f1*(f1-1)/(2*(f2+1)),
                source_row=int(part.iloc[t-1].record_id),
                order='Original source-file row order; not proven temporal order or iid',
                guarantee='Descriptive only; no 95% species-total guarantee'))
        if len(prefix)!=S:raise ArithmeticError('Incomplete file replay')
    pd.DataFrame(rows).to_csv(OUT/'fish_source_order.csv',index=False)
    dump(OUT/'reporting_wrapper_protocol.json',dict(post_hoc=True,new_random_draws=0,
         rule='max(D_union, original output)',why='Do not discard already observed species due to clipping at n.',
         valid_target='[min(effective_support_epsilon,n), S] on the original success event. No relative-error guarantee.',
         inference='Separate Bonferroni family for wrapper relative-error rates; no tuning or selection of successful populations.',
         source_order_note='File row order is not a timestamp. Fish case study literally reads the retained rows sequentially.'))


if __name__=='__main__':
    main()
