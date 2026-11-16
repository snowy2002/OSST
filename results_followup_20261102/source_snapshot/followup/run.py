"""Run frozen fresh-seed experiments, resume by hashed cells, keep legacy data intact."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import tracemalloc

import numpy as np
from flint import arb, ctx

from benchmark_core import Profiles, stable_rng
from c7016.populations import Population, sample_profiles, synthetic_cases
from guaranteed import certified_count_budget
from paper_reference import NumericalCertificationError
from parameter_search import algorithm as alg
from support_size import Fingerprint
from followup import core
from followup.protocol import ROOT, OUT, design, freeze, save, sha, csv_save


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]


def saved_cell(out,key):
    path=out/'cells'/f'{key}.json'
    if not path.exists():
        return None
    cell=json.loads(path.read_text())
    if sha(out/cell['raw_file'])!=cell['raw_sha256']:
        raise ArithmeticError('Saved raw data changed')
    return cell


def write_cell(out,key,raw,rows,**metadata):
    file=f'raw/{key}.npz'
    np.savez_compressed(out/file,**raw)
    cell=dict(raw_file=file,raw_sha256=sha(out/file),rows=rows,**metadata)
    save(out/'cells'/f'{key}.json',cell)
    return cell


def sample(pop,plan,T,d,key):
    profiles=[];defect=0.
    for block in range(d['blocks']):
        fp,err=sample_profiles(pop,float(plan.m),T//d['blocks'],plan.degree,
                              stable_rng(d['seed'],*key,block))
        profiles.append(fp);defect=max(defect,err)
    return Profiles(np.concatenate([f.observed for f in profiles]),
                    np.concatenate([f.low for f in profiles])),defect


def row_for(pop,plan,name,method,values,refinements,**metadata):
    truth,tail,eff=pop.truth(plan.n,str(plan.epsilon));T=len(values['reject'])
    return dict(scenario=name,method=method,C=plan.C,L0=plan.L0,n=plan.n,epsilon=str(plan.epsilon),
        mean=float(plan.m),degree=plan.degree,truth=truth,tail_exact=str(tail),S=pop.support,
        effective=eff,trials=T,errors=int(values['binary'].sum()) if truth!='Gap' else None,
        coverage_errors=int((values['over']|values['short']).sum()),joint_errors=int(values['joint'].sum()),
        mean_lower_lo=float(values['lower_lo'].mean()),mean_lower_hi=float(values['lower_hi'].mean()),
        nonunique=int(np.sum(values['lower_lo']!=values['lower_hi'])),refinements=refinements,**metadata)


def selected_cases(n,eps,names):
    cases=dict(synthetic_cases(n,eps))
    return [(name,cases[name]) for name in names]


def mechanism(out,d):
    spec=d['mechanism'];rows=[]
    for n in spec['ns']:
        for eps in spec['epsilons']:
            p14=alg.parameters('70',14,n,eps);p16=alg.parameters('70',16,n,eps)
            for name,pop in selected_cases(n,eps,spec['scenarios']):
                key=identity(['mechanism',n,eps,name,pop.as_dict()]);old=saved_cell(out,key)
                if old:
                    rows+=old['rows'];continue
                fp,defect=sample(pop,p14,spec['trials'],d,('mechanism',key))
                a,b=core.scale_intervals(fp,p14)
                # Explicitly retain all scales, including ones which never win.
                raw=dict(D=fp.observed,low=fp.low,scale_lo=a,scale_hi=b)
                configs=[('full14',p14,'full'),('full16',p16,'full'),
                         ('single14',p14,'single'),('count70',p14,'count')]
                if p14.scales: configs.append(('no_count14',p14,'no_count'))
                rr=[]
                for method,plan,mode in configs:
                    aa,bb=(a,b) if plan.scales else (a[:,:1],b[:,:1])
                    values,ref=core.outcomes(fp,plan,pop,aa,bb,mode)
                    raw.update({method+'__'+k:v for k,v in values.items()})
                    rr.append(row_for(pop,plan,name,method,values,ref,suite='mechanism',cell=key,
                                     pmf_defect=defect,mean_D=float(fp.observed.mean()),
                                     extra_rejects=int(np.sum(values['reject']&(fp.observed<=n))),
                                     improvements=int(np.sum(values['lower_lo']>np.clip(fp.observed,1,n)))))
                # -1 denotes overlapping intervals or tied polynomial maxima; 0 denotes D.
                winner=np.full(len(a),-1,np.int16)
                if a.shape[1]==1:
                    winner[:]=0
                else:
                    winner[fp.observed>=b[:,1:].max(axis=1)]=0
                    for j in range(1,a.shape[1]):
                        other=np.max(np.delete(b,j,axis=1),axis=1)
                        winner[a[:,j]>other]=j
                raw['winner_certified']=winner
                # Independently score representative rows at higher precision.
                for i in [0,len(a)//2,len(a)-1]:
                    v=core.arb_value(fp.observed[i],fp.low[i],p14,bits=1024)
                    with ctx.workprec(1024):
                        if not(v>=arb(float(a[i].max())) and v<=arb(float(b[i].max()))):
                            raise ArithmeticError('Mechanism enclosure cross-check failed')
                write_cell(out,key,raw,rr,suite='mechanism',population=pop.as_dict())
                rows+=rr
            print('mechanism complete',n,eps,flush=True)
    csv_save(out/'mechanism.csv',rows)


def budget_populations(n,eps,spec):
    result=selected_cases(n,eps,spec['scenarios'])
    # This population grid is independent of the method being evaluated.
    reference=alg.parameters(str(spec['fixed_stress_reference_C']),14,n,eps)
    for lam in spec['fixed_stress_lambdas']:
        p=Fraction(lam)/Fraction(str(float(reference.m)))
        mass=(n-1)*p
        if mass<Fraction(19,20):
            result.append(('fixed_C60_lambda_'+lam,Population((1,n-1),(1-mass,p))))
    return result


def budgets(out,d):
    spec=d['budget'];rows=[];plans_table=[]
    for n in spec['ns']:
        for eps in spec['epsilons']:
            plans={f'C{C}':alg.parameters(C,14,n,eps) for C in spec['constants']}
            count_plans={}
            for risk in spec['counting_risks']:
                mean,tail=certified_count_budget(n,Fraction(eps),Fraction(risk))
                count_plans['count_risk_'+risk]=(mean,tail)
            for method,plan in plans.items():
                plans_table.append(dict(n=n,epsilon=eps,method=method,mean=float(plan.m),
                    guarantee='.0009' if method=='C70' else 'exploratory; no uniform risk guarantee',
                    target_met_by_theorem=method=='C70',mean_kind='real Poisson mean',
                    ratio_to_count_01=float(plan.m)/count_plans['count_risk_.01'][0]))
            for method,(m,tail) in count_plans.items():
                plans_table.append(dict(n=n,epsilon=eps,method=method,mean=m,
                    guarantee=method.removeprefix('count_risk_'),target_met_by_theorem=True,
                    mean_kind='integer Poisson mean',tail_interval=tail,
                    ratio_to_count_01=m/count_plans['count_risk_.01'][0]))
            # Save exact populations before drawing any trials at this problem size.
            populations=budget_populations(n,eps,spec)
            manifest=out/f'budget_populations_{n}_{eps}.json'
            payload={name:pop.as_dict() for name,pop in populations}
            if manifest.exists() and json.loads(manifest.read_text())!=payload:
                raise RuntimeError('Common population grid changed')
            save(manifest,payload)
            for name,pop in populations:
                for method in [*plans,*count_plans]:
                    key=identity(['budget',n,eps,name,method,pop.as_dict()]);old=saved_cell(out,key)
                    if old:
                        rows+=old['rows'];continue
                    if method in plans:
                        plan=plans[method]
                    else:
                        # Count scoring ignores C; mean is the certified independent budget.
                        plan=alg.Plan(method,1024,n,Fraction(eps),arb(1),arb(count_plans[method][0]),(),512)
                    fp,defect=sample(pop,plan,spec['trials'],d,('budget',key))
                    a,b=core.scale_intervals(fp,plan)
                    raw=dict(D=fp.observed,low=fp.low)
                    rr=[]
                    modes=[(method,'full')]+([(method+'_count','count')] if method in plans else [])
                    for label,mode in modes:
                        values,ref=core.outcomes(fp,plan,pop,a,b,mode)
                        raw.update({label+'__'+k:v for k,v in values.items()})
                        rr.append(row_for(pop,plan,name,label,values,ref,suite='budget',cell=key,
                                  pmf_defect=defect,extra_rejects=int(np.sum(values['reject']&(fp.observed<=n)))))
                    write_cell(out,key,raw,rr,suite='budget',population=pop.as_dict())
                    rows+=rr
            print('budget complete',n,eps,flush=True)
    csv_save(out/'budget.csv',rows);csv_save(out/'budget_plans.csv',plans_table)


def refine_archive(out,d):
    spec=d['numeric'];source=ROOT/'results_parameter_search';rows=[];total=0
    for path in sorted((source/'cells').glob('*.json')):
        original=json.loads(path.read_text())
        rr=[r for r in original['rows'] if r['C']=='70' and r['L0']==14 and r['nonunique_bounds']]
        if not rr:continue
        r=rr[0];key='refine_'+path.stem
        if 'archive_limit' in spec and total>=spec['archive_limit']:break
        old=saved_cell(out,key)
        if old:
            rows+=old['rows'];total+=old['rows'][0]['refined'];continue
        rawpath=source/original['raw_file']
        if sha(rawpath)!=original['raw_sha256']:raise ArithmeticError('Legacy raw hash mismatch')
        with np.load(rawpath) as raw:
            ids=np.flatnonzero(raw['70_14_lower_lo']!=raw['70_14_lower_hi'])
            if 'archive_limit' in spec:ids=ids[:max(0,spec['archive_limit']-total)]
            plan=alg.parameters('70',14,r['n'],r['epsilon'])
            exact=[];bits=[];elapsed=[]
            for i in ids:
                start=time.perf_counter()
                result=core.exact_output(raw['D_70'][i],raw['low_70'][i],plan)
                elapsed.append(time.perf_counter()-start);exact.append(result['lower']);bits.append(result['bits'])
                if not raw['70_14_lower_lo'][i]<=result['lower']<=raw['70_14_lower_hi'][i]:
                    raise ArithmeticError('Refined integer outside saved enclosure')
                if (result['decision']=='No')!=bool(raw['70_14_reject'][i]):
                    raise ArithmeticError('Refined binary decision changed')
                if (result['lower']>r['S'])!=bool(raw['70_14_over'][i]):
                    raise ArithmeticError('Refined upper coverage changed')
                if (result['lower']<min(r['n'],r['effective']))!=bool(raw['70_14_short'][i]):
                    raise ArithmeticError('Refined lower coverage changed')
        row=dict(suite='refinement',cell=key,n=r['n'],epsilon=r['epsilon'],scenario=r['scenario'],
                 refined=len(ids),seconds=sum(elapsed),max_bits=max(bits),
                 changed_decisions=0,changed_coverage=0,legacy_raw=original['raw_file'],
                 legacy_raw_sha256=original['raw_sha256'])
        write_cell(out,key,dict(indices=ids,exact_lower=np.array(exact,np.int64),
                               bits=np.array(bits,np.int32),seconds=np.array(elapsed)),[row],suite='refinement')
        rows.append(row);total+=len(ids)
        if len(rows)%10==0:print('legacy integers refined',total,flush=True)
    csv_save(out/'integer_refinement.csv',rows)
    print('legacy refinement complete',total,flush=True)


def formula_checks(out,d):
    results=[]
    for n in d['numeric']['independent_formula_ns']:
        with ctx.workprec(2048):
            plan=alg.parameters('70',14,n,'.1',bits=2048)
            for s in plan.scales:
                direct=core.closed_form_weights(s.A,s.B,s.k)
                for ell,w in enumerate(direct[1:]):
                    recurrence=s.corrections[ell]+1
                    if not w.overlaps(recurrence):raise ArithmeticError('Independent coefficient mismatch')
                    # Ensure comparison is informative, not merely overlapping wide balls.
                    if (w-recurrence).abs_upper()>arb('1e-100')*(1+w.abs_upper()):
                        raise ArithmeticError('Independent coefficient accuracy insufficient')
                results.append(dict(n=n,j=s.j,degree=s.k,coefficients=s.k,passed=True))
    save(out/'independent_coefficients.json',dict(passed=True,rows=results,
                    checks=sum(r['coefficients'] for r in results),precision_bits=2048))


def integration(out,d):
    spec=d['numeric'];rows=[]
    for n in spec['explicit_count_ns']:
        plan=alg.parameters('70',14,n,spec['epsilon'])
        for name,pop in selected_cases(n,spec['epsilon'],spec['scenarios']):
            for rep in range(spec['explicit_replicates']):
                key=identity(['explicit',n,name,rep]);file=out/'raw'/f'{key}_fingerprint.json'
                rng=stable_rng(d['seed'],'explicit',n,name,rep);frequencies=Counter();M=0
                start=time.perf_counter()
                for g,p in zip(pop.multiplicities,pop.probabilities):
                    counts=rng.poisson(float(plan.m)*float(p),size=g)
                    M+=int(counts.sum())
                    values,multiplicity=np.unique(counts[counts>0],return_counts=True)
                    frequencies.update({int(v):int(k) for v,k in zip(values,multiplicity)})
                sampling=time.perf_counter()-start
                fp=Fingerprint(frequencies)
                if M!=fp.total:raise ArithmeticError('Realized total mismatch')
                save(file,{str(k):v for k,v in fp.frequencies.items()})
                start=time.perf_counter();result=core.test_fingerprint(fp,n,spec['epsilon'],mean=plan.m)
                seconds=time.perf_counter()-start
                low=np.array([[fp.frequencies.get(k,0) for k in range(1,plan.degree+1)]],np.int64)
                profiles=Profiles(np.array([fp.observed],np.int64),low)
                a,b=core.scale_intervals(profiles,plan)
                events,_=core.outcomes(profiles,plan,pop,a,b)
                if bool(events['reject'][0])!=(result['decision']=='No'):raise ArithmeticError('API mismatch')
                if not events['lower_lo'][0]<=result['lower']<=events['lower_hi'][0]:raise ArithmeticError('API lower mismatch')
                rows.append(dict(kind='explicit_category_poisson',scenario=name,rep=rep,S=pop.support,
                            sampling_seconds=sampling,inference_seconds=seconds,
                            fingerprint_file=str(file.relative_to(out)),fingerprint_sha256=sha(file),**result))
    # Actual labels in a polynomial-active regime; no grouped shortcuts here.
    n=spec['label_n'];eps=spec['epsilon'];plan=alg.parameters('70',14,n,eps)
    for rep in range(spec['label_replicates']):
        rng=stable_rng(d['seed'],'labels',n,rep);M=int(rng.poisson(float(plan.m)))
        # Uniform support n is a deliberately fixed null, not outcome-selected.
        start=time.perf_counter();labels=rng.integers(n,size=M,dtype=np.int64)
        generation=time.perf_counter()-start
        start=time.perf_counter();fp=Fingerprint.from_labels(map(int,labels))
        counting=time.perf_counter()-start
        counts=np.bincount(labels,minlength=n)
        if fp.frequencies!=Fingerprint.from_counts(counts).frequencies:
            raise ArithmeticError('Label/count/fingerprint pipeline mismatch')
        start=time.perf_counter();result=core.test_fingerprint(fp,n,eps,mean=plan.m)
        scoring=time.perf_counter()-start
        file=out/'raw'/f'label_integration_{rep}.npz';np.savez_compressed(file,counts=counts)
        rows.append(dict(kind='actual_iid_labels',scenario='uniform_null',rep=rep,S=n,
                         generation_seconds=generation,counting_seconds=counting,inference_seconds=scoring,
                         label_array_bytes=labels.nbytes,counts_file=str(file.relative_to(out)),
                         counts_sha256=sha(file),**result))
        del labels,counts
    save(out/'integration.json',dict(runs=rows,passed=True,
         scope='Numerical integration checks; small replicate counts are not statistical risk estimates.'))


def runtime(out,d):
    spec=d['numeric'];rows=[];cold=[]
    for n in spec['runtime_ns']:
        plans={}
        for L0 in [14,16]:
            alg.geometry.cache_clear();alg.compiled.cache_clear()
            start=time.perf_counter();p=alg.parameters('70',L0,n,'.1')
            alg.compiled('70',L0,n,p.bits)
            cold.append(dict(n=n,L0=L0,seconds=time.perf_counter()-start,degree=p.degree))
            plans[L0]=p
        pop=dict(synthetic_cases(n,'.1'))['epsilon_scaled_rare_1/10_null']
        fp,_=sample_profiles(pop,float(plans[14].m),spec['runtime_batch'],plans[14].degree,
                            stable_rng(d['seed'],'runtime',n))
        funcs={'count':lambda:np.clip(fp.observed,1,n),
               'full14':lambda:alg.score_intervals(fp,plans[14]),
               'full16':lambda:alg.score_intervals(fp,plans[16]),
               'all_scale_diagnostics':lambda:core.scale_intervals(fp,plans[14])}
        for f in funcs.values():f()
        rng=stable_rng(d['seed'],'timing_order',n)
        for rep in range(spec['runtime_repeats']):
            for rank,method in enumerate(rng.permutation(list(funcs))):
                start=time.perf_counter();funcs[method]();elapsed=time.perf_counter()-start
                rows.append(dict(n=n,method=method,rep=rep,order=rank,seconds=elapsed,trials=len(fp.observed)))
        for method,f in funcs.items():
            # Tracing outside timing repeats: does not inflate reported timing.
            tracemalloc.start();f();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append(dict(n=n,method=method,rep=-1,order=-1,seconds=None,trials=len(fp.observed),
                             traced_peak_bytes=peak,profile_array_bytes=fp.observed.nbytes+fp.low.nbytes))
    csv_save(out/'runtime.csv',rows);csv_save(out/'cold_compile.csv',cold)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--phase',choices=['all','mechanism','budget','numeric'],default='all')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv);d=design(args.smoke);out=args.out.resolve()
    freeze(out,d)
    if args.phase in ('all','mechanism'):mechanism(out,d)
    if args.phase in ('all','budget'):budgets(out,d)
    if args.phase in ('all','numeric'):
        formula_checks(out,d);refine_archive(out,d);integration(out,d);runtime(out,d)
    print('PHASE COMPLETE',args.phase,out,flush=True)


if __name__=='__main__':main()
