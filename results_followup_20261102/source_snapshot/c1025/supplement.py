"""Supplemental public-API, actual-group and runtime checks; separately frozen."""
from __future__ import annotations
import argparse,json,resource,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy import sparse
from benchmark_core import Population,stable_rng
from paper_reference import reference_parameters,reference_test
from support_size import poisson_fingerprint
from c1025.core import budget,effective_support,truth,cp_upper
from c1025.data import ROOT,DATA,load,save,sha
from c1025.run import csv_save


def runtime_child(n,c):
    p=np.full(n,1/n);rng=stable_rng(10252106,n,c);times=[]
    for rep in range(11):
        start=time.perf_counter();plan=reference_parameters(n,'0.1',c0=c);t1=time.perf_counter()
        fp=poisson_fingerprint(p,plan.m,rng);t2=time.perf_counter()
        answer=reference_test(fp,n,'0.1',m=plan.m,c0=c);t3=time.perf_counter()
        assert answer['decision']=='Yes'
        if rep:times.append([t1-start,t2-t1,t3-t2])
    a=np.asarray(times)
    return dict(n=n,C0=c,m=plan.m,repeats=len(a),parameter_median_seconds=float(np.median(a[:,0])),
                count_generation_median_seconds=float(np.median(a[:,1])),reference_decision_median_seconds=float(np.median(a[:,2])),
                full_count_pipeline_median_seconds=float(np.median(a.sum(axis=1))),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                observed=fp.observed,fingerprint_bins=len(fp.frequencies),
                scope='Actual independent Poisson category counts; simulated count generation is not physical label acquisition time')


def run(out):
    protocol=dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
      disclosure='Supplemental implementation/runtime checks added after primary summary inspection; no primary statistical design or parameters changed.',
      public_api={'datasets':'all ten','epsilons':[.01,.1,.25],'n':['effective_support-1','S'],'repetitions':5},
      group_blocks=[1,20,100,1000,10000],group_trials=1000,
      runtime={'n':[100,10000,1000000],'C0':[1025,2**26],'repeats':10,'warmup':1,'isolated_processes':True})
    save(out/'supplement_protocol.json',protocol)
    api=[];field=[];group=[]
    for spec in json.loads((out/'datasets.json').read_text()):
        name=spec['dataset'];counts=load(name);p=counts/counts.sum();pop=Population.from_counts(counts)
        for eps in protocol['public_api']['epsilons']:
            for n in sorted(set([max(1,effective_support(pop,eps)-1),pop.support])):
                plan=reference_parameters(n,str(eps),c0=1025)
                for rep in range(5):
                    fp=poisson_fingerprint(p,plan.m,stable_rng(10252903,name,n,eps,rep))
                    answer=reference_test(fp,n,str(eps),m=plan.m,c0=1025)
                    assert answer['decision']==('No' if fp.observed>n else 'Yes')
                    api.append(dict(answer,dataset=name,repeat=rep,scope='iid Poisson counts from the finite empirical distribution'))
        # Execute the finite-file Poisson adapter only where the frozen main
        # design actually has enough records. Every overflow is retained.
        import csv
        for r in csv.DictReader((out/'natural_feasibility.csv').open()):
            if r['dataset']!=name or r['has_mean_budget']!='True':continue
            n=int(r['n']);eps=r['epsilon'];m=int(r['m']);labels=np.repeat(np.arange(len(counts)),counts)
            rng=stable_rng(10252904,name,n,eps)
            for rep in range(200):
                M=int(rng.poisson(m))
                if M>len(labels):
                    field.append(dict(dataset=name,n=n,epsilon=eps,m=m,M=M,repeat=rep,status='overflow',decision='Inconclusive'))
                    continue
                chosen=rng.choice(len(labels),M,replace=False)
                from support_size import Fingerprint
                fp=Fingerprint.from_counts(np.bincount(labels[chosen],minlength=len(counts)))
                answer=reference_test(fp,n,eps,m=m,c0=1025)
                field.append(dict(answer,dataset=name,repeat=rep,status='executed',
                    scope='Finite-file subsampling without replacement; real core samples are not verified iid; output is descriptive'))
        matrix_file=DATA/'prepared'/f'{name}_groups.npz'
        if not matrix_file.exists():continue
        matrix=sparse.load_npz(matrix_file).toarray();totals=matrix.sum(axis=1);weights=totals/totals.sum()
        for eps in [.01,.1]:
            n=max(1,effective_support(pop,eps)-1);label,_,_=truth(pop,n,eps);m=budget(n,str(eps),'C1025')
            for B in protocol['group_blocks']:
                rng=stable_rng(10252905,name,eps,B);T=protocol['group_trials'];seen=np.zeros((T,len(counts)),dtype=bool);M=np.zeros(T,dtype=np.int64)
                for row,total,w in zip(matrix,totals,weights):
                    blocks=rng.poisson(float(m/B)*w,size=T)
                    samples=blocks*B;M+=samples
                    draw=rng.multinomial(samples,row/total)
                    seen|=draw>0
                D=seen.sum(axis=1);errors=None if label=='Gap' else int(np.sum(D>n if label=='Yes' else D<=n))
                group.append(dict(dataset=name,n=n,epsilon=eps,m=m,truth=label,block_size=B,trials=T,errors=errors,
                  error_rate=None if errors is None else errors/T,mean_reported_samples=float(M.mean()),mean_observed=float(D.mean()),
                  design='Poi(m/B) groups selected with probability proportional to group abundance; B iid local labels per selected group; global label marginal p unchanged; dependence is artificial'))
        print('supplement',name,flush=True)
    K=sum(r['errors'] is not None for r in group)
    for r in group:
        if r['errors'] is not None:r['error_upper95_simultaneous']=cp_upper(r['errors'],r['trials'],.05/K)
    save(out/'natural_public_api.json',api);save(out/'finite_file_runs.json',field);csv_save(out/'natural_group_stress.csv',group)
    timing=[]
    for n in protocol['runtime']['n']:
        for c in protocol['runtime']['C0']:
            result=subprocess.run([sys.executable,'-m','c1025.supplement','--child',str(n),str(c)],cwd=ROOT,capture_output=True,text=True,check=True)
            timing.append(json.loads(result.stdout));print('runtime',n,c,flush=True)
    csv_save(out/'runtime.csv',timing)
    save(out/'supplement_complete.json',dict(public_api_runs=len(api),finite_file_runs=len(field),group_cells=len(group),runtime_cells=len(timing)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',nargs=2,type=int);parser.add_argument('--out',type=Path,default=ROOT/'results_c1025')
    args=parser.parse_args()
    if args.child:print(json.dumps(runtime_child(*args.child)))
    else:run(args.out)
