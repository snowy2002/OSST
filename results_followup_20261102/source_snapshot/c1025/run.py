"""Reproducible C0=1025 experiment. Run: python -m c1025.run --phase all.

Statistical experiments retain L0=1024. Astronomical-branch checks are
deterministic interval calculations, never called physical sampling.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,platform,shutil,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import poisson
from flint import arb,ctx
from benchmark_core import Population,stable_rng
from support_size import Fingerprint,poisson_fingerprint
from paper_reference import reference_parameters,reference_test
from c1025.data import ROOT,DATA,prepare,load,save,sha
from c1025.core import budget,sample_path,truth,effective_support,scenarios,cp_upper,exact_occupancy_cdf,count_decisions

OUT=ROOT/'results_c1025'
DESIGN={
 'version':1,'C0':1025,'L0':1024,'assumption':'Accept the supplied C0=1025 theorem as the working hypothesis; no acceptance guarantee for any venue',
 'sizes':[1,2,10,100,1000,10000,1000000,1000000000,1000000000000],
 'epsilons':[.001,.01,.05,.1,.25],
 'methods':['C1025','C67108864','C1024','count_chernoff_99'],
 'seed_blocks':4,'trials_per_block':2500,
 'curve_sizes':[10,1000,1000000,1000000000000],
 'curve_epsilons':[.001,.01,.1,.25],
 'curve_constants':[.25,.5,1,2,4,8,16,32,64,128,256,512,1024,1025],
 'curve_scenarios':['uniform_null','uniform_boundary','diffuse_boundary','multiscale_boundary','gap_near'],
 'curve_trials_per_block':1000,
 'real_epsilons':[.001,.01,.05,.1,.25],
 'real_n_rules':['floor(S/2)','eff_epsilon-1','eff_epsilon','S','ceil(1.5S)'],
 'dependence':['bci','mite','fish_2007_seine','fish_2008_seine'],
 'dependence_epsilons':[.01,.1],'dependence_trials':4000,'duplicate_factors':[1,10,100,1000],
 'group_holdout_splits':5,'group_holdout_test_fraction':.3,
 'seeds':{'synthetic':10252101,'natural':10252102,'curves':10252103,'dependence':10252104,'stream':10252105,'runtime':10252106},
 'stream_trials':300,'stream_sizes':[10,100,1000],'stream_epsilons':[.01,.1],
 'statistics':'Exact one-sided Clopper-Pearson 95%; Bonferroni simultaneous error bounds separately within each suite over every non-Gap method cell; separate lower-bound family over all cells. Four independent seed blocks. No stopping based on observed errors.',
 'sampling':'Joint Poisson-process occupancy, binomial thinning within equal-probability groups; distributionally exact in real arithmetic, NumPy binary64 RNG. Real resampling is iid from finite empirical populations, not iid validation of nature. M is not reconstructed from occupancy.',
 'controls':'Original coefficient and C1024, certified Chernoff count baseline at same 1% target; fixed coefficient curves; full-label public API runs; independent finite occupancy PGF and mean checks; duplicated-observation stress; group holdout; immediate count stopping.',
 'disclosure':'Local frozen follow-up informed by earlier experiments and proof; not externally preregistered. No parameters selected using this follow-up outcomes.',
}

def csv_save(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def freeze(out):
    out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    data=prepare()
    files=[ROOT/'paper_reference.py',ROOT/'support_size.py',ROOT/'benchmark_core.py',ROOT/'publication/core.py',
           ROOT/'test_c1025.py',*[ROOT/'c1025'/name for name in ['__init__.py','core.py','data.py','run.py']]]
    metadata=dict(design=DESIGN,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                  data_hashes={p.name:sha(p) for p in sorted((DATA/'prepared').glob('*.npz'))})
    target=out/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text())
        for k,v in metadata.items():
            if old[k]!=v:raise RuntimeError(f'Frozen {k} differs. Use new output and disclose changes.')
        return
    metadata.update(frozen_at=datetime.now(timezone.utc).isoformat(),environment={'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count()})
    save(target,metadata);save(out/'datasets.json',data)
    shutil.copy2(DATA/'sources.json',out/'data_sources.json')
    for p in files:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    deps=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'requirements-executed.txt').write_text(deps.stdout)


def evaluate_cell(pop,n,eps,methods,scenario,suite,out,block_trials,**meta):
    label,tail,eff=truth(pop,n,eps);means=[budget(n,str(eps),m) for m in methods]
    plan=reference_parameters(n,str(eps),c0=1025)
    raw=[];blocks=[]
    key=hashlib.sha256(repr((suite,scenario,n,eps,meta)).encode()).hexdigest()[:20]
    for block in range(DESIGN['seed_blocks']):
        rng=stable_rng(DESIGN['seeds'][suite],scenario,n,eps,meta,block)
        paths=sample_path(pop,means,block_trials,rng)
        observed=np.stack([paths[m] for m in means]);raw.append(observed)
        for i,method in enumerate(methods):
            reject,lower=count_decisions(observed[i],plan)
            errors=None if label=='Gap' else int(np.sum(reject if label=='Yes' else ~reject))
            blocks.append(dict(suite=suite,cell=key,method=method,block=block,trials=block_trials,errors=errors,
                               rejects=int(reject.sum()),lower_errors=int(np.sum((lower>pop.support)|(lower<min(eff,n))))))
    observed=np.concatenate(raw,axis=1)
    np.savez_compressed(out/'raw'/f'{suite}_{key}.npz',observed=observed,means=np.asarray([str(x) for x in means]),
                        methods=np.asarray(methods),multiplicities=pop.multiplicities,probabilities=pop.probabilities)
    rows=[]
    for i,(method,m) in enumerate(zip(methods,means)):
        reject,lower=count_decisions(observed[i],plan);T=len(reject)
        errors=None if label=='Gap' else int(np.sum(reject if label=='Yes' else ~reject))
        pgf=exact_occupancy_cdf(pop,m,n) if suite!='curves' else None
        expected=float(pop.multiplicities@(-np.expm1(-float(m)*pop.probabilities)))
        rows.append(dict(suite=suite,cell=key,scenario=scenario,**meta,n=n,epsilon=eps,S=pop.support,
                         truth=label,delta_n=float(tail),delta_n_exact=str(tail),effective_support=eff,
                         method=method,m=m,branch=plan.branch,trials=T,rejects=int(reject.sum()),errors=errors,
                         error_rate=None if errors is None else errors/T,reject_rate=float(reject.mean()),
                         lower_over_errors=int(np.sum(lower>pop.support)),lower_short_errors=int(np.sum(lower<min(eff,n))),
                         lower_errors=int(np.sum((lower>pop.support)|(lower<min(eff,n)))),
                         mean_lower=float(lower.mean()),mean_observed=float(observed[i].mean()),expected_observed=expected,
                         pgf_failure_probability=None if label=='Gap' or pgf is None else (1-pgf if label=='Yes' else pgf),
                         raw_file=f'raw/{suite}_{key}.npz',raw_index=i))
    return rows,blocks


def finalize(out,suite,rows,blocks):
    K=sum(r['errors'] is not None for r in rows)
    for r in rows:
        if r['errors'] is not None:
            r['error_upper95']=cp_upper(r['errors'],r['trials'])
            r['error_upper95_simultaneous']=cp_upper(r['errors'],r['trials'],.05/max(1,K))
        r['lower_error_upper95_simultaneous']=cp_upper(r['lower_errors'],r['trials'],.05/len(rows))
    csv_save(out/f'{suite}.csv',rows);csv_save(out/f'{suite}_seed_blocks.csv',blocks)
    save(out/f'{suite}_complete.json',dict(cells=len(rows),base_cells=len({r['cell'] for r in rows}),
          method_trial_evaluations=sum(r['trials'] for r in rows),non_gap_comparisons=K))


def synthetic(out):
    rows=[];blocks=[]
    for n in DESIGN['sizes']:
        for eps in DESIGN['epsilons']:
            for name,pop in scenarios(n,eps):
                a,b=evaluate_cell(pop,n,eps,DESIGN['methods'],name,'synthetic',out,DESIGN['trials_per_block'])
                rows+=a;blocks+=b
        print('synthetic',n,len(rows),flush=True)
    finalize(out,'synthetic',rows,blocks)


def natural(out):
    rows=[];blocks=[];feasibility=[]
    for spec in json.loads((out/'datasets.json').read_text()):
        name=spec['dataset'];counts=load(name);pop=Population.from_counts(counts)
        for eps in DESIGN['real_epsilons']:
            eff=effective_support(pop,eps)
            for n in sorted(set([max(1,pop.support//2),max(1,eff-1),eff,pop.support,math.ceil(1.5*pop.support)])):
                a,b=evaluate_cell(pop,n,eps,DESIGN['methods'],name,'natural',out,DESIGN['trials_per_block'],dataset=name)
                rows+=a;blocks+=b;m=budget(n,str(eps),'C1025');N=int(counts.sum())
                feasibility.append(dict(dataset=name,n=n,epsilon=eps,N=N,S=pop.support,m=m,budget_to_records=m/N,
                  poisson_overflow_probability=float(poisson.sf(N,m)),has_mean_budget=N>=m,
                  direct_observed_reject=pop.support>n,
                  interpretation='Observed > n certifies rejection without iid; insufficient budget cannot certify acceptance; finite-record support is not external ecological truth'))
        print('natural',name,len(rows),flush=True)
    finalize(out,'natural',rows,blocks);csv_save(out/'natural_feasibility.csv',feasibility)


def curves(out):
    methods=['C'+str(c) for c in DESIGN['curve_constants']];rows=[];blocks=[]
    for n in DESIGN['curve_sizes']:
        for eps in DESIGN['curve_epsilons']:
            for name,pop in scenarios(n,eps):
                if name not in DESIGN['curve_scenarios']:continue
                a,b=evaluate_cell(pop,n,eps,methods,name,'curves',out,DESIGN['curve_trials_per_block']);rows+=a;blocks+=b
        print('curves',n,len(rows),flush=True)
    finalize(out,'curves',rows,blocks)


def dependence(out):
    """Exact marginal p but deliberately correlated duplicate observations."""
    rows=[];hold=[];T=DESIGN['dependence_trials']
    for name in DESIGN['dependence']:
        matrix=sparse.load_npz(DATA/'prepared'/f'{name}_groups.npz').toarray();counts=matrix.sum(axis=0);pop=Population.from_counts(counts)
        for eps in DESIGN['dependence_epsilons']:
            n=max(1,effective_support(pop,eps)-1);label,tail,eff=truth(pop,n,eps);m=budget(n,str(eps),'C1025')
            for B in DESIGN['duplicate_factors']:
                # Poi(m/B) iid labels, each copied B times: expected observations m,
                # unchanged marginal category distribution, violated independence.
                reduced=max(1,int(round(m/B)))
                obs=sample_path(pop,[reduced],T,stable_rng(DESIGN['seeds']['dependence'],name,eps,B))[reduced]
                err=int(np.sum(obs<=n)) if label=='No' else None
                rows.append(dict(dataset=name,n=n,epsilon=eps,truth=label,m=m,duplicate_factor=B,
                    independent_label_mean=reduced,expected_reported_records=B*reduced,trials=T,
                    errors=err,error_rate=None if err is None else err/T,
                    design='artificial duplication stress, same marginal p; not observed field dependence'))
        # Freeze n from the training groups only. Evaluate against held-out
        # finite-population truth, not an unknown ecosystem support.
        for split in range(DESIGN['group_holdout_splits']):
            rng=stable_rng(DESIGN['seeds']['dependence'],name,'split',split);order=rng.permutation(len(matrix));cut=max(1,int(.7*len(matrix)))
            train=matrix[order[:cut]].sum(axis=0);test=matrix[order[cut:]].sum(axis=0)
            train_pop=Population.from_counts(train);test_pop=Population.from_counts(test);eps=.1
            for rule,n in [('train_effective_boundary',max(1,effective_support(train_pop,eps)-1)),('train_support',train_pop.support)]:
                label,tail,eff=truth(test_pop,n,eps);m=budget(n,str(eps),'C1025')
                obs=sample_path(test_pop,[m],T,rng)[m];lower=np.clip(obs,1,n)
                err=None if label=='Gap' else int(np.sum(obs>n if label=='Yes' else obs<=n))
                hold.append(dict(dataset=name,split=split,rule=rule,n=n,epsilon=eps,S=test_pop.support,truth=label,
                    train_groups=len(order[:cut]),test_groups=len(order[cut:]),train_S=train_pop.support,
                    test_N=int(test.sum()),m=m,trials=T,errors=err,error_rate=None if err is None else err/T,
                    lower_errors=int(np.sum((lower>test_pop.support)|(lower<min(eff,n)))),
                    delta_n_exact=str(tail),scope='iid resampling from held-out finite group aggregate'))
            np.savez_compressed(out/'raw'/f'holdout_{name}_{split}.npz',train_groups=order[:cut],test_groups=order[cut:],train_counts=train,test_counts=test)
        print('dependence',name,flush=True)
    for frame in [rows,hold]:
        K=sum(r['errors'] is not None for r in frame)
        for r in frame:
            if r['errors'] is not None:r['error_upper95_simultaneous']=cp_upper(r['errors'],r['trials'],.05/max(1,K))
    csv_save(out/'dependence.csv',rows);csv_save(out/'group_holdout.csv',hold)


def reference(out):
    """Run the public full-fingerprint API on actual iid Poisson counts."""
    rows=[];samplers=[]
    for n in [1,10,100,1000]:
        for eps in [.001,.01,.1,.25]:
            for name,pop in scenarios(n,eps):
                if pop.support>100000:continue
                probabilities=np.repeat(pop.probabilities,pop.multiplicities)
                for c in [1025,2**26]:
                    plan=reference_parameters(n,str(eps),c0=c)
                    for rep in range(3):
                        fp=poisson_fingerprint(probabilities,plan.m,stable_rng(10252901,n,eps,name,c,rep))
                        start=time.perf_counter();actual=reference_test(fp,n,str(eps),m=plan.m,c0=c);elapsed=time.perf_counter()-start
                        assert actual['decision']==('No' if fp.observed>n else 'Yes')
                        assert actual['support_lower_bound']==min(n,max(1,fp.observed))
                        rows.append(dict(actual,scenario=name,epsilon_numeric=eps,repeat=rep,elapsed=elapsed))
    save(out/'reference_api_runs.json',rows)
    # Numerically independent finite-PGF check at challenging, unsaturated budgets.
    for n in [2,10,50]:
        for eps in [.01,.1]:
            for name,pop in scenarios(n,eps):
                if name not in ['uniform_boundary','diffuse_boundary','uniform_null']:continue
                m=budget(n,str(eps),'C1');obs=sample_path(pop,[m],20000,stable_rng(10252902,n,eps,name))[m]
                exact=exact_occupancy_cdf(pop,m,n)
                samplers.append(dict(n=n,epsilon=eps,scenario=name,m=m,trials=len(obs),
                   empirical_accept=float(np.mean(obs<=n)),pgf_accept=exact,
                   expected_observed=float(pop.multiplicities@(-np.expm1(-m*pop.probabilities))),
                   mean_observed=float(obs.mean()),mean_standard_error=float(obs.std(ddof=1)/math.sqrt(len(obs)))))
    csv_save(out/'sampler_validation.csv',samplers)
    print('public API runs',len(rows),flush=True)


def audit(out):
    # Rerun independent high-degree audit, preserving all historical artifacts.
    path=ROOT/'artifacts/audit_c0_1025.py';spec=importlib.util.spec_from_file_location('c1025_independent_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=out/'algebra_audit';module.main()


def stream(out):
    """Actual labels: demonstrate fixed acquisition versus immediate D>n stop."""
    rows=[];T=DESIGN['stream_trials']
    for n in DESIGN['stream_sizes']:
        for eps in DESIGN['stream_epsilons']:
            for name,pop in scenarios(n,eps):
                if name not in ['uniform_null','uniform_boundary','diffuse_boundary']:continue
                p=np.repeat(pop.probabilities,pop.multiplicities);p/=p.sum()
                m=budget(n,str(eps),'C1025');count_m=budget(n,str(eps),'count_chernoff_99')
                # Immediate count cap uses a Poisson draw; early rejection is exact.
                rng=stable_rng(DESIGN['seeds']['stream'],n,eps,name);cost=[];rejects=[]
                for t in range(T):
                    M=int(rng.poisson(count_m));seen=set();used=0;rejected=False
                    while used<M and not rejected:
                        labels=rng.choice(len(p),size=min(4096,M-used),p=p)
                        for label in labels:
                            seen.add(int(label));used+=1
                            if len(seen)>n:rejected=True;break
                    cost.append(used);rejects.append(rejected)
                label,_,_=truth(pop,n,eps)
                rows.append(dict(n=n,epsilon=eps,scenario=name,truth=label,trials=T,
                    C1025_fixed_expected_samples=m,count_fixed_expected_samples=count_m,
                    count_immediate_mean_samples=float(np.mean(cost)),count_immediate_median=float(np.median(cost)),
                    count_immediate_se=float(np.std(cost,ddof=1)/math.sqrt(T)),
                    errors=int(np.sum(rejects)) if label=='Yes' else T-int(np.sum(rejects))))
                np.savez_compressed(out/'raw'/f'stream_{n}_{eps}_{name}.npz',cost=cost,rejects=rejects)
        print('stream',n,flush=True)
    csv_save(out/'stream.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    phases={'synthetic':synthetic,'natural':natural,'curves':curves,'dependence':dependence,'reference':reference,'audit':audit,'stream':stream}
    parser.add_argument('--phase',choices=['all','prepare',*phases],default='all');parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();out=args.out.resolve()
    if args.phase in ['prepare','all']:freeze(out)
    elif not (out/'protocol.json').exists():raise RuntimeError('Run prepare first')
    if args.phase=='prepare':return
    for name in (list(phases) if args.phase=='all' else [args.phase]):
        marker=out/f'execution_{name}.json'
        if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':
            print(name,'already complete',flush=True);continue
        record=dict(status='running',started=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'protocol.json'))
        save(marker,record);start=time.perf_counter()
        try:phases[name](out)
        except Exception as error:
            record.update(status='failed',error=repr(error),elapsed_seconds=time.perf_counter()-start);save(marker,record);raise
        record.update(status='complete',elapsed_seconds=time.perf_counter()-start,finished=datetime.now(timezone.utc).isoformat());save(marker,record)


if __name__=='__main__':main()
