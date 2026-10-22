"""Frozen fresh-seed natural, synthetic, lifted, and adversarial comparisons."""
from fractions import Fraction as Q
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
from scipy.stats import beta
from flint import arb,ctx

from benchmark_core import stable_rng
from c1025.data import load
from c7016.populations import Population,sample_profiles,synthetic_cases
from parameter_search.algorithm import parameters,score_intervals,compare,arb_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_parameter_search'
METHODS=[('star',14),('70',14),('70',15),('70',16),('70',18),('1025',1024)]
DESIGN=dict(seed=202610224142,blocks=4,trials_per_block=500,
            ns=[1,2,10,1000,442413,442414,1202604,1202605,3269017,3269018,
                24154952,24154953,10**9,10**12],epsilons=['.001','.1'],
            natural_epsilons=['.01','.1'],lifted_targets=[10**6,10**9],
            adversarial_C=['6','10','20','40','60','69','70'],
            adversarial_n=[10000,1000000,10**12],adversarial_eps=['.001','.1'],
            adversarial_lambda=['.25','.5','1','2','4','8','16','32','64','128'],
            methods=METHODS,
            sampling='Independent Poisson category counts via grouped fingerprints; binary64 RNG.',
            pairing='C=70 branches share each fingerprint. Different C uses independent streams.',
            natural_scope='Iid draws from empirical distributions; lifted labels are constructed data.',
            timing_scope='Grouped simulation and scoring only; not physical data collection.',
            inferential_scope='Per-cell exact binomial intervals; not a distribution-free empirical proof.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def freeze():
    for folder in ['raw','cells','source_snapshot']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    sources=['parameter_search/experiments.py','parameter_search/algorithm.py',
             'c7016/populations.py','c7016/algorithm.py','paper_reference.py','guaranteed.py',
             'benchmark_core.py','c1025/core.py','c1025/data.py']
    protocol=dict(design=DESIGN,source_sha256={s:sha(ROOT/s) for s in sources},
                  data_sha256={p.name:sha(p) for p in sorted((ROOT/'data/c1025/prepared').glob('*.npz'))})
    path=OUT/'experiment_protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
            raise RuntimeError('Frozen experiment protocol changed')
    else:
        dump(path,protocol)
        for s in sources:
            target=OUT/'source_snapshot'/s;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/s).read_bytes())


def evaluate(suite,name,pop,n,eps,methods=METHODS):
    key=hashlib.sha256(repr((suite,name,n,eps,methods,pop.as_dict())).encode()).hexdigest()[:24]
    cell_path=OUT/'cells'/f'{key}.json'
    if cell_path.exists():
        cell=json.loads(cell_path.read_text())
        if sha(OUT/cell['raw_file'])!=cell['raw_sha256']:
            raise RuntimeError('Raw data changed')
        return cell['rows']
    label,tail,effective=pop.truth(n,eps)
    plans={f'{C}_{L0}':parameters(C,L0,n,eps) for C,L0 in methods}
    by_constant={}
    for method,plan in plans.items():
        by_constant.setdefault(plan.C,[]).append(method)
    buffers={m:[] for m in plans};raw={};timings={m:[0.,0.] for m in plans}
    audits={m:0 for m in plans};defects={C:0. for C in by_constant}
    for C,mlist in by_constant.items():
        sampling_plan=max((plans[m] for m in mlist),key=lambda p:p.degree)
        allD=[];alllow=[]
        for block in range(DESIGN['blocks']):
            start=time.perf_counter()
            fp,defect=sample_profiles(pop,float(sampling_plan.m),DESIGN['trials_per_block'],
                                     sampling_plan.degree,stable_rng(DESIGN['seed'],key,C,block))
            sample_time=time.perf_counter()-start
            defects[C]=max(defects[C],defect);allD.append(fp.observed);alllow.append(fp.low)
            for method in mlist:
                plan=plans[method];start=time.perf_counter()
                lo,hi=score_intervals(fp,plan)
                reject,ref=compare(fp,plan,lo,hi,n)
                over=np.zeros(len(lo),dtype=bool)
                if pop.support<n:
                    over,r=compare(fp,plan,lo,hi,pop.support);ref+=r
                t=min(n,effective)
                short=np.zeros(len(lo),dtype=bool)
                if t>1:
                    good,r=compare(fp,plan,lo,hi,t-1);short=~good;ref+=r
                lowbound=np.clip(np.ceil(lo),1,n);highbound=np.clip(np.ceil(hi),1,n)
                timings[method][0]+=sample_time;timings[method][1]+=time.perf_counter()-start
                if block==0:
                    exact=arb_score(fp.observed[0],fp.low[0],plan,1024)
                    with ctx.workprec(1024):
                        if not(exact>=arb(float(lo[0])) and exact<=arb(float(hi[0]))):
                            raise ArithmeticError('Independent precision check failed')
                    audits[method]+=1
                buffers[method].append(dict(lo=lo,hi=hi,reject=reject,over=over,short=short,
                                           lower_lo=lowbound,lower_hi=highbound,refinements=ref))
        raw[f'D_{C}']=np.concatenate(allD);raw[f'low_{C}']=np.concatenate(alllow)
    rows=[]
    for method,plan in plans.items():
        vals={k:np.concatenate([b[k] for b in buffers[method]]) for k in
              ['lo','hi','reject','over','short','lower_lo','lower_hi']}
        for k,v in vals.items():raw[f'{method}_{k}']=v
        D=raw[f'D_{plan.C}'];T=len(D)
        errors=None if label=='Gap' else int(np.sum(vals['reject'] if label=='Yes' else ~vals['reject']))
        row=dict(suite=suite,scenario=name,cell=key,C=plan.C,L0=plan.L0,n=n,epsilon=eps,
                 S=pop.support,effective=effective,truth=label,tail_exact=str(tail),
                 mean=float(plan.m),branch=plan.branch,degree=plan.degree,scales=len(plan.scales),
                 trials=T,errors=errors,coverage_errors=int(np.sum(vals['over']|vals['short'])),
                 mean_D=float(D.mean()),mean_lower_lo=float(vals['lower_lo'].mean()),
                 mean_lower_hi=float(vals['lower_hi'].mean()),
                 improvements=int(np.sum(vals['lower_lo']>np.clip(D,1,n))),
                 extra_rejects=int(np.sum(vals['reject']&(D<=n))),
                 nonunique_bounds=int(np.sum(vals['lower_lo']!=vals['lower_hi'])),
                 refinements=sum(b['refinements'] for b in buffers[method]),
                 arb_audits=audits[method],pmf_defect=defects[plan.C],
                 sampling_seconds=timings[method][0],scoring_seconds=timings[method][1])
        rows.append(row)
    raw_file=f'raw/{key}.npz'
    np.savez_compressed(OUT/raw_file,**raw)
    dump(cell_path,dict(population=pop.as_dict(),rows=rows,raw_file=raw_file,raw_sha256=sha(OUT/raw_file)))
    return rows


def finish(rows):
    K=sum(r['errors'] is not None for r in rows)
    for r in rows:
        for event in ['errors','coverage_errors']:
            errors=r[event]
            if errors is not None:
                r[event+'_upper95_simultaneous']=1. if errors==r['trials'] else float(
                    beta.ppf(1-.05/max(K,len(rows)),errors+1,r['trials']-errors))
    with (OUT/'experiments.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for suite,C,L0 in sorted(set((r['suite'],r['C'],r['L0']) for r in rows)):
        chosen=[r for r in rows if (r['suite'],r['C'],r['L0'])==(suite,C,L0)]
        summary.append(dict(suite=suite,C=C,L0=L0,cells=len(chosen),trials=sum(r['trials'] for r in chosen),
                            errors=sum(r['errors'] or 0 for r in chosen),
                            coverage_errors=sum(r['coverage_errors'] for r in chosen),
                            polynomial_trials=sum(r['trials'] for r in chosen if r['degree']),
                            max_error_rate=max((r['errors'] or 0)/r['trials'] for r in chosen),
                            improvements=sum(r['improvements'] for r in chosen)))
    paths={}
    for r in rows:paths[(r['cell'],r['C'])]=r['trials']
    dump(OUT/'experiment_summary.json',dict(complete=True,rows=len(rows),
          method_evaluations=sum(r['trials'] for r in rows),independent_sample_paths=sum(paths.values()),
          arb_audits=sum(r['arb_audits'] for r in rows),summary=summary))


def main():
    freeze();rows=[]
    for n in DESIGN['ns']:
        for eps in DESIGN['epsilons']:
            for name,pop in synthetic_cases(n,eps):
                rows+=evaluate('synthetic',name,pop,n,eps)
        print('synthetic complete n=',n,'rows',len(rows),flush=True)
    specs=json.loads((ROOT/'data/c1025/datasets.json').read_text())
    for spec in specs:
        name=spec['dataset'];pop=Population.from_counts(load(name))
        for eps in DESIGN['natural_epsilons']:
            eff=pop.effective(eps)
            for n in sorted({max(1,eff-1),eff,pop.support,max(1,pop.support//2)}):
                rows+=evaluate('natural',name,pop,n,eps)
        for target in DESIGN['lifted_targets']:
            lifted=pop.split_categories(max(1,target//pop.support))
            for n in sorted({max(1,lifted.effective('.1')-1),lifted.effective('.1'),lifted.support}):
                rows+=evaluate('lifted',name+f'_target_{target}',lifted,n,'.1')
        print('natural/lifted complete',name,'rows',len(rows),flush=True)
    for C in DESIGN['adversarial_C']:
        methods=[(C,14)] if C=='70' else [(C,14),('70',14)]
        for n in DESIGN['adversarial_n']:
            for eps in DESIGN['adversarial_eps']:
                mean=Q(str(float(parameters(C,14,n,eps).m)))
                for lam in DESIGN['adversarial_lambda']:
                    p=Q(lam)/mean
                    mass=(n-1)*p
                    if mass>=Q(19,20):continue
                    pop=Population((1,n-1),(1-mass,p))
                    rows+=evaluate('adversarial',f'target_C{C}_lambda{lam}',pop,n,eps,methods)
        for n in [1,2,3,10]:
            pop=Population((10**6,),(Q(1,10**6),))
            rows+=evaluate('diffuse_control',f'target_C{C}',pop,n,'.99',methods)
        print('adversarial complete C=',C,'rows',len(rows),flush=True)
    finish(rows)
    print('ALL EXPERIMENTS COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
