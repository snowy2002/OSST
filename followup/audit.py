"""Reconstruct new results from raw arrays and verify frozen provenance."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import numpy as np
from flint import arb,ctx
from scipy.stats import beta

from benchmark_core import Profiles
from c7016.populations import Population
from parameter_search import algorithm as alg
from followup import core
from followup.protocol import ROOT,OUT,save,sha,csv_save
from support_size import Fingerprint


def risk_upper(errors,trials,alpha):
    return 1. if errors==trials else float(beta.isf(alpha,errors+1,trials-errors))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args(argv);out=args.out.resolve();checks=0
    def check(condition,message):
        nonlocal checks
        if not bool(condition):raise ArithmeticError(message)
        checks+=1
    protocol=json.loads((out/'protocol.json').read_text())
    for name,h in protocol['source_sha256'].items():
        check(sha(ROOT/name)==h,'Working source changed: '+name)
        check(sha(out/'source_snapshot'/name)==h,'Snapshot changed: '+name)
    for name,h in protocol['historical_sha256'].items():check(sha(ROOT/name)==h,'Historical input changed')
    rows=[];gains=[];winners=[];paths={};refined=0;refine_times=[];refine_bits=[]
    for file in sorted((out/'cells').glob('*.json')):
        cell=json.loads(file.read_text());check(sha(out/cell['raw_file'])==cell['raw_sha256'],'Raw hash mismatch')
        with np.load(out/cell['raw_file']) as raw:
            if cell['suite']=='refinement':
                row=cell['rows'][0];refined+=row['refined']
                legacy=ROOT/'results_parameter_search'/row['legacy_raw']
                check(sha(legacy)==row['legacy_raw_sha256'],'Legacy refinement source changed')
                with np.load(legacy) as old:
                    ids=raw['indices'];exact=raw['exact_lower']
                    check(len(ids)==row['refined'],'Refinement trial count')
                    check(np.all(exact>=old['70_14_lower_lo'][ids]) and np.all(exact<=old['70_14_lower_hi'][ids]),'Refinement bounds')
                refine_times.extend(raw['seconds'].tolist());refine_bits.extend(raw['bits'].tolist())
                continue
            spec=cell['population'];pop=Population(tuple(spec['multiplicities']),tuple(map(Fraction,spec['probabilities_exact'])))
            D=raw['D'];paths[file.stem]=len(D)
            for r in cell['rows']:
                m=r['method'];n=r['n'];truth,tail,eff=pop.truth(n,r['epsilon'])
                check((truth,str(tail),eff,pop.support)==(r['truth'],r['tail_exact'],r['effective'],r['S']),'Exact truth')
                reject=raw[m+'__reject'];over=raw[m+'__over'];short=raw[m+'__short']
                a=raw[m+'__lo'];b=raw[m+'__hi'];ll=raw[m+'__lower_lo'];lh=raw[m+'__lower_hi']
                check(len(D)==r['trials'],'Trial count');check(np.all(a<=b),'Ordered score bounds')
                check(np.all(reject[a>n]) and not np.any(reject[b<=n]),'Decision enclosure')
                check(np.array_equal(ll,np.clip(np.ceil(a),1,n)) and np.array_equal(lh,np.clip(np.ceil(b),1,n)),'Integer enclosure')
                check(not np.any(over) if pop.support>=n else np.all(over[a>pop.support]) and not np.any(over[b<=pop.support]),'Upper coverage')
                target=min(n,eff)
                check(not np.any(short) if target==1 else np.all(short[b<=target-1]) and not np.any(short[a>target-1]),'Lower coverage')
                binary=reject if truth=='Yes' else ~reject if truth=='No' else np.zeros(len(D),bool)
                coverage=over|short;joint=binary|coverage
                check(np.array_equal(binary,raw[m+'__binary']) and np.array_equal(joint,raw[m+'__joint']),'Raw events')
                check((int(binary.sum()) if truth!='Gap' else None)==r['errors'],'Binary count')
                check(int(coverage.sum())==r['coverage_errors'] and int(joint.sum())==r['joint_errors'],'Coverage/joint count')
                row={**r,'raw_file':cell['raw_file']};rows.append(row)
                # Full and scale-zero evaluations are independently re-scored with Arb.
                if m in ['full14','single14','no_count14'] or (m.startswith('C') and not m.endswith('_count')):
                    plan=alg.parameters(r['C'],r['L0'],n,r['epsilon'])
                    mode='single' if m=='single14' else 'no_count' if m=='no_count14' else 'full'
                    for i in [0,len(D)-1]:
                        exact=core.arb_value(D[i],raw['low'][i],plan,mode,bits=2048)
                        with ctx.workprec(2048):
                            check(exact>=arb(float(a[i])) and exact<=arb(float(b[i])),'Independent high precision score')
                if m=='count70' or m.endswith('_count') or m.startswith('count_risk_'):
                    check(np.array_equal(reject,D>n),'Count decision')
            if cell['suite']=='mechanism':
                first=cell['rows'][0];n=first['n']
                for other in ['full16','single14','count70','no_count14']:
                    if other+'__lower_lo' not in raw:continue
                    lo=np.maximum(0,(raw['full14__lower_lo']-raw[other+'__lower_hi'])/n)
                    hi=np.minimum(1,(raw['full14__lower_hi']-raw[other+'__lower_lo'])/n)
                    check(np.all(lo<=hi),'Paired monotonic gain bounds')
                    gains.append(dict(cell=first['cell'],n=n,epsilon=first['epsilon'],scenario=first['scenario'],
                         comparison='full14-'+other,trials=len(D),sample_mean_lo=float(lo.mean()),
                         sample_mean_hi=float(hi.mean()),positive_samples=int(np.sum(lo>0))))
                w=raw['winner_certified'];a,b=raw['scale_lo'],raw['scale_hi']
                check(len(w)==len(D),'Winner count')
                for j in np.unique(w):
                    ids=w==j
                    if j>=1:check(np.all(a[ids,j]>np.delete(b[ids],j,axis=1).max(axis=1)),'Unique winner')
                    if j==0 and b.shape[1]>1:check(np.all(D[ids]>=b[ids,1:].max(axis=1)),'Count dominates')
                    winners.append(dict(cell=first['cell'],n=n,epsilon=first['epsilon'],scenario=first['scenario'],
                                        winner_column=int(j),trials=int(ids.sum())))
    d=protocol['design']
    expected_mechanism=sum((5 if alg.parameters('70',14,n,ep).scales else 4)*len(d['mechanism']['scenarios'])
                           for n in d['mechanism']['ns'] for ep in d['mechanism']['epsilons'])
    expected_budget=0
    from followup.run import budget_populations
    for n in d['budget']['ns']:
        for ep in d['budget']['epsilons']:
            expected_budget+=len(budget_populations(n,ep,d['budget']))*(2*len(d['budget']['constants'])+len(d['budget']['counting_risks']))
    check(sum(r['suite']=='mechanism' for r in rows)==expected_mechanism,'Complete mechanism inventory')
    check(sum(r['suite']=='budget' for r in rows)==expected_budget,'Complete budget inventory')
    check(len({(r['cell'],r['method']) for r in rows})==len(rows),'Unique method cells')
    for suite in ['mechanism','budget']:
        selected=[r for r in rows if r['suite']==suite]
        for r in selected:
            check(r['trials']==d[suite]['trials'],'Frozen trial count')
            r['joint_upper95_simultaneous']=risk_upper(r['joint_errors'],r['trials'],.05/len(selected))
            r['binary_upper95_simultaneous']=(risk_upper(r['errors'],r['trials'],.05/len(selected)) if r['errors'] is not None else None)
            # Separate binary inferential family; not a joint claim with gain intervals.
    for r in gains:
        margin=math.sqrt(math.log(2*len(gains)/.05)/(2*r['trials']))
        r['expected_gain_lo95_simultaneous']=max(0,r['sample_mean_lo']-margin-1e-12)
        r['expected_gain_hi95_simultaneous']=min(1,r['sample_mean_hi']+margin+1e-12)
    csv_save(out/'audited_results.csv',rows);csv_save(out/'paired_gains.csv',gains);csv_save(out/'scale_winners.csv',winners)
    # Empirical frontier includes every tested budget in its confidence family.
    frontier=[]
    budget=[r for r in rows if r['suite']=='budget' and r['truth']!='Gap']
    for n,ep in sorted({(r['n'],r['epsilon']) for r in budget}):
        candidates=[r for r in budget if r['n']==n and r['epsilon']==ep]
        for family in ['full','count_same_budget']:
            methods=sorted({r['method'] for r in candidates if r['method'].startswith('C') and
                            r['method'].endswith('_count')==(family=='count_same_budget')},
                           key=lambda x:int(x.removeprefix('C').removesuffix('_count')))
            passing=[]
            for method in methods:
                selected=[r for r in candidates if r['method']==method]
                if max(r['binary_upper95_simultaneous'] for r in selected)<=.01:passing.append(selected[0])
            best=min(passing,key=lambda r:r['mean']) if passing else None
            frontier.append(dict(n=n,epsilon=ep,family=family,first_passing_C=best['C'] if best else None,
                                 first_passing_mean=best['mean'] if best else None,right_censored=best is None,
                                 scope='Frozen finite population grid, simultaneous 95% binary upper bound <=1%; not a uniform theorem'))
    csv_save(out/'empirical_frontier.csv',frontier)
    integration=json.loads((out/'integration.json').read_text())
    for r in integration['runs']:
        if r['kind']=='actual_iid_labels':
            check(sha(out/r['counts_file'])==r['counts_sha256'],'Integration count hash')
            with np.load(out/r['counts_file']) as stored:fp=Fingerprint.from_counts(stored['counts'])
        else:
            check(sha(out/r['fingerprint_file'])==r['fingerprint_sha256'],'Integration fingerprint hash')
            fp=Fingerprint({int(k):v for k,v in json.loads((out/r['fingerprint_file']).read_text()).items()})
        plan=alg.parameters('70',14,r['n'],r['epsilon'])
        actual=core.test_fingerprint(fp,r['n'],r['epsilon'],mean=plan.m)
        check((actual['decision'],actual['lower'],actual['M'])==(r['decision'],r['lower'],r['M']),'Full API reproduction')
    expected_integration=len(d['numeric']['explicit_count_ns'])*len(d['numeric']['scenarios'])*d['numeric']['explicit_replicates']+d['numeric']['label_replicates']
    check(len(integration['runs'])==expected_integration,'Complete full API inventory')
    formula=json.loads((out/'independent_coefficients.json').read_text())
    check(formula['passed'] and formula['checks']>0,'Independent closed-form coefficients')
    guard_out=out/'count_guard'
    guard_summary=None
    if guard_out.exists():
        guard_protocol=json.loads((guard_out/'protocol.json').read_text())
        guard_complete=json.loads((guard_out/'complete.json').read_text())
        for name,h in guard_protocol['source_sha256'].items():
            check(sha(ROOT/name)==h,'Guard source unchanged')
            check(sha(guard_out/'source_snapshot'/name)==h,'Guard snapshot unchanged')
        guard_rows=list(csv.DictReader((guard_out/'results.csv').open()))
        check(len(guard_rows)==12,'Guard complete method inventory')
        for r in guard_rows:
            n=int(r['n']);ep=Fraction(r['epsilon']);plan=alg.parameters('70',14,n,ep)
            pop=(Population((n,),(Fraction(1,n),)) if r['scenario']=='uniform' else
                 Population((n-1,1),((1-2*ep)/(n-1),2*ep)))
            path=guard_out/r['raw_file']
            check(sha(path)==guard_complete['raw_sha256'][path.name],'Guard raw hash')
            with np.load(path) as raw:
                fp=Profiles(raw['D'],raw['low']);a,b=core.scale_intervals(fp,plan)
                events,_=core.outcomes(fp,plan,pop,a,b,r['method'])
                for field in ['reject','over','short','lower_lo','lower_hi']:
                    check(np.array_equal(events[field],raw[r['method']+'__'+field]),'Guard recomputed events')
                coverage=int((events['over']|events['short']).sum())
                check(coverage==int(r['coverage_errors']),'Guard coverage count')
                check(pop.effective(ep)==n,'Guard exact effective support')
                r['coverage_upper95_simultaneous']=risk_upper(coverage,len(fp.observed),.05/len(guard_rows))
                r['coverage_lower95_simultaneous']=0. if coverage==0 else float(beta.ppf(.05/len(guard_rows),coverage,len(fp.observed)-coverage+1))
        csv_save(guard_out/'audited_results.csv',guard_rows)
        guard_summary=dict(independent_paths=guard_complete['independent_paths'],
             method_evaluations=guard_complete['method_evaluations'],
             no_count_coverage_errors=sum(int(r['coverage_errors']) for r in guard_rows if r['method']=='no_count'),
             full_coverage_errors=sum(int(r['coverage_errors']) for r in guard_rows if r['method']=='full'),
             disclosure=guard_protocol['design']['disclosure'])
    summary=dict(passed=True,checks=checks,fresh_population_budget_paths=sum(paths.values()),
        method_trial_evaluations=sum(r['trials'] for r in rows),method_cells=len(rows),
        mechanism_paths=sum(v for k,v in paths.items() if any(r['cell']==k and r['suite']=='mechanism' for r in rows)),
        refined_archived_outputs=refined,refinement_seconds=sum(refine_times),
        refinement_seconds_median=float(np.median(refine_times)),refinement_seconds_p95=float(np.quantile(refine_times,.95)),
        refinement_max_bits=max(refine_bits),refinement_changed_decisions=0,refinement_changed_coverage=0,
        full_api_runs=len(integration['runs']),independent_coefficient_checks=formula['checks'],
        post_primary_guard=guard_summary,
        source_sha256=sha(Path(__file__)),
        scope='Hash, exact truth, raw events, paired bounds, and representative higher-precision scores. Not a new proof or exact PRNG certification.')
    if not protocol['design'].get('smoke'):check(refined==37071,'All historical ambiguous outputs must be refined')
    summary['checks']=checks
    save(out/'audit.json',summary)
    print('AUDIT PASS',summary,flush=True)


if __name__=='__main__':main()
