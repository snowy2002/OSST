"""Development, fixed-budget, sequential, and adversarial experiment suites."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmark_core import (Population, Profiles, sample_profiles, stable_rng, uniform, headtail, rank_blocks)
from experiments import write_csv
from guaranteed_experiments import simulate_trajectory, validation_draws
from publication.core import (BASE, FAMILIES, signature, compile_scales, certified_reject, certified_lower,
                              fh_native_reject, trial_summary, cp_upper, make_seq_plan, decide,
                              formal_cases, near_boundary, scale_rejections)
from publication.protocol import OUT, DESIGN, SEEDS, save, sha

K=24


def candidate_configs():
    return {
      "multi": [{**BASE,"b_factor":b,"weight_cap":cap} for b in [.25,.5,1.,2.] for cap in [10.,20.]],
      "single": [{**BASE,"b_factor":b,"single_j":j} for b in [.5,1.] for j in [0,2,4,6]],
      "fh_certified": [{**BASE,"fh_a":a,"fh_b":b,"fh_k":k} for a in [.03125,.125] for b in [.25,1.] for k in [.25,.5]],
    }


def sample_blocks(pop,m,trials,seed,*key):
    # Four separately keyed random blocks expose run-to-run variation in raw data.
    chunks=np.array_split(np.arange(trials),4)
    fps=[sample_profiles(pop,m,len(c),K,stable_rng(seed,*key,"replicate",i)) for i,c in enumerate(chunks) if len(c)]
    return Profiles(np.concatenate([p.observed for p in fps]),np.concatenate([p.low for p in fps]))


def add_intervals(rows, alpha=.05):
    tested=[r for r in rows if r.get("errors") is not None]
    H=max(1,len(tested))
    for r in rows:
        r["error_simultaneous_upper95"]=None if r.get("errors") is None else cp_upper(r["errors"],r["trials"],alpha/H)
        r["simultaneous_family_size"]=H
    return rows


def finish(out,name,rows):
    add_intervals(rows)
    write_csv(out/f"{name}.csv",rows)
    save(out/f"{name}_complete.json",dict(rows=len(rows),method_trials=sum(r["trials"] for r in rows),
                                         sha256=sha(out/f"{name}.csv")))


def development(out=OUT, quick=False):
    design=DESIGN["development"]
    choices=candidate_configs()
    choices["fh_native_tuned"]=choices["fh_certified"]
    rows=[]
    trials=40 if quick else design["trials"]
    for n in design["sizes"][:1] if quick else design["sizes"]:
      for eps in design["epsilons"][:1] if quick else design["epsilons"]:
        cases=[("uniform_null",uniform(n)),("head_null",headtail(.83,n-1)),
               ("power_null",rank_blocks(n,1.07)),("uniform_no",uniform(math.ceil(n/(1-eps))+3)),
               ("tail_no",near_boundary(n,eps,1.04,40)),("layer_no",near_boundary(n,eps,1.03,40,.73))]
        for C in design["factors"]:
          m=math.ceil(C*n/(eps*(1+math.log(n))))
          compiled={(fam,i):compile_scales(n,str(eps),m,fam,signature(cfg))
                    for fam in FAMILIES for i,cfg in enumerate(choices[fam])}
          for case,pop in cases:
            fp=sample_blocks(pop,m,trials,SEEDS["development"],n,eps,C,case)
            truth=pop.truth(n,eps)
            for fam,candidates in choices.items():
              for i,cfg in enumerate(candidates):
                reject=(fh_native_reject(fp,n,eps,m,cfg) if fam=="fh_native_tuned" else
                        certified_reject(fp,compiled[fam,i],n))
                rows.append(dict(family=fam,candidate=i,n=n,epsilon=eps,C=C,scenario=case,truth=truth,
                                 **trial_summary(reject,None,truth)))
        print(f"development n={n} eps={eps}: complete",flush=True)
    ranking=[]
    selected={}
    for fam,candidates in choices.items():
      local=[]
      for i,cfg in enumerate(candidates):
        rr=[r for r in rows if r["family"]==fam and r["candidate"]==i]
        fp=max(r["error_rate"] for r in rr if r["truth"]=="Yes")
        fn=float(np.mean([r["error_rate"] for r in rr if r["truth"]=="No"]))
        # All certified candidates share a mathematical null guarantee. For the
        # heuristic preserve failures and penalize null inflation before power.
        loss=fn+10*max(0,fp-.05)
        local.append(dict(family=fam,candidate=i,loss=loss,max_fp=fp,mean_fn=fn,config=cfg))
      local.sort(key=lambda r:(r["loss"],r["candidate"]))
      selected[fam]=local[0]["config"]
      ranking.extend(local)
    write_csv(out/"development.csv",rows)
    save(out/"calibration.json",dict(configs=selected,ranking=ranking,seed=SEEDS["development"],
        trials=trials,criterion="Mean failure-to-reject over all No development cells; heuristic adds 10*max(0,max_FP-.05). Eight candidates per family. No formal-data tuning."))
    return selected


def fixed_cell(pop,n,eps,m,trials,configs,key,seed=SEEDS["formal"],lower=True,alpha=.05,fp=None):
    fp=sample_blocks(pop,m,trials,seed,*key) if fp is None else fp
    if fp.trials!=trials:raise ValueError("Trial count mismatch")
    truth=pop.truth(n,eps)
    outputs={"count":fp.observed>n}
    scales={fam:compile_scales(n,str(eps),m,fam,signature(configs[fam]),str(alpha)) for fam in FAMILIES}
    outputs.update({fam:certified_reject(fp,ss,n) for fam,ss in scales.items()})
    outputs["fh_native_tuned"]=fh_native_reject(fp,n,eps,m,configs["fh_native_tuned"])
    f1,f2=fp.low[:,0].astype(float),fp.low[:,1].astype(float)
    outputs["chao1_point"]=(fp.observed+f1*(f1-1)/(2*(f2+1)))>n
    rows=[]
    raw={f"reject_{fam}":arr for fam,arr in outputs.items()}
    raw["observed"]=fp.observed
    eff=pop.effective_support(eps)
    for fam,reject in outputs.items():
      r=dict(method=fam,n=n,epsilon=eps,delta=alpha,m=m,truth=truth,S=pop.support,
             delta_n=pop.tail_mass(n),effective_support=eff,**trial_summary(reject,None,truth),
             multi_only_reject=int(np.sum(outputs["multi"] & ~reject)),
             comparator_only_reject=int(np.sum(reject & ~outputs["multi"])))
      if lower and fam in ("count",*FAMILIES):
        values=np.minimum(n,fp.observed) if fam=="count" else certified_lower(fp,scales[fam],n)
        values=np.maximum(1,values)
        over=values>pop.support
        under=values<min(n,eff)
        r.update(lower_over_errors=int(over.sum()),lower_under_errors=int(under.sum()),
                 lower_joint_errors=int((over|under).sum()),lower_joint_success=float(np.mean(~(over|under))),
                 mean_lower=float(values.mean()),mean_gain_over_clipped_D=float(np.mean(values-np.minimum(n,fp.observed))))
        raw[f"lower_{fam}"]=values
      else:
        r.update({k:None for k in ["lower_over_errors","lower_under_errors","lower_joint_errors",
                                  "lower_joint_success","mean_lower","mean_gain_over_clipped_D"]})
      rows.append(r)
    mechanism=[]
    hits=scale_rejections(fp,scales["multi"],n)
    if hits:
      total=np.sum(hits,axis=0)
      for s,hit in zip(scales["multi"],hits):
        mechanism.append(dict(j=s.certificate.j,degree=s.certificate.degree,A=s.A,B=s.B,
                              trigger_rate=float(hit.mean()),exclusive_rate=float(np.mean(hit & (total==1))),
                              delta_power_if_removed=float(np.mean(hit & (total==1))),
                              rho_upper=s.rho_upper,margin_upper=s.margin_upper))
        raw[f"scale_{s.certificate.j}"]=hit
    return rows,raw,mechanism


def fixed_suite(configs,out=OUT,quick=False):
    design=DESIGN["fixed"]
    rows,mechanism=[],[]
    trials=40 if quick else design["trials"]
    (out/"raw").mkdir(exist_ok=True)
    for n in design["sizes"][:1] if quick else design["sizes"]:
      for eps in design["epsilons"][:1] if quick else design["epsilons"]:
        raw={}
        factors=design["factors"][:2] if quick else design["factors"]
        from guaranteed import certified_count_budget
        cap,_=certified_count_budget(n,Fraction(str(eps)),Fraction(1,20))
        budgets=[(f"C{C:g}",C,math.ceil(C*n/(eps*(1+math.log(n))))) for C in factors]
        budgets.append(("count_cap",None,cap))
        name=f"fixed_n{n}_e{eps:g}.npz"
        for case,pop in formal_cases(n,eps):
          for budget,C,m in budgets:
            key=("fixed",n,eps,case,budget)
            rr,arrays,mm=fixed_cell(pop,n,eps,m,trials,configs,key)
            prefix=f"{case}__{budget}"
            for r in rr:
              rows.append(dict(suite="fixed",scenario=case,budget=budget,C=C,raw_file="raw/"+name,
                               raw_prefix=prefix,**r))
            mechanism.extend(dict(n=n,epsilon=eps,scenario=case,m=m,C=C,truth=pop.truth(n,eps),trials=trials,**a) for a in mm)
            raw.update({prefix+"__"+k:v for k,v in arrays.items()})
        np.savez_compressed(out/"raw"/name,**raw)
        write_csv(out/"fixed_progress.csv",rows)
        print(f"fixed n={n} eps={eps}: {len(rows)} method cells",flush=True)
    finish(out,"fixed",rows)
    # Three separate failure criteria are protected as a single additional family.
    H=3*sum(r["lower_joint_errors"] is not None for r in rows)
    for r in rows:
      for metric in ["over","under","joint"]:
        k=r[f"lower_{metric}_errors"]
        r[f"lower_{metric}_simultaneous_upper95"]=None if k is None else cp_upper(k,r["trials"],.05/max(1,H))
    write_csv(out/"fixed.csv",rows)
    save(out/"fixed_complete.json",dict(rows=len(rows),method_trials=sum(r["trials"] for r in rows),sha256=sha(out/"fixed.csv")))
    write_csv(out/"scale_mechanism.csv",mechanism)
    return rows


def seq_plans(n,eps,configs,delta=.05):
    return {"count_batch":make_seq_plan(n,str(eps),str(delta),"none",False,signature(configs["multi"])),
            "count_validation":make_seq_plan(n,str(eps),str(delta),"none",True,signature(configs["multi"])),
            "polynomial_only":make_seq_plan(n,str(eps),str(delta),"multi",False,signature(configs["multi"])),
            "multi_hybrid":make_seq_plan(n,str(eps),str(delta),"multi",True,signature(configs["multi"])),
            "single_hybrid":make_seq_plan(n,str(eps),str(delta),"single",True,signature(configs["single"])),
            "fh_hybrid":make_seq_plan(n,str(eps),str(delta),"fh_certified",True,signature(configs["fh_certified"]))}


def sequential_cell(pop,n,eps,trials,configs,key,seed=SEEDS["formal"],delta=.05,ablation=False):
    plans=seq_plans(n,eps,configs,delta)
    means=sorted({m for p in plans.values() for m in [p.final_m,*[l.m for l in p.looks]]})
    degree=max(p.max_degree for p in plans.values())
    stages=simulate_trajectory(pop,means,trials,degree,stable_rng(seed,"trajectory",*key))
    profiles={m:s.profiles for m,s in stages.items()}
    outputs={}
    for method,plan in plans.items():
      # Identical validation plans share their outside draws (common random numbers).
      draws=validation_draws(plan,stages,stable_rng(seed,"validation",*key,str(delta)))
      outputs[method]=decide(plan,profiles,draws)
    base=outputs["count_batch"]["cost"]
    rows=[]
    truth=pop.truth(n,eps)
    for method,output in outputs.items():
      r=dict(method=method,n=n,epsilon=eps,delta=delta,truth=truth,S=pop.support,delta_n=pop.tail_mass(n),
             cost_metric="training_poisson_mean_plus_validation",**trial_summary(output["reject"],output["cost"],truth,base))
      r.update({f"{reason}_rate":float(np.mean(output["reason"]==reason)) for reason in ["observed","polynomial","validation","fallback"]})
      r["mean_validation"]=float(output["validation"].mean())
      rows.append(r)
    removed=[]
    if ablation:
      p=plans["multi_hybrid"]
      draws=validation_draws(p,stages,stable_rng(seed,"validation",*key,str(delta)))
      full=outputs["multi_hybrid"]
      for j in sorted({c.j for look in p.looks for c in look.certificates}):
        o=decide(p,profiles,draws,omit_j=j)
        removed.append(dict(n=n,epsilon=eps,j=j,trials=trials,truth=truth,
                            first_winner_rate=float(np.mean(full["scale"]==j)),
                            mean_extra_cost=float(np.mean(o["cost"]-full["cost"])),
                            changed_decisions=int(np.sum(o["reject"]!=full["reject"]))))
    return rows,outputs,removed


def sequential_suite(configs,out=OUT,quick=False):
    d=DESIGN["sequential"]
    rows,mechanism=[],[]
    for n in d["sizes"][:1] if quick else d["sizes"]:
      for eps in d["epsilons"][:1] if quick else d["epsilons"]:
        arrays={}
        name=f"sequential_n{n}_e{eps:g}.npz"
        for scenario,pop in formal_cases(n,eps):
          rr,outputs,mm=sequential_cell(pop,n,eps,40 if quick else d["trials"],configs,
                                       ("sequential",n,eps,scenario),ablation=True)
          rows.extend(dict(scenario=scenario,raw_file="raw/"+name,**r) for r in rr)
          mechanism.extend(dict(scenario=scenario,**r) for r in mm)
          for method,o in outputs.items():
            arrays.update({f"{scenario}__{method}__{k}":v for k,v in o.items()})
        np.savez_compressed(out/"raw"/name,**arrays)
        print(f"sequential n={n} eps={eps}: complete",flush=True)
    finish(out,"sequential",rows)
    write_csv(out/"scale_removal.csv",mechanism)
    return rows


def adversarial_search(configs,out=OUT,quick=False):
    d=DESIGN["adversarial"]
    rows,selected=[],[]
    for n in d["sizes"][:1] if quick else d["sizes"]:
      for eps in d["epsilons"][:1] if quick else d["epsilons"]:
        candidates=[]
        for f in [2,10,100,1000]:
          for exponent in [None,.4,.8,1.2]:
            for ratio in [1.,1.01]:
              try:
                p=near_boundary(n,eps,ratio,f,exponent)
              except ValueError:
                continue
              candidates.append((f"tail_f{f}_a{exponent}_r{ratio}",p))
        for sigma in [1.,2.,4.,8.]:
          for variant in range(3):
            rng=stable_rng(SEEDS["search"],"null_shape",n,eps,sigma,variant)
            g=np.diff(np.linspace(0,n,49).astype(int))
            w=np.exp(rng.normal(0,sigma,len(g)))
            candidates.append((f"lognormal_s{sigma}_v{variant}",Population(g,w/(g@w))))
        if quick:candidates=candidates[:3]+candidates[-2:]
        local=[]
        m=math.ceil(4*n/(eps*(1+math.log(n))))
        for i,(name,pop) in enumerate(candidates):
          rr,_,_=fixed_cell(pop,n,eps,m,40 if quick else d["search_trials"],configs,
                            ("search",n,eps,name),SEEDS["search"],lower=False)
          sr,_,_=sequential_cell(pop,n,eps,40 if quick else d["search_trials"],configs,
                                ("search",n,eps,name),SEEDS["search"])
          fixed=next(r for r in rr if r["method"]=="multi")
          seq=next(r for r in sr if r["method"]=="multi_hybrid")
          r=dict(n=n,epsilon=eps,candidate=i,scenario=name,truth=pop.truth(n,eps),S=pop.support,
                 delta_n=pop.tail_mass(n),fixed_error=fixed["error_rate"],sequential_error=seq["error_rate"],
                 relative_cost=seq["relative_cost"],fallback_rate=seq["fallback_rate"],
                 trials=40 if quick else d["search_trials"])
          local.append((r,pop));rows.append(r)
        # Three null and three alternative candidates, sorted by primary fixed
        # error then sequential cost. Tie-break by name, never test outcomes.
        for truth in ["Yes","No"]:
          eligible=sorted([(r,p) for r,p in local if r["truth"]==truth],
                          key=lambda rp:(-rp[0]["fixed_error"],-rp[0]["relative_cost"],rp[0]["scenario"]))[:3]
          for r,p in eligible:
            selected.append(dict(**r,multiplicities=p.multiplicities.tolist(),probabilities=p.probabilities.tolist()))
        print(f"adversarial search n={n} eps={eps}: {len(candidates)} candidates",flush=True)
    write_csv(out/"adversarial_search.csv",rows)
    save(out/"adversarial_selected.json",dict(selection_rule="Top 3 per Yes/No and n/epsilon; descending fixed C=4 error then relative sequential cost",populations=selected,search_sha256=sha(out/"adversarial_search.csv")))
    return selected


def adversarial_test(configs,out=OUT,quick=False):
    frozen=json.loads((out/"adversarial_selected.json").read_text())
    save(out/"adversarial_test_protocol.json",dict(selected_sha256=sha(out/"adversarial_selected.json"),seed=SEEDS["adversarial_test"],design=DESIGN["adversarial"]))
    fixed,seq=[],[]
    d=DESIGN["adversarial"]
    for i,p in enumerate(frozen["populations"]):
      pop=Population(np.array(p["multiplicities"]),np.array(p["probabilities"]))
      n,eps=p["n"],p["epsilon"]
      arrays={}
      for C in [2,3,4,6,8]:
        m=math.ceil(C*n/(eps*(1+math.log(n))))
        rr,raw,_=fixed_cell(pop,n,eps,m,80 if quick else d["test_trials"],configs,
                            ("adversarial",i,C),SEEDS["adversarial_test"])
        fixed.extend(dict(candidate=i,scenario=p["scenario"],C=C,**r) for r in rr)
        arrays.update({f"C{C}__{k}":v for k,v in raw.items()})
      rr,outputs,_=sequential_cell(pop,n,eps,80 if quick else d["sequential_test_trials"],configs,
                                  ("adversarial",i),SEEDS["adversarial_test"])
      seq.extend(dict(candidate=i,scenario=p["scenario"],**r) for r in rr)
      for method,o in outputs.items():arrays.update({f"seq__{method}__{k}":v for k,v in o.items()})
      np.savez_compressed(out/"raw"/f"adversarial_{i}.npz",**arrays)
      print(f"independent adversarial test {i+1}/{len(frozen['populations'])}",flush=True)
    finish(out,"adversarial_fixed",fixed)
    finish(out,"adversarial_sequential",seq)
