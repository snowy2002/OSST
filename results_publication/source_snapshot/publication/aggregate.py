"""Outcome-only analysis; this module never selects or changes tested methods."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from publication.protocol import ROOT, OUT, sha, save

SUITES=["fixed","sequential","adversarial_fixed","adversarial_sequential","stream",
        "structure","group_holdout","sensitivity","reference"]


def read_all(out=OUT):
    return {name:pd.read_csv(out/f"{name}.csv") for name in [*SUITES,"runtime","scale_mechanism","scale_removal"]}


def exact_effective_support(pop,epsilon):
    weights=[Fraction.from_float(float(x)) for x in pop.probabilities]
    mass=sum(int(g)*p for g,p in zip(pop.multiplicities,weights))
    remaining=Fraction(str(epsilon))*mass
    discard=0
    for g,p in zip(pop.multiplicities[::-1],weights[::-1]):
        g=int(g)
        if g*p<=remaining:
            discard+=g;remaining-=g*p
        else:
            discard+=min(g,remaining//p);break
    return max(1,pop.support-discard)


def exact_lower_metrics(frames,out):
    """Canonical analysis uses rational integer targets, keeping raw CSVs immutable.

    Legacy effective_support uses a floating floor that can be off by one at an
    equality. Recompute targets and the saved-trial metrics, not the algorithms.
    This changes neither parameters nor random draws nor binary decisions.
    """
    from benchmark_core import Population
    from publication.core import formal_cases,exact_tail_mass
    from publication.structure import load_matrix,empirical
    selected=json.loads((out/"adversarial_selected.json").read_text())["populations"]
    populations={};matrices={};changes=[];target_changes=0;changed_error_cells=0
    for suite in ["fixed","adversarial_fixed","group_holdout"]:
      frame=frames[suite].copy()
      frame["effective_support_float_diagnostic"]=frame.effective_support
      for idx,r in frame.iterrows():
        if suite=="fixed":
          key=(suite,int(r.n),float(r.epsilon),r.scenario)
          if key not in populations:
            populations.update({(suite,int(r.n),float(r.epsilon),name):pop for name,pop in formal_cases(int(r.n),float(r.epsilon))})
        elif suite=="adversarial_fixed":
          key=(suite,int(r.candidate))
          if key not in populations:
            spec=selected[int(r.candidate)];populations[key]=Population(np.array(spec["multiplicities"]),np.array(spec["probabilities"]))
        else:
          key=(suite,r.dataset,int(r.split))
          if key not in populations:
            if r.dataset not in matrices:matrices[r.dataset]=load_matrix(r.dataset)
            with np.load(out/r.raw_file) as raw:groups=raw["test_groups"]
            populations[key]=empirical(matrices[r.dataset][groups])
        pop=populations[key]
        # Cache each exact target, since many methods and budgets share a population.
        target_key=("exact",key,float(r.epsilon))
        if target_key not in populations:populations[target_key]=exact_effective_support(pop,float(r.epsilon))
        eff=populations[target_key]
        if eff!=int(r.effective_support):
          changes.append(dict(suite=suite,row=int(idx),population_key=str(key),old_effective_support=int(r.effective_support),exact_effective_support=eff))
          frame.at[idx,"effective_support"]=eff
          if min(eff,int(r.n))!=min(int(r.effective_support),int(r.n)) and pd.notna(r.lower_joint_errors):
            target_changes+=1
            if suite=="fixed":filename=r.raw_file;prefix=r.raw_prefix+"__"
            elif suite=="adversarial_fixed":filename=f"raw/adversarial_{int(r.candidate)}.npz";prefix=f"C{int(r.C)}__"
            else:filename=r.raw_file;prefix=f"{r.scenario}_C{int(r.C)}__"
            with np.load(out/filename) as raw:values=raw[prefix+"lower_"+r.method]
            over=values>int(r.S);under=values<min(eff,int(r.n))
            new_joint=int((over|under).sum())
            changed_error_cells+=new_joint!=int(r.lower_joint_errors)
            frame.at[idx,"lower_under_errors"]=int(under.sum())
            frame.at[idx,"lower_joint_errors"]=new_joint
            frame.at[idx,"lower_joint_success"]=float(np.mean(~(over|under)))
      if suite=="fixed":
        from publication.core import cp_upper
        H=3*int(frame.lower_joint_errors.notna().sum())
        for idx,r in frame[frame.lower_joint_errors.notna()].iterrows():
          for metric in ["over","under","joint"]:
            frame.at[idx,f"lower_{metric}_simultaneous_upper95"]=cp_upper(int(r[f"lower_{metric}_errors"]),int(r.trials),.05/H)
      frame.to_csv(out/"tables"/f"{suite}_exact_metrics.csv",index=False)
      frames[suite]=frame
    save(out/"effective_support_audit.json",dict(changed_metadata_rows=len(changes),
        affected_population_epsilon_pairs=len({(r["suite"],r["population_key"],r["old_effective_support"],r["exact_effective_support"]) for r in changes}),
        changed_clipped_targets_in_method_cells=target_changes,changed_lower_error_cells=int(changed_error_cells),
        changes=changes,policy="Raw simulation CSVs preserved. Exact-target analysis CSVs in tables/ are canonical for lower-bound metrics; no decisions, costs, tuning or samples changed."))
    return frames


def verify(out):
    protocol=json.loads((out/"protocol.json").read_text())
    issues=[]
    for name,digest in protocol["source_hashes"].items():
      if sha(ROOT/name)!=digest:issues.append("source hash: "+name)
    for name,digest in protocol["data_hashes"].items():
      if sha(ROOT/"data/large/prepared"/name)!=digest:issues.append("data hash: "+name)
    # Fish raw records were already pinned by the legacy preparation code. Verify
    # the derived stream independently rather than trusting its current filename.
    from fish_data import SHA256
    if sha(ROOT/"data/InchLake1.csv")!=SHA256:issues.append("fish raw source hash")
    original=list(csv.DictReader((ROOT/"data/InchLake1.csv").open(encoding="utf-8-sig")))
    expected=[dict(record_id=str(i),year=r["year"],gear=r["netType"],event_id=r["netID"],
                   species=r["species"],length_inch=r["length"]) for i,r in enumerate(original,1)
              if r["species"]!="Rusty Crayfish"]
    derived=list(csv.DictReader((ROOT/"data/inch_lake_fish.csv").open()))
    if derived!=expected:issues.append("fish derived records disagree with pinned raw cleaning")
    save(out/"fish_data_audit.json",dict(raw_sha256=SHA256,derived_sha256=sha(ROOT/"data/inch_lake_fish.csv"),
                                      raw_records=len(original),derived_records=len(derived),matches_pinned_cleaning=derived==expected,
                                      scope="Post-run independent verification against pre-existing pinned raw-source hash."))
    for phase in ["fixed","sequential","search","adversarial","stream","structure","sensitivity","reference","runtime"]:
      path=out/f"execution_{phase}.json"
      if not path.exists() or json.loads(path.read_text())["status"]!="complete":issues.append("unfinished phase: "+phase)
    for suite in SUITES:
      manifest=json.loads((out/f"{suite}_complete.json").read_text())
      if sha(out/f"{suite}.csv")!=manifest["sha256"]:issues.append("result hash: "+suite)
    raw_checked=0
    fixed=pd.read_csv(out/"fixed.csv")
    for filename,group in fixed.groupby("raw_file",sort=False):
      with np.load(out/filename) as data:
        for r in group.itertuples():
          reject=data[r.raw_prefix+"__reject_"+r.method]
          if len(reject)!=r.trials or reject.sum()!=r.rejects:issues.append("raw fixed: "+r.raw_prefix)
          if pd.notna(r.lower_joint_errors):
            values=data[r.raw_prefix+"__lower_"+r.method]
            errors=(values>r.S)|(values<min(r.n,r.effective_support))
            if int(errors.sum())!=r.lower_joint_errors:issues.append("raw lower: "+r.raw_prefix)
          raw_checked+=1
    for suite in ["stream","sequential"]:
      frame=pd.read_csv(out/f"{suite}.csv")
      for filename,group in frame.groupby("raw_file",sort=False):
        with np.load(out/filename) as data:
          for r in group.itertuples():
            prefix=(r.scenario+"__" if suite=="sequential" else "")+r.method+"__"
            if not math.isclose(float(data[prefix+"cost"].mean()),r.mean_cost,rel_tol=1e-12):issues.append("raw cost: "+prefix)
            if int(data[prefix+"reject"].sum())!=r.rejects:issues.append("raw decision: "+prefix)
            raw_checked+=1
    for suite in ["adversarial_fixed","adversarial_sequential","structure","group_holdout","sensitivity","reference"]:
      frame=pd.read_csv(out/f"{suite}.csv")
      cache={}
      for r in frame.itertuples():
        if suite.startswith("adversarial"):
          filename=f"raw/adversarial_{r.candidate}.npz"
          prefix=f"C{int(r.C)}__" if suite=="adversarial_fixed" else "seq__"+r.method+"__"
        elif suite=="reference":
          filename=f"raw/reference_n{r.n}_e{r.epsilon:g}_{r.scenario}.npz";prefix=""
        else:
          filename=r.raw_file
          prefix=(f"{r.scenario}_C{int(r.C)}__" if suite=="group_holdout" else
                  f"{r.scenario}__{r.method}__" if suite=="sensitivity" else "")
        if filename not in cache:
          # Keep only one archive open to bound file handles and memory.
          for old in cache.values():old.close()
          cache={filename:np.load(out/filename)}
        data=cache[filename]
        if suite=="reference":
          votes=data["wang_votes" if r.method.startswith("wang") else "fh_votes"]
          reject=votes>r.repetitions//2
        elif suite in ["adversarial_sequential","sensitivity"]:
          reject=data[prefix+"reject"]
          if not math.isclose(float(data[prefix+"cost"].mean()),r.mean_cost,rel_tol=1e-12):issues.append("raw cost: "+suite+prefix)
        else:reject=data[prefix+"reject_"+r.method]
        if len(reject)!=r.trials or int(reject.sum())!=r.rejects:issues.append("raw decisions: "+suite+prefix)
        if r.truth!="Gap":
          errors=int(np.sum(reject if r.truth=="Yes" else ~reject))
          if errors!=r.errors:issues.append("raw error count: "+suite+prefix)
        raw_checked+=1
      for data in cache.values():data.close()
    record=dict(passed=not issues,issues=issues,raw_method_cells_checked=raw_checked,
                evaluation_source_hashes_checked=len(protocol["source_hashes"]),data_hashes_checked=len(protocol["data_hashes"]))
    save(out/"result_audit.json",record)
    if issues:raise RuntimeError(str(issues[:10]))
    return record


def efficiency(frame):
    results=[]
    for (n,eps,method),g in frame[frame.truth!="Gap"].groupby(["n","epsilon","method"]):
      by=g.groupby(["budget","m"],dropna=False).agg(worst_upper=("error_simultaneous_upper95","max"),
                                                    worst_observed=("error_rate","max")).reset_index()
      passing=by[by.worst_upper<=.05].sort_values("m")
      m=None if passing.empty else int(passing.iloc[0].m)
      results.append(dict(n=int(n),epsilon=float(eps),method=method,m_star=m,
                          normalized_n_over_eps=None if m is None else m*eps/n,
                          normalized_n_over_eps_log=None if m is None else m*eps*(1+math.log(n))/n,
                          right_censored=m is None,largest_tested_m=int(by.m.max()),
                          worst_upper_at_m_star=None if m is None else float(passing.iloc[0].worst_upper)))
    return pd.DataFrame(results)


def cost_summary(frames):
    records=[]
    rng=np.random.default_rng(193311)
    for suite in ["sequential","stream","adversarial_sequential"]:
      for (truth,method),g in frames[suite].query("truth != 'Gap'").groupby(["truth","method"]):
        x=g.relative_cost.to_numpy()
        boot=np.mean(rng.choice(x,size=(5000,len(x))),axis=1)
        records.append(dict(suite=suite,truth=truth,method=method,cells=len(g),
                            mean_relative_cost=float(x.mean()),median_relative_cost=float(np.median(x)),
                            worst_relative_cost=float(x.max()),grid_bootstrap_low95=float(np.quantile(boot,.025)),
                            grid_bootstrap_high95=float(np.quantile(boot,.975)),
                            max_error=float(g.error_rate.max()),max_error_upper=float(g.error_simultaneous_upper95.max()),
                            mean_validation=float(g.mean_validation.mean())))
    return pd.DataFrame(records)


def paired_costs(frames,out):
    cells=[]
    for suite in ["stream","sequential"]:
      frame=frames[suite]
      base="count_fixed_immediate" if suite=="stream" else "count_batch"
      for filename,g in frame.groupby("raw_file",sort=False):
        with np.load(out/filename) as data:
          for (scenario,n,eps,truth),gg in g.groupby(["scenario","n","epsilon","truth"]):
            if truth=="Gap":continue
            prefix=scenario+"__" if suite=="sequential" else ""
            a=data[prefix+"multi_hybrid__cost"].astype(float)
            b0=data[prefix+base+"__cost"].astype(float)
            for comparator in [base,"count_validation","single_hybrid","fh_hybrid"]:
              b=data[prefix+comparator+"__cost"].astype(float)
              diff=a-b;r=diff.mean()/b0.mean()
              se=(diff-r*b0).std(ddof=1)/(math.sqrt(len(a))*b0.mean())
              cells.append(dict(suite=suite,scenario=scenario,n=int(n),epsilon=float(eps),truth=truth,
                                 comparator=comparator,normalized_paired_difference=float(r),mc_se=float(se),trials=len(a)))
    detail=pd.DataFrame(cells)
    aggregate=[]
    for (suite,truth,comparator),g in detail.groupby(["suite","truth","comparator"]):
      effect=float(g.normalized_paired_difference.mean());se=float(np.linalg.norm(g.mc_se)/len(g))
      p=float(2*norm.sf(abs(effect)/se)) if se>0 else (0. if effect else 1.)
      aggregate.append(dict(suite=suite,truth=truth,comparator=comparator,cells=len(g),effect=effect,mc_se=se,
                            mc_low95=effect-1.96*se,mc_high95=effect+1.96*se,pvalue=p))
    agg=pd.DataFrame(aggregate)
    order=np.argsort(agg.pvalue.to_numpy());adj=np.zeros(len(agg));running=0.
    for rank,i in enumerate(order):
      running=max(running,min(1.,float(agg.iloc[i].pvalue)*(len(agg)-rank)));adj[i]=running
    agg["holm_pvalue"]=adj
    detail.to_csv(out/"tables/paired_cost_cells.csv",index=False)
    return agg


def seed_blocks(frame,out):
    rows=[]
    for filename,g in frame.groupby("raw_file",sort=False):
      with np.load(out/filename) as data:
        for r in g.itertuples():
          if r.truth=="Gap":continue
          reject=data[r.raw_prefix+"__reject_"+r.method]
          for block,arr in enumerate(np.array_split(reject,4)):
            err=arr if r.truth=="Yes" else ~arr
            rows.append(dict(n=r.n,epsilon=r.epsilon,scenario=r.scenario,budget=r.budget,method=r.method,
                             truth=r.truth,seed_block=block,trials=len(arr),errors=int(err.sum()),error_rate=float(err.mean())))
    result=pd.DataFrame(rows);result.to_csv(out/"tables/seed_blocks.csv",index=False)
    return result


def aggregate(out=OUT):
    (out/"tables").mkdir(exist_ok=True)
    audit=verify(out)
    frames=read_all(out)
    frames=exact_lower_metrics(frames,out)
    eff=efficiency(frames["fixed"]);eff.to_csv(out/"tables/sample_complexity.csv",index=False)
    cost=cost_summary(frames);cost.to_csv(out/"tables/cost_summary.csv",index=False)
    paired=paired_costs(frames,out);paired.to_csv(out/"tables/paired_cost_comparisons.csv",index=False)
    seeds=seed_blocks(frames["fixed"],out)
    inventory=[]
    for name in SUITES:
      df=frames[name]
      inventory.append(dict(suite=name,method_cells=len(df),method_trial_evaluations=int(df.trials.sum()),
                            non_gap_method_cells=int((df.truth!="Gap").sum())))
    pd.DataFrame(inventory).to_csv(out/"tables/inventory.csv",index=False)
    certified=frames["fixed"].query("method in ['multi','single','fh_certified'] and truth == 'Yes'")
    actual=cost.query("suite == 'stream' and method == 'multi_hybrid'")
    structural=frames["structure"].query("method == 'multi' and truth == 'No' and sampling == 'natural_group_blocks'")
    summary=dict(inventory=inventory,total_method_trial_evaluations=sum(r["method_trial_evaluations"] for r in inventory),
                  certified_fixed_max_fp=float(certified.error_rate.max()),
                  actual_stream_hybrid=actual.to_dict(orient="records"),
                  sequential_hybrid=cost.query("suite == 'sequential' and method == 'multi_hybrid'").to_dict(orient="records"),
                  fixed_complete_grid_passes=eff.groupby("method").right_censored.apply(lambda x:int((~x).sum())).to_dict(),
                  structural_max_observed_fn=float(structural.error_rate.max()),
                  source_and_raw_audit=audit,
                  analysis_notes=["Scenario bootstrap describes sensitivity to the chosen finite scenario mix; not generalization to all distributions.",
                                  "Monte Carlo paired intervals use delta-method standard errors; Holm corrects the reported aggregate cost contrasts.",
                                  "Non-rejection is not a certified Yes for fixed-budget polynomial methods."])
    target_audit=json.loads((out/"effective_support_audit.json").read_text())
    summary["exact_effective_support_audit"]={k:v for k,v in target_audit.items() if k!="changes"}
    save(out/"summary.json",summary)
    return frames,eff,cost,paired,summary
