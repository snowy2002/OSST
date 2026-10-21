"""Exact labelled-stream costs with immediate discovery stopping.

All methods share iid training labels and independent look-specific validation
labels. Poisson increment sizes are independent of labels. Validation is never
reused for fitting. Costs are actual label counts, not Poisson means.
"""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import math
import time

import numpy as np
from flint import ctx

from benchmark_core import Profiles, Population, stable_rng
from guaranteed import ball
from guaranteed_experiments import batch_no
from support_size import count_plan
from publication.core import formal_cases, trial_summary
from publication.suites import seq_plans, finish
from publication.protocol import DESIGN, OUT, SEEDS


@lru_cache(maxsize=128)
def fixed_cap(n,eps,delta=.05):
    ep,de=Fraction(str(eps)),Fraction(str(delta))
    start=count_plan(n,float(ep),float(de),sampling="fixed")["m"]
    with ctx.workprec(256):
      for N in range(start,start+50):
        bound=(1-ball(ep)).beta_lower(N-n,n+1,regularized=1)
        if bound<=ball(de):return N
    raise ArithmeticError("Cannot certify fixed count budget")


def grouped_labels(pop,size,rng):
    """Exact categorical labels for a grouped population without expanding p."""
    mass=pop.multiplicities*pop.probabilities
    groups=rng.choice(len(mass),size=size,p=mass/mass.sum())
    offsets=np.r_[0,np.cumsum(pop.multiplicities)[:-1]]
    # integers accepts broadcast high bounds, avoiding float-to-label conversion.
    return offsets[groups]+rng.integers(pop.multiplicities[groups])


def simulate_stream_case(pop,n,eps,trials,configs,key,seed=SEEDS["stream"]):
    plans=seq_plans(n,eps,configs)
    plans.pop("count_batch")
    cap=next(iter(plans.values())).baseline_m
    fcap=fixed_cap(n,eps)
    times=sorted({cap,*[m for p in plans.values() for m in [p.final_m,*[l.m for l in p.looks]]]})
    K=max(p.max_degree for p in plans.values())
    lookups={method:{l.m:l for l in p.looks} for method,p in plans.items()}
    methods=["count_fixed_immediate","count_poisson_immediate","count_checkpoint",*plans]
    outputs={method:dict(reject=np.zeros(trials,bool),cost=np.zeros(trials,np.int64),
                         validation=np.zeros(trials,np.int64),reason=np.full(trials,"",dtype="U16"),
                         scale=np.full(trials,-999,np.int64)) for method in methods}
    for t in range(trials):
      rng=stable_rng(seed,"training",*key,t)
      rng_size=stable_rng(seed,"poisson_sizes",*key,t)
      counts=np.zeros(pop.support,np.int64)
      low=np.zeros(K,np.int64)
      observed,consumed,previous=0,0,0
      first_hit=None
      active={method:True for method in methods}
      spent={method:0 for method in methods}
      def stop(method,reject,training,reason,j=-999):
        o=outputs[method]
        o["reject"][t]=reject;o["cost"][t]=training+spent[method]
        o["validation"][t]=spent[method];o["reason"][t]=reason;o["scale"][t]=j
        active[method]=False
      for m in times:
        size=int(rng_size.poisson(m-previous))
        labels=grouped_labels(pop,size,rng)
        ids,first,freq=np.unique(labels,return_index=True,return_counts=True)
        old=counts[ids]
        newmask=old==0
        if first_hit is None and observed+int(newmask.sum())>n:
          needed=n+1-observed
          first_hit=consumed+int(np.partition(first[newmask],needed-1)[needed-1])+1
        low-=np.bincount(np.minimum(old,K+1),minlength=K+2)[1:K+1]
        counts[ids]+=freq
        low+=np.bincount(np.minimum(counts[ids],K+1),minlength=K+2)[1:K+1]
        observed+=int(newmask.sum())
        consumed+=size;previous=m
        fp=Profiles(np.array([observed]),low[None,:])
        if active["count_fixed_immediate"]:
          if first_hit is not None and first_hit<=fcap:stop("count_fixed_immediate",True,first_hit,"observed")
          elif consumed>=fcap:stop("count_fixed_immediate",False,fcap,"fallback")
        if active["count_poisson_immediate"]:
          if first_hit is not None and m<=cap:stop("count_poisson_immediate",True,first_hit,"observed")
          elif m==cap:stop("count_poisson_immediate",False,consumed,"fallback")
        checkpoint_times={l.m for l in next(iter(plans.values())).looks}|{cap}
        if active["count_checkpoint"] and m in checkpoint_times:
          if observed>n:stop("count_checkpoint",True,consumed,"observed")
          elif m==cap:stop("count_checkpoint",False,consumed,"fallback")
        maxr=max([0]+[lookups[method][m].validation_r for method in plans if active[method] and m in lookups[method]])
        validation_labels=(grouped_labels(pop,maxr,stable_rng(seed,"validation",*key,t,m)) if maxr else None)
        for method,p in plans.items():
          if not active[method]:continue
          if m not in lookups[method] and m!=p.final_m:continue
          if first_hit is not None:
            stop(method,True,first_hit,"observed");continue
          if m==p.final_m:
            stop(method,False,consumed,"fallback");continue
          look=lookups[method][m]
          winning=None
          for c in look.certificates:
            if c.rejects(observed,{l+1:int(f) for l,f in enumerate(low[:c.degree])}):
              winning=c.j;break
          if winning is not None:
            stop(method,True,consumed,"polynomial",winning);continue
          gate_limit=int(look.m*p.epsilon*Fraction(str(p.config["validation_gate"])))
          if look.validation_r and low[0]<=gate_limit:
            spent[method]+=look.validation_r
            outside=int(np.count_nonzero(counts[validation_labels[:look.validation_r]]==0))
            if outside<=look.validation_k:stop(method,False,consumed,"validation")
        if not any(active.values()):break
      # A small realization of the largest Poisson count can be shorter than the
      # fixed baseline budget. Extend that independent training sequence once.
      if active["count_fixed_immediate"]:
        labels=grouped_labels(pop,fcap-consumed,rng)
        ids,first=np.unique(labels,return_index=True)
        unseen=first[counts[ids]==0]
        need=n+1-observed
        if len(unseen)>=need:
          stop("count_fixed_immediate",True,consumed+int(np.partition(unseen,need-1)[need-1])+1,"observed")
        else:stop("count_fixed_immediate",False,fcap,"fallback")
      if any(active.values()):raise AssertionError(active)
    rows=[]
    truth=pop.truth(n,eps)
    base=outputs["count_fixed_immediate"]["cost"]
    for method,o in outputs.items():
      row=dict(method=method,n=n,epsilon=eps,truth=truth,S=pop.support,delta_n=pop.tail_mass(n),
               cost_metric="actual_labels_including_validation",**trial_summary(o["reject"],o["cost"],truth,base),
               mean_validation=float(o["validation"].mean()))
      row.update({f"{reason}_rate":float(np.mean(o["reason"]==reason)) for reason in ["observed","polynomial","validation","fallback"]})
      rows.append(row)
    return rows,outputs


def stream_suite(configs,out=OUT,quick=False):
    d=DESIGN["stream"]
    rows=[]
    for n in d["sizes"][:1] if quick else d["sizes"]:
      for eps in d["epsilons"][:1] if quick else d["epsilons"]:
        for scenario,pop in formal_cases(n,eps):
          if scenario not in d["scenarios"]:continue
          start=time.perf_counter()
          rr,outputs=simulate_stream_case(pop,n,eps,20 if quick else d["trials"],configs,(n,eps,scenario))
          name=f"stream_n{n}_e{eps:g}_{scenario}.npz"
          np.savez_compressed(out/"raw"/name,**{method+"__"+k:v for method,o in outputs.items() for k,v in o.items()})
          rows.extend(dict(scenario=scenario,raw_file="raw/"+name,**r) for r in rr)
          print(f"stream n={n} eps={eps} {scenario}: {time.perf_counter()-start:.1f}s",flush=True)
    finish(out,"stream",rows)
    return rows
