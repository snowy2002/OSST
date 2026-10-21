"""Natural-group dependence, sampling-design repair, and group-held-out spectra."""
from __future__ import annotations
import csv
import math

import numpy as np
from scipy import sparse

from benchmark_core import Population, Profiles, stable_rng
from large_experiments import clustered_profiles
from publication.protocol import ROOT, DATA, OUT, DESIGN, SEEDS, save
from publication.suites import fixed_cell, sample_blocks, finish, K


def load_matrix(dataset):
    if dataset!="fish_2007_seine":return sparse.load_npz(DATA/f"{dataset}_groups.npz").tocsr()
    records=[r for r in csv.DictReader((ROOT/"data/inch_lake_fish.csv").open()) if r["year"]=="2007" and r["gear"]=="seine"]
    groups={s:i for i,s in enumerate(sorted({r["event_id"] for r in records}))}
    species={s:i for i,s in enumerate(sorted({r["species"] for r in records}))}
    return sparse.coo_matrix((np.ones(len(records),np.int64),
                  ([groups[r["event_id"]] for r in records],[species[r["species"]] for r in records])),
                  shape=(len(groups),len(species))).tocsr()


def empirical(matrix,equal_groups=False):
    if equal_groups:
        totals=np.asarray(matrix.sum(axis=1)).ravel()
        matrix=matrix[totals>0]
        probs=np.asarray(matrix.multiply(1/totals[totals>0,None]).mean(axis=0)).ravel()
        probs=probs[probs>0]
        return Population(np.ones(len(probs),np.int64),probs/probs.sum())
    return Population.from_counts(np.asarray(matrix.sum(axis=0)).ravel())


def structure_suite(configs,out=OUT,quick=False):
    d=DESIGN["structure"]
    rows,holdout,manifest=[],[],[]
    datasets=d["datasets"][:1] if quick else d["datasets"]
    for dataset in datasets:
      matrix=load_matrix(dataset)
      pop=empirical(matrix)
      eventpop=empirical(matrix,True)
      eps=.1
      T=20 if quick else d["stress_trials"]
      for scenario,n in [("boundary",max(1,pop.effective_support(eps)-1)),("null",pop.support)]:
        m=math.ceil(4*n/(eps*(1+math.log(n))))
        for block in d["block_sizes"][:2] if quick else d["block_sizes"]:
          key=("structure",dataset,scenario,block)
          if block==1:
            fp=sample_blocks(pop,m,T,SEEDS["structure"],*key)
            actual=None
          else:
            fp,actual=clustered_profiles(matrix,m,block,T,stable_rng(SEEDS["structure"],*key))
          rr,raw,_=fixed_cell(pop,n,eps,m,T,configs,key,SEEDS["structure"],lower=False,fp=fp)
          name=f"structure_{dataset}_{scenario}_b{block}.npz"
          np.savez_compressed(out/"raw"/name,**raw)
          rows.extend(dict(dataset=dataset,sampling="iid_record" if block==1 else "natural_group_blocks",
                           scenario=scenario,block_size=block,iid_assumptions=block==1,
                           mean_actual_M=actual,raw_file="raw/"+name,**r) for r in rr)
      # Choosing a group independently with probability proportional to size and
      # one within-group record is EXACTLY the original iid record distribution.
      # Equal-probability groups define a different, explicitly recorded target.
      for sampling,current in [("repaired_size_weighted_one_per_draw",pop),("repaired_equal_event_one_per_draw",eventpop)]:
        for scenario,n in [("boundary",max(1,current.effective_support(eps)-1)),("null",current.support)]:
          m=math.ceil(4*n/(eps*(1+math.log(n))))
          T=40 if quick else d["iid_trials"]
          rr,raw,_=fixed_cell(current,n,eps,m,T,configs,(dataset,sampling,scenario),SEEDS["structure"],lower=False)
          name=f"structure_{dataset}_{sampling}_{scenario}.npz"
          np.savez_compressed(out/"raw"/name,**raw)
          rows.extend(dict(dataset=dataset,sampling=sampling,scenario=scenario,block_size=1,iid_assumptions=True,
                           mean_actual_M=None,raw_file="raw/"+name,**r) for r in rr)
      for split in range(1 if quick else d["holdout_splits"]):
        rng=stable_rng(SEEDS["structure"],"group_split",dataset,split)
        order=rng.permutation(matrix.shape[0]);half=len(order)//2
        train,test=order[:half],order[half:]
        ptrain,ptest=empirical(matrix[train]),empirical(matrix[test])
        name=f"holdout_{dataset}_split{split}.npz"
        raw={"train_groups":train,"test_groups":test}
        for scenario,n in [("train_effective_boundary",max(1,ptrain.effective_support(eps)-1)),("train_support",ptrain.support)]:
          for C in [4,8]:
            m=math.ceil(C*n/(eps*(1+math.log(n))))
            T=40 if quick else d["iid_trials"]
            rr,arrays,_=fixed_cell(ptest,n,eps,m,T,configs,("holdout",dataset,split,scenario,C),SEEDS["structure"])
            raw.update({f"{scenario}_C{C}__{k}":v for k,v in arrays.items()})
            holdout.extend(dict(dataset=dataset,split=split,scenario=scenario,C=C,train_groups=len(train),
                                test_groups=len(test),train_support=ptrain.support,raw_file="raw/"+name,
                                evaluation_scope="iid resampling of unseen-group empirical spectrum",**r) for r in rr)
        np.savez_compressed(out/"raw"/name,**raw)
      manifest.append(dict(dataset=dataset,groups=matrix.shape[0],records=int(matrix.sum()),support=pop.support,
                           record_effective_support=pop.effective_support(eps),equal_event_effective_support=eventpop.effective_support(eps),
                           external_population_truth=False))
      print(f"structure and held-out groups: {dataset} complete",flush=True)
    finish(out,"structure",rows)
    finish(out,"group_holdout",holdout)
    save(out/"structure_targets.json",manifest)
