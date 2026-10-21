"""Confidence levels and one-at-a-time robustness of a frozen configuration."""
from __future__ import annotations
import numpy as np
from publication.core import formal_cases
from publication.suites import sequential_cell, finish
from publication.protocol import OUT, DESIGN, SEEDS


def sensitivity_suite(configs,out=OUT,quick=False):
    d=DESIGN["sensitivity"]
    rows=[]
    scenarios=["uniform_null","head90_null","uniform_boundary","exact_diffuse_boundary"]
    for n in d["sizes"][:1] if quick else d["sizes"]:
      for eps in d["epsilons"][:1] if quick else d["epsilons"]:
        for delta in d["deltas"][:1] if quick else d["deltas"]:
          T=40 if quick else (d["small_delta_trials"] if delta==.001 else d["trials"])
          raw={}
          name=f"risk_n{n}_e{eps:g}_d{delta:g}.npz"
          for scenario,pop in formal_cases(n,eps):
            if scenario not in scenarios:continue
            rr,outputs,_=sequential_cell(pop,n,eps,T,configs,("risk",n,eps,delta,scenario),SEEDS["sensitivity"],delta)
            rows.extend(dict(variant="delta",scenario=scenario,raw_file="raw/"+name,**r) for r in rr)
            raw.update({f"{scenario}__{method}__{k}":v for method,o in outputs.items() for k,v in o.items()})
          np.savez_compressed(out/"raw"/name,**raw)
          print(f"risk sensitivity n={n} eps={eps} delta={delta}",flush=True)
    variations=[("frozen",{})]+[(f"{key}={value}",{key:value}) for key,values in [
                ("b_factor",[.25,1.]),("max_degree",[6,20]),("weight_cap",[10.,40.]),
                ("validation_factor",[4.,24.]),("validation_gate",[.5,1.]),("density",[2,4])]
                for value in values]
    for label,changes in variations[:2] if quick else variations:
      updated={fam:{**cfg,**changes} for fam,cfg in configs.items()}
      n,eps=10000,.1
      arrays={}
      name=f"sensitivity_{label.replace('=','_')}.npz"
      for scenario,pop in formal_cases(n,eps):
        if scenario not in scenarios:continue
        rr,outputs,_=sequential_cell(pop,n,eps,40 if quick else d["trials"],updated,
                                    ("one_at_a_time",scenario),SEEDS["sensitivity"])
        rows.extend(dict(variant=label,scenario=scenario,raw_file="raw/"+name,**r) for r in rr)
        arrays.update({f"{scenario}__{method}__{k}":v for method,o in outputs.items() for k,v in o.items()})
      np.savez_compressed(out/"raw"/name,**arrays)
      print(f"parameter sensitivity {label}",flush=True)
    finish(out,"sensitivity",rows)
