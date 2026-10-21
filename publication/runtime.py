"""Fresh-process compilation/scoring timings; absolute process peak RSS."""
from __future__ import annotations
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
import numpy as np

from benchmark_core import sample_profiles, stable_rng
from publication.core import compile_scales, signature, near_boundary, certified_reject
from publication.protocol import ROOT, OUT, save
from experiments import write_csv


def child(n,eps,family,config,trials):
    m=math.ceil(4*n/(eps*(1+math.log(n))))
    t=time.perf_counter()
    scales=compile_scales(n,str(eps),m,family,signature(config))
    compile_seconds=time.perf_counter()-t
    t=time.perf_counter()
    fp=sample_profiles(near_boundary(n,eps),m,trials,24,stable_rng(768512,n,eps))
    simulation_seconds=time.perf_counter()-t
    timings=[]
    for _ in range(9):
      t=time.perf_counter();result=certified_reject(fp,scales,n);timings.append(time.perf_counter()-t)
    return dict(n=n,epsilon=eps,method=family,trials=trials,compile_seconds=compile_seconds,
                simulation_seconds=simulation_seconds,score_batch_median_seconds=float(np.median(timings)),
                score_microseconds_per_profile=float(np.median(timings)*1e6/trials),
                score_batch_p90_seconds=float(np.quantile(timings,.9)),
                peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                active_scales=len(scales),degrees=";".join(str(s.certificate.degree) for s in scales),
                reject_rate=float(result.mean()))


def runtime_suite(configs,out=OUT,quick=False):
    rows=[]
    for n in [1000] if quick else [1000,10000,100000,1000000]:
      for eps in [.1] if quick else [.001,.1]:
        for family in ["multi","single","fh_certified"]:
          cmd=[sys.executable,"-m","publication.runtime","--n",str(n),"--epsilon",str(eps),
               "--family",family,"--config",json.dumps(configs[family]),"--trials",str(40 if quick else 2000)]
          proc=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True,
                              env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
          rows.append(json.loads(proc.stdout))
        print(f"runtime n={n} epsilon={eps}",flush=True)
    write_csv(out/"runtime.csv",rows)
    save(out/"runtime_complete.json",dict(rows=len(rows),scope="Fresh Python process per configuration; peak RSS includes interpreter, dependencies and sampler. No GPU; BLAS threads=1. Nine timed scoring batches; compilation separated."))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--n",type=int);p.add_argument("--epsilon",type=float)
    p.add_argument("--family");p.add_argument("--config");p.add_argument("--trials",type=int,default=2000)
    a=p.parse_args();print(json.dumps(child(a.n,a.epsilon,a.family,json.loads(a.config),a.trials)))
