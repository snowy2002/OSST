"""Check the D guard where epsilon=1/(4n); frozen separately after primary inspection."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from benchmark_core import stable_rng
from c7016.populations import Population,sample_profiles
from parameter_search import algorithm as alg
from followup import core
from followup.protocol import ROOT,OUT,save,sha,csv_save


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT/'count_guard')
    args=parser.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    design=dict(ns=[442414,3269018],trials=2000,seed=202611027803,
                epsilon='1/(4*n)',scenarios=['uniform','rare_mass_2epsilon'],
                disclosure='Added after observing no failures of no_count on the primary grid. Separate mechanistic diagnostic, not prospectively included in the primary confidence family.',
                guarantee='Full C70 L14 is certified. Removing D is an ablation without a coverage theorem.',
                sampling='Grouped iid Poisson fingerprint; binary64 RNG. Huge means are simulated, not acquired observations.')
    paths=['followup/guard/run.py','followup/core.py','parameter_search/algorithm.py','c7016/populations.py']
    hashes={p:sha(ROOT/p) for p in paths}
    protocol=out/'protocol.json'
    if protocol.exists():
        previous=json.loads(protocol.read_text())
        if previous['design']!=design or previous['source_sha256']!=hashes:
            raise RuntimeError('Separate diagnostic protocol changed')
    else:
        save(protocol,dict(design=design,source_sha256=hashes,frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        for p in paths:
            target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    rows=[]
    for n in design['ns']:
        ep=Fraction(1,4*n);plan=alg.parameters('70',14,n,ep)
        populations={'uniform':Population((n,),(Fraction(1,n),)),
                     'rare_mass_2epsilon':Population((n-1,1),((1-2*ep)/(n-1),2*ep))}
        for name,pop in populations.items():
            fp,defect=sample_profiles(pop,float(plan.m),design['trials'],plan.degree,
                                     stable_rng(design['seed'],n,name))
            a,b=core.scale_intervals(fp,plan);raw=dict(D=fp.observed,low=fp.low)
            for mode in ['full','no_count','count']:
                events,ref=core.outcomes(fp,plan,pop,a,b,mode)
                raw.update({mode+'__'+k:v for k,v in events.items()})
                rows.append(dict(n=n,scenario=name,method=mode,epsilon=str(ep),mean=float(plan.m),
                    trials=design['trials'],effective=pop.effective(ep),S=pop.support,
                    binary_errors=int(events['binary'].sum()),coverage_errors=int((events['over']|events['short']).sum()),
                    mean_lower_lo=float(events['lower_lo'].mean()),mean_lower_hi=float(events['lower_hi'].mean()),
                    refinements=ref,pmf_defect=defect,raw_file=f'{n}_{name}.npz'))
            np.savez_compressed(out/f'{n}_{name}.npz',**raw)
    csv_save(out/'results.csv',rows)
    save(out/'complete.json',dict(complete=True,independent_paths=8000,method_evaluations=24000,
          raw_sha256={p.name:sha(p) for p in out.glob('*.npz')},scope=design['disclosure']))
    print('COUNT GUARD COMPLETE',rows,flush=True)


if __name__=='__main__':main()
