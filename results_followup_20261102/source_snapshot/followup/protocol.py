"""Immutable design and provenance; informed by earlier results, not preregistration."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results_followup_20261102'
SCENARIOS = ['uniform_null', 'geometric_blocks_null', 'epsilon_scaled_rare_1/10_null',
             'epsilon_scaled_rare_1/2_null', 'uniform_boundary', 'diffuse_boundary',
             'multiscale_boundary', 'gap_near']
DESIGN = dict(
    version=1, seed=20261102173801, blocks=4,
    mechanism=dict(ns=[442413,442414,1000000,3269017,3269018,10**9],
                   epsilons=['.001','.01','.1','.25'], trials=4000, scenarios=SCENARIOS,
                   methods=['full14','full16','single14','count70','no_count14']),
    budget=dict(ns=[1000,10**6,10**9], epsilons=['.001','.01','.1'], trials=4000,
                constants=['4','8','16','32','60','69','70'], scenarios=SCENARIOS,
                fixed_stress_reference_C=60, fixed_stress_lambdas=['.5','2','8','32'],
                target_risk='.01', counting_risks=['.01','.0009']),
    numeric=dict(refine_all_archived_70_14_ambiguous=True, precisions=[512,1024,2048,4096,8192],
                 independent_formula_ns=[442414,10**6,10**9,10**12],
                 explicit_count_ns=[442414,10**6], scenarios=['uniform_null','epsilon_scaled_rare_1/10_null'],
                 epsilon='.1', explicit_replicates=2,
                 label_n=442414,label_replicates=2,
                 runtime_ns=[1000,442414,10**6,3269018,10**9,10**12],
                 runtime_batch=1000,runtime_repeats=7),
    inference=dict(risk='One-sided exact binomial, Bonferroni within each suite over method rows; joint binary/coverage event. Gap has coverage only.',
                   gain='Two-sided Hoeffding on paired normalized gains in [0,1] (nested maxima), simultaneous within the mechanism comparison family.',
                   budget='First passing grid budget is only a finite-family empirical result. All budgets included in multiplicity correction; no monotonicity assumed.',
                   cost='Expected oracle sample counts in budget table; actual labels/M and compute stages separately in integration benchmarks.'),
    scope=dict(sampling='Independent Poisson category counts; grouped simulation is exact in real arithmetic, NumPy/SciPy sampling is binary64.',
               pairing='Mechanism variants and same-budget count share fingerprints. Distinct budgets have independently keyed streams.',
               parameters='Only C70 L14/L16 inherit the new theorem; C<70 and ablations are explicitly exploratory.',
               timing='Single-thread local CPU computation, no physical collection time. Method order randomized per repeat; compile, score and memory separated.',
               prior_results='Design informed by historical reports. Fresh random seeds; not external preregistration. Archived refinement is deterministic reanalysis, not new trials.',
               no_field_claim='No new natural population survey or unknown-support ground truth. No FH/BGY empirical performance claim.'),
)


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_save(path, rows):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)


def design(smoke=False):
    d=json.loads(json.dumps(DESIGN))
    if smoke:
        d['mechanism'].update(ns=[442414],epsilons=['.1'],trials=24)
        d['budget'].update(ns=[10**6],epsilons=['.1'],trials=24,constants=['8','70'])
        d['numeric'].update(independent_formula_ns=[442414],explicit_count_ns=[442414],
                            explicit_replicates=1,label_replicates=1,runtime_ns=[442414],
                            runtime_batch=24,runtime_repeats=2,archive_limit=20)
        d['smoke']=True
    return d


def freeze(out,d):
    for sub in ['raw','cells','source_snapshot','figures']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    sources=[*ROOT.glob('*.py')]
    for package in ['followup','parameter_search','c7016','c1025','publication']:
        sources.extend((ROOT/package).glob('*.py'))
    sources=[p for p in sources if not (p.parent.name=='followup' and p.name in ['report.py','audit.py'])]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
    historical={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'results_parameter_search/manifest.json',ROOT/'results_parameter_search/certificate.json',
        ROOT/'results_parameter_search/experiment_protocol.json']}
    path=out/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['design']!=d or old['source_sha256']!=hashes or old['historical_sha256']!=historical:
            raise RuntimeError('Frozen inputs changed; use a new output directory and disclose revision')
        return
    import numpy, scipy, flint
    save(path,dict(design=d,source_sha256=hashes,historical_sha256=historical,
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,
                                   scipy=scipy.__version__,flint=flint.__version__,
                                   OPENBLAS_NUM_THREADS=os.getenv('OPENBLAS_NUM_THREADS'),
                                   OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS'))))
    for p in sources:
        target=out/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
