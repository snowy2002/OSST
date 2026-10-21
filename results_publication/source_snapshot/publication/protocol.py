"""Prospective design, immutable stage manifests, and local evidence provenance."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "large" / "prepared"
OUT = ROOT / "results_publication"
SEEDS = {"development": 8726101901, "formal": 8726101902, "search": 8726101903,
         "adversarial_test": 8726101904, "stream": 8726101905, "structure": 8726101906,
         "sensitivity": 8726101907, "reference": 8726101908}
DESIGN = {
    "version": 2,
    "correctness_amendment": {
        "reason": "Exact rational audit found some nominal Delta_n=epsilon candidates below the promise boundary by floating-point roundoff (roughly 1e-20 to 1e-18).",
        "repair": "Certify the exact normalized rational tail of stored binary probability weights, moving the mixture upward by ulps if necessary; rerun every formal suite.",
        "original_output_preserved": "results_publication_superseded_boundary_v1",
        "selection": "Reuse the original independently selected configurations unchanged; no outcome-based tuning. The same keyed seeds enable paired correction.",
        "disclosure": "Some v1 sequential/stream aggregate outcomes were inspected before this correctness amendment. This is not a claim of an untouched externally preregistered experiment."
    },
    "primary_question": "When do certified polynomial tests save samples relative to competitive count and validation baselines?",
    "fixed": {"sizes": [1000,10000,100000,1000000],
              "epsilons": [.001,.005,.01,.02,.05,.1,.2],
              "factors": [.5,1,1.5,2,2.5,3,3.5,4,5,6,8,12,16], "trials": 2000, "seed_blocks": 4,
              "methods": ["count","multi","single","fh_certified","fh_native_tuned","chao1_point"]},
    "development": {"sizes": [1500,15000], "epsilons": [.005,.05,.2],
                    "factors": [1,2,4,8], "trials": 400, "candidates_per_family": 8},
    "sequential": {"sizes": [1000,10000,100000,1000000], "epsilons": [.001,.01,.1], "trials": 2000},
    "stream": {"sizes": [50,200,1000], "epsilons": [.01,.1], "trials": 1000,
               "scenarios": ["uniform_null","head90_null","uniform_boundary","exact_diffuse_boundary","near_multilevel_boundary"]},
    "adversarial": {"sizes": [1000,10000], "epsilons": [.001,.01,.1], "search_trials": 300,
                    "test_trials": 10000, "sequential_test_trials": 5000, "selected_per_problem": 6},
    "structure": {"datasets": ["nips","enron","movielens","retail","bci","fish_2007_seine"],
                  "block_sizes": [1,5,20,100], "stress_trials": 500, "iid_trials": 2000,
                  "holdout_splits": 3, "holdout_epsilon": .1},
    "sensitivity": {"sizes": [1000,100000], "epsilons": [.01,.1], "deltas": [.1,.05,.01,.001],
                    "trials": 2000, "small_delta_trials": 20000},
    "reference": {"sizes": [100,1000], "epsilons": [.01,.1], "trials": 1000,
                  "note": "Strict Wang Poisson branch and FH fixed-sample small-epsilon branch; conservative amplification for 5% risk."},
    "statistical_analysis": {
        "errors": "Exact one-sided Clopper-Pearson; 95% simultaneous bounds Bonferroni within each frozen suite across all non-Gap method cells. Fixed-suite lower-bound failure metrics use a separate simultaneous family.",
        "cost": "Equal-weight cell means, Yes/No and synthetic/empirical strata reported separately. CRN paired differences with descriptive 95% normal intervals and bootstrap aggregate intervals.",
        "m_star": "Smallest tested fixed budget for which all promised synthetic scenarios for the method/n/epsilon pass the simultaneous error upper bound. Report right censoring and do not interpolate as proven power.",
        "search": "All adversarial search scores retained; selected populations frozen before independent new-seed test. No formal outcomes used for parameter changes.",
        "gap": "No error labels for S>n and Delta_n<epsilon.",
        "scope": "Empirical tests are not an optimal-order proof; FH-shaped practical variants are not original FH theorem algorithms; reference empirical supports are not external population truth."
    },
    "external_limitations": ["No new ecological field survey or independently established full-lake species inventory is available.",
                             "Prospective local timestamp/hash manifest is not an external preregistration.",
                             "All new designs were informed by previously inspected v1 results; only this follow-up's outcomes are held out."],
    "seeds": SEEDS,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def freeze_design(out=OUT):
    out.mkdir(exist_ok=True)
    path = out/"design.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["design"] != DESIGN:
            raise RuntimeError("Frozen design differs; use a new output directory and disclose amendment")
        return old
    value = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "design": DESIGN}
    save(path, value)
    return value


def freeze_evaluation(configs, out=OUT):
    freeze_design(out)
    truth_path=out/"population_truth.json"
    if not truth_path.exists():
        from fractions import Fraction
        from publication.core import formal_cases,exact_tail_mass
        truth_rows=[]
        for n in DESIGN["fixed"]["sizes"]:
            for eps in DESIGN["fixed"]["epsilons"]:
                for name,pop in formal_cases(n,eps):
                    tail=exact_tail_mass(pop,n)
                    truth="Yes" if pop.support<=n else "No" if tail>=Fraction(str(eps)) else "Gap"
                    if truth!=pop.truth(n,eps):raise ArithmeticError("Exact/float truth disagreement")
                    truth_rows.append(dict(n=n,epsilon=eps,scenario=name,S=pop.support,truth=truth,
                                           exact_normalized_tail=str(tail),exact_margin=str(tail-Fraction(str(eps)))))
        save(truth_path,truth_rows)
    paths = [*[p for p in sorted((ROOT/"publication").glob("*.py")) if p.name not in {"report.py","aggregate.py"}],
             *sorted(ROOT.glob("*.py")),ROOT/"requirements-lock.txt"]
    payload = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "design_sha256": sha(out/"design.json"),
               "population_truth_sha256":sha(truth_path),
               "calibration_sha256": sha(out/"calibration.json"), "configs": configs,
               "source_hashes": {str(p.relative_to(ROOT)):sha(p) for p in paths},
               "data_hashes": {p.name:sha(p) for p in sorted(DATA.glob("*.npz"))},
               "environment": {"python":sys.version,"platform":platform.platform(),"cpu_count":os.cpu_count(),
                               "machine":platform.machine(),"blas_threads":os.environ.get("OPENBLAS_NUM_THREADS")}}
    path=out/"protocol.json"
    if path.exists():
        old=json.loads(path.read_text())
        for k in ["design_sha256","population_truth_sha256","calibration_sha256","configs","source_hashes","data_hashes"]:
            if old[k]!=payload[k]:raise RuntimeError(f"Frozen protocol mismatch: {k}")
        return old
    save(path,payload)
    for p in paths:
        target=out/"source_snapshot"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    return payload
