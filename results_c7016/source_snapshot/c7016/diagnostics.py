"""Supplementary full-fingerprint integration and isolated kernel timings.

These deterministic coverage choices are disclosed as post-primary numerical
checks, not additional confirmatory statistical risk trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from benchmark_core import Profiles, stable_rng
from c1025.data import ROOT, save, sha
from c7016.algorithm import (parameters, test_fingerprint, score_intervals,
                              geometry, compiled, compare_scores)
from c7016.populations import synthetic_cases, sample_profiles
from c7016.run import csv_save
from support_size import Fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results_c7016')
    args = parser.parse_args(argv); out = args.out
    design = dict(scope=__doc__, sizes=[3269017, 3269018, 10**7], epsilon='0.1',
                  scenarios=['uniform_null', 'uniform_boundary', 'epsilon_scaled_rare_1/10_null'],
                  replicates=2, timing_sizes=[1000, 3269018, 10**9, 10**12],
                  timing_trials=1000, timing_repeats=5, seed=70168822,
                  source_sha256=sha(Path(__file__)))
    metadata = dict(design=design, frozen_at=datetime.now(timezone.utc).isoformat())
    protocol = out/'diagnostics_protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())['design'] != design:
            raise RuntimeError('Supplement protocol changed')
    else:
        save(protocol, metadata)
    results = []
    for n in design['sizes']:
        for name, pop in synthetic_cases(n, design['epsilon']):
            if name not in design['scenarios']:
                continue
            plan = parameters(n, design['epsilon'])
            for rep in range(design['replicates']):
                rng = stable_rng(design['seed'], n, name, rep)
                frequencies = Counter(); M = 0
                start = time.perf_counter()
                for g, p in zip(pop.multiplicities, pop.probabilities):
                    counts = rng.poisson(float(plan.m)*float(p), size=g)
                    M += int(counts.sum())
                    values, repeats = np.unique(counts[counts > 0], return_counts=True)
                    frequencies.update({int(v): int(r) for v, r in zip(values, repeats)})
                sampling_seconds = time.perf_counter()-start
                fp = Fingerprint(frequencies)
                if fp.total != M:
                    raise ArithmeticError('Explicit sample total mismatch')
                start = time.perf_counter(); result = test_fingerprint(fp, n, design['epsilon'])
                inference_seconds = time.perf_counter()-start
                profiles = Profiles(np.array([fp.observed], dtype=np.int64),
                         np.array([[fp.frequencies.get(k, 0) for k in range(1, plan.degree+1)]], dtype=np.int64))
                lo, hi, _, _ = score_intervals(profiles, plan)
                reject, _ = compare_scores(profiles, n, lo, hi, n)
                if bool(reject[0]) != (result['decision'] == 'No'):
                    raise ArithmeticError('Full public API and batch decision disagree')
                lower = result['support_lower_bound']
                if not max(1, min(n, np.ceil(lo[0]))) <= lower <= max(1, min(n, np.ceil(hi[0]))):
                    raise ArithmeticError('Exact public output outside batch enclosure')
                label, tail, eff = pop.truth(n, design['epsilon'])
                path = f'raw/full_fingerprint_{n}_{name.replace("/", "_")}_{rep}.json'
                save(out/path, {str(k): v for k, v in frequencies.items()})
                results.append(dict(scenario=name, rep=rep, truth=label, S=pop.support,
                      effective_support=eff, sampling_seconds=sampling_seconds,
                      inference_seconds=inference_seconds, fingerprint_file=path, **result))
            print('full public fingerprint', n, name, flush=True)
    save(out/'full_polynomial_api_runs.json', results)
    timings = []
    for n in design['timing_sizes']:
        geometry.cache_clear(); compiled.cache_clear()
        start = time.perf_counter(); plan = parameters(n, '.1'); compiled(n, plan.bits)
        compile_seconds = time.perf_counter()-start
        pop = dict(synthetic_cases(n, '.1'))['epsilon_scaled_rare_1/10_null']
        start = time.perf_counter()
        fp, _ = sample_profiles(pop, float(plan.m), design['timing_trials'], plan.degree,
                               stable_rng(design['seed'], n, 'timing'))
        sampling_seconds = time.perf_counter()-start
        score_intervals(fp, plan)
        measurements = []
        for _ in range(design['timing_repeats']):
            start = time.perf_counter(); score_intervals(fp, plan)
            measurements.append(time.perf_counter()-start)
        timings.append(dict(n=n, branch=plan.branch, degree=plan.degree, scales=len(plan.scales),
              trials=design['timing_trials'], compile_seconds=compile_seconds,
              grouped_sampling_seconds=sampling_seconds,
              scoring_seconds_median=float(np.median(measurements)),
              scoring_seconds_min=min(measurements), scoring_seconds_max=max(measurements),
              scope='Local compute only, excludes physical sample collection and file writing'))
    csv_save(out/'runtime.csv', timings)
    save(out/'diagnostics_complete.json', dict(full_api_calls=len(results),
          explicit_category_poisson_draws=sum(r['S'] for r in results),
          polynomial_api_calls=sum(r['branch'] == 'chebyshev' for r in results), timing_cells=len(timings)))


if __name__ == '__main__':
    main()
