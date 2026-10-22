"""Frozen, resumable experiments for C0=70, L0=16.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m c7016.run.
All iid suites use fixed repetitions. Different budget methods have independent
streams; the full statistic and its ablations share the same fingerprints.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from flint import arb, ctx
from scipy import sparse
from scipy.stats import beta, poisson

from benchmark_core import Profiles, stable_rng
from c1025.data import DATA, ROOT, load, save, sha
from c7016.algorithm import (FAILURE, arb_score, ball, compare_scores, parameters,
                              score_intervals, test_fingerprint)
from c7016.populations import Population, sample_profiles, synthetic_cases
from paper_reference import certified_integer, reference_parameters
from support_size import Fingerprint

DESIGN = dict(
    version=1, C0=70, L0=16, mean='70*n/(epsilon*(1+ln(n))); real, not ceil',
    sizes=[1, 10, 1000, 100000, 1000000, 3269017, 3269018, 3269019, 10**7, 10**9, 10**12],
    epsilons=['0.001', '0.01', '0.1', '0.25'], seed_blocks=4, trials_per_block=1000,
    natural_epsilons=['0.001', '0.01', '0.05', '0.1', '0.25'],
    natural_n_rules=['floor(S/2)', 'eff_epsilon-1', 'eff_epsilon', 'S', 'ceil(1.5S)'],
    lifted_target_supports=[10**7, 10**9], lifted_epsilons=['0.01', '0.1'],
    lifted_trials_per_block=500, lifted_n_rules=['eff_epsilon-1', 'eff_epsilon', 'S'],
    baselines=['count_matched_confidence', 'count_99', 'C1025_L1024', 'original'],
    ablations=['count_same_budget', 'scale0_same_budget'],
    amplification_repeats=27, amplification_trials_per_block=250,
    amplification_sizes=[1000, 3269018, 10**9], amplification_epsilons=['0.01', '0.1'],
    amplification_scenarios=['uniform_null', 'epsilon_scaled_rare_1/2_null',
                             'diffuse_boundary', 'multiscale_boundary', 'gap_near'],
    finite_file_trials=300, finite_file_epsilons=['0.01', '0.1', '0.25'],
    group_datasets=['mite', 'bci', 'fish_2007_seine', 'fish_2008_seine'],
    group_splits=5, group_trials=2000, duplication_factors=[1, 10, 100, 1000],
    numerical_audit_profiles_per_block=2,
    seeds=dict(synthetic=70162201, natural=70162202, lifted=70162203,
               amplification=70162204, finite=70162205, groups=70162206),
    statistics='One-sided Clopper-Pearson 95%, Bonferroni across all non-Gap method cells '
               'within each iid suite; a separate family for support-bound failure. '
               'Four independent seed blocks. No error-dependent stopping.',
    truth='Exact rational probabilities and exact tail/effective-support classification; '
          'Gap inputs have no binary error label.',
    sampling='Grouped independent Poisson-category counts via multinomial fingerprints. '
             'Exact law in real arithmetic, binary64 probability evaluation and PRNG. '
             'M is not inferred from a coarsened high-count bucket.',
    numeric='Arb parameter/weight enclosures, outward binary64 statistics with dot-product '
            'roundoff bounds, Arb refinement for unresolved event thresholds. '
            'Full arbitrary-precision spot checks are independent of the batch arithmetic.',
    natural_scope='Iid resampling from published finite empirical populations; natural '
                  'field sampling is not assumed iid. Cloned empirical shapes are explicitly synthetic.',
    disclosure='Locally frozen follow-up informed by prior proof/experiments, not external preregistration. '
               'Natural groups/files are descriptive stress tests, not validation of unseen ecosystem truth.',
)


def csv_save(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def cp_upper(errors, trials, alpha=.05):
    return 1. if errors == trials else float(beta.isf(alpha, errors+1, trials-errors))


def smoke_design():
    return dict(DESIGN, sizes=[1, 3269017, 3269018, 10**12], epsilons=['0.1'],
                seed_blocks=2, trials_per_block=12, natural_epsilons=['0.1'],
                lifted_target_supports=[10**7], lifted_epsilons=['0.1'],
                lifted_trials_per_block=8, amplification_sizes=[3269018],
                amplification_epsilons=['0.1'], amplification_trials_per_block=3,
                finite_file_trials=8, group_trials=12, group_splits=1)


def freeze(out, design):
    out.mkdir(parents=True, exist_ok=True)
    for name in ['raw', 'cells', 'figures']:
        (out/name).mkdir(exist_ok=True)
    if not (DATA/'datasets.json').exists():
        from c1025.data import prepare
        prepare()
    datasets = json.loads((DATA/'datasets.json').read_text())
    for spec in datasets:
        if sha(DATA/'prepared'/f'{spec["dataset"]}.npz') != spec['sha256']:
            raise RuntimeError('Prepared natural data hash changed')
    files = [ROOT/p for p in ['paper_reference.py', 'support_size.py', 'benchmark_core.py',
             'guaranteed.py', 'guaranteed_experiments.py', 'publication/core.py',
             'c1025/core.py', 'c1025/data.py', 'test_c7016.py',
             'c7016/__init__.py', 'c7016/algorithm.py', 'c7016/populations.py', 'c7016/run.py']]
    proof = ROOT/'artifacts/joint_constants_submission'
    files += [proof/name for name in ['submission_proof.tex', 'arithmetic_appendix.tex',
                                      'exact_certificate.py', 'rational_interval.py',
                                      'audit_exact.py', 'exact_certificate.json', 'audit_exact.json']]
    metadata = dict(design=design,
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
                    data_hashes={p.name: sha(p) for p in sorted((DATA/'prepared').glob('*.npz'))})
    path = out/'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key, value in metadata.items():
            if previous[key] != value:
                raise RuntimeError(f'Frozen {key} changed; use a new output directory')
        return
    metadata.update(frozen_at=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=sys.version, platform=platform.platform(),
                                     cpu_count=os.cpu_count(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')))
    save(path, metadata); save(out/'datasets.json', datasets)
    shutil.copy2(DATA/'sources.json', out/'data_sources.json')
    for p in files:
        dst = out/'source_snapshot'/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True)
    (out/'requirements-executed.txt').write_text(packages.stdout)


@lru_cache(maxsize=2048)
def budget(n, eps, method):
    if method == 'C70_L16':
        return float(parameters(n, eps).m)
    if method in ['original', 'C1025_L1024']:
        return reference_parameters(n, eps, c0=2**26 if method == 'original' else 1025).m
    error = FAILURE if method == 'count_matched_confidence' else Fraction(1, 100)
    with ctx.workprec(256):
        a = (1/ball(error)).log()
        lam = n+a+(a*a+2*n*a).sqrt()
        m = certified_integer(lam/ball(Fraction(eps)), 'ceil')
        t = m*ball(Fraction(eps))-n
        if not (-(t*t)/(2*(n+t))).exp() <= ball(error):
            raise ArithmeticError('Uncertified count-baseline budget')
        return m


def enclosed_outcomes(profiles, plan, lo, hi, S, effective, *, single=False):
    reject, r1 = compare_scores(profiles, plan.n, lo, hi, plan.n, single=single)
    target = min(effective, plan.n)
    if S >= plan.n:
        over = np.zeros(len(lo), dtype=bool); r2 = 0
    else:
        over, r2 = compare_scores(profiles, plan.n, lo, hi, S, single=single)
    if target <= 1:
        short = np.zeros(len(lo), dtype=bool); r3 = 0
    else:
        enough, r3 = compare_scores(profiles, plan.n, lo, hi, target-1, single=single)
        short = ~enough
    low_bound = np.clip(np.ceil(lo), 1, plan.n).astype(np.int64)
    high_bound = np.clip(np.ceil(hi), 1, plan.n).astype(np.int64)
    return dict(reject=reject, over=over, short=short, lower_low=low_bound,
                lower_high=high_bound, score_low=lo, score_high=hi,
                refinements=r1+r2+r3)


def count_outcomes(D, n, S, effective):
    lower = np.clip(D, 1, n)
    return dict(reject=D > n, over=lower > S, short=lower < min(effective, n),
                lower_low=lower, lower_high=lower, score_low=D.astype(float),
                score_high=D.astype(float), refinements=0)


def audit_profiles(profiles, plan, lo, hi, count):
    count = min(count, len(lo))
    ids = np.unique(np.linspace(0, len(lo)-1, count, dtype=int))
    for i in ids:
        exact = arb_score(profiles.observed[i], profiles.low[i], plan.n, bits=768)
        if not (exact >= arb(float(lo[i])) and exact <= arb(float(hi[i]))):
            raise ArithmeticError('Fast interval failed full-Arb cross-check')
        if not (exact > plan.n or exact <= plan.n):
            raise ArithmeticError('Arb audit unresolved')
    return len(ids)


def evaluate_cell(pop, n, eps, scenario, suite, out, design, block_trials, **meta):
    key = hashlib.sha256(repr((suite, scenario, n, eps, meta)).encode()).hexdigest()[:20]
    cell_path = out/'cells'/f'{suite}_{key}.json'
    if cell_path.exists():
        cached = json.loads(cell_path.read_text())
        if sha(out/cached['raw_file']) != cached['raw_sha256']:
            raise RuntimeError('Saved raw fingerprint changed')
        return cached['rows'], cached['blocks']
    plan = parameters(n, eps)
    label, tail, effective = pop.truth(n, eps)
    methods = ['C70_L16'] + (design['ablations'] if plan.scales else []) + design['baselines']
    means = {method: budget(n, eps, 'C70_L16' if method in design['ablations'] else method)
             for method in methods}
    outputs = {method: [] for method in methods}; Ds = {method: [] for method in methods}
    profiles_low = []; block_rows = []; max_mass_error = 0.; audited = 0
    elapsed_sampling = 0.; elapsed_scoring = 0.
    for block in range(design['seed_blocks']):
        rng = stable_rng(design['seeds'][suite], key, 'C70_L16', block)
        start = time.perf_counter()
        profiles, defect = sample_profiles(pop, float(plan.m), block_trials, plan.degree, rng)
        elapsed_sampling += time.perf_counter()-start
        max_mass_error = max(max_mass_error, defect); profiles_low.append(profiles.low)
        start = time.perf_counter()
        lo, hi, slo, shi = score_intervals(profiles, plan)
        current = {'C70_L16': enclosed_outcomes(profiles, plan, lo, hi, pop.support, effective)}
        if plan.scales:
            current['count_same_budget'] = count_outcomes(profiles.observed, n, pop.support, effective)
            current['scale0_same_budget'] = enclosed_outcomes(profiles, plan, slo, shi, pop.support, effective, single=True)
        elapsed_scoring += time.perf_counter()-start
        audited += audit_profiles(profiles, plan, lo, hi, design['numerical_audit_profiles_per_block'])
        for method in current:
            Ds[method].append(profiles.observed)
        for method in design['baselines']:
            baseline, _ = sample_profiles(pop, float(means[method]), block_trials, 0,
                            stable_rng(design['seeds'][suite], key, method, block))
            current[method] = count_outcomes(baseline.observed, n, pop.support, effective)
            Ds[method].append(baseline.observed)
        for method, values in current.items():
            outputs[method].append(values)
            errors = None if label == 'Gap' else int(np.sum(values['reject'] if label == 'Yes' else ~values['reject']))
            block_rows.append(dict(suite=suite, cell=key, method=method, block=block,
                                   trials=block_trials, errors=errors,
                                   lower_errors=int(np.sum(values['over'] | values['short']))))
    arrays = dict(low=np.concatenate(profiles_low), methods=np.asarray(methods),
                  means=np.asarray([str(means[m]) for m in methods]),
                  multiplicities=np.asarray(pop.multiplicities, dtype=np.int64),
                  probabilities_exact=np.asarray(list(map(str, pop.probabilities))))
    rows = []
    for index, method in enumerate(methods):
        values = {field: np.concatenate([v[field] for v in outputs[method]])
                  for field in ['reject', 'over', 'short', 'lower_low', 'lower_high', 'score_low', 'score_high']}
        D = np.concatenate(Ds[method]); T = len(D)
        for field, value in values.items():
            arrays[f'{field}_{index}'] = value
        arrays[f'observed_{index}'] = D
        errors = None if label == 'Gap' else int(np.sum(values['reject'] if label == 'Yes' else ~values['reject']))
        rows.append(dict(suite=suite, cell=key, scenario=scenario, **meta, n=n, epsilon=eps,
                         S=pop.support, truth=label, delta_n=float(tail), delta_n_exact=str(tail),
                         effective_support=effective, method=method, m=means[method],
                         branch=plan.branch if method in ['C70_L16', 'scale0_same_budget'] else 'observed_count',
                         max_degree=plan.degree if method == 'C70_L16' else plan.scales[0].k if method == 'scale0_same_budget' else 0,
                         scales=len(plan.scales) if method == 'C70_L16' else 1 if method == 'scale0_same_budget' else 0,
                         trials=T, errors=errors, error_rate=None if errors is None else errors/T,
                         rejects=int(values['reject'].sum()), reject_rate=float(values['reject'].mean()),
                         lower_errors=int(np.sum(values['over'] | values['short'])),
                         lower_over_errors=int(values['over'].sum()), lower_short_errors=int(values['short'].sum()),
                         mean_observed=float(D.mean()), mean_score_low=float(values['score_low'].mean()),
                         mean_score_high=float(values['score_high'].mean()),
                         mean_lower_low=float(values['lower_low'].mean()), mean_lower_high=float(values['lower_high'].mean()),
                         lower_integer_interval_nonunique=int(np.sum(values['lower_low'] != values['lower_high'])),
                         score_enclosure_max_width=float(np.max(values['score_high']-values['score_low'])),
                         extra_rejects_beyond_count=int(np.sum(values['reject'] & (D <= n))),
                         certified_lower_improvements=int(np.sum(values['lower_low'] > np.clip(D, 1, n))),
                         numerical_refinements=sum(v['refinements'] for v in outputs[method]),
                         full_arb_audits=audited if method == 'C70_L16' else 0,
                         max_poisson_bin_mass_defect=max_mass_error if method == 'C70_L16' else 0,
                         primary_sampling_seconds=elapsed_sampling if method == 'C70_L16' else 0,
                         primary_scoring_seconds=elapsed_scoring if method == 'C70_L16' else 0,
                         raw_file=f'raw/{suite}_{key}.npz', raw_index=index))
    raw_file = f'raw/{suite}_{key}.npz'
    np.savez_compressed(out/raw_file, **arrays)
    save(cell_path, dict(plan=plan.as_dict(), population=pop.as_dict(), rows=rows, blocks=block_rows,
                         raw_file=raw_file, raw_sha256=sha(out/raw_file)))
    return rows, block_rows


def finalize(out, suite, rows, blocks):
    K = sum(row['errors'] is not None for row in rows)
    for row in rows:
        if row['errors'] is not None:
            row['error_upper95'] = cp_upper(row['errors'], row['trials'])
            row['error_upper95_simultaneous'] = cp_upper(row['errors'], row['trials'], .05/max(1, K))
        row['lower_error_upper95_simultaneous'] = cp_upper(row['lower_errors'], row['trials'], .05/len(rows))
    csv_save(out/f'{suite}.csv', rows); csv_save(out/f'{suite}_seed_blocks.csv', blocks)
    save(out/f'{suite}_complete.json', dict(base_cells=len({r['cell'] for r in rows}),
         method_cells=len(rows), method_trial_evaluations=sum(r['trials'] for r in rows),
         independent_sampling_paths=sum(r['trials'] for r in rows if r['method'] not in DESIGN['ablations']),
         primary_trials=sum(r['trials'] for r in rows if r['method'] == 'C70_L16'),
         full_arb_audits=sum(r.get('full_arb_audits', 0) for r in rows)))


def synthetic(out, design):
    rows = []; blocks = []
    for n in design['sizes']:
        for eps in design['epsilons']:
            for name, pop in synthetic_cases(n, eps):
                a, b = evaluate_cell(pop, n, eps, name, 'synthetic', out, design, design['trials_per_block'])
                rows += a; blocks += b
        print('synthetic', n, 'method cells', len(rows), flush=True)
    finalize(out, 'synthetic', rows, blocks)


def natural(out, design):
    rows = []; blocks = []; feasible = []
    for spec in json.loads((out/'datasets.json').read_text()):
        name = spec['dataset']; counts = load(name); pop = Population.from_counts(counts)
        for eps in design['natural_epsilons']:
            eff = pop.effective(eps)
            for n in sorted({max(1, pop.support//2), max(1, eff-1), eff, pop.support, math.ceil(1.5*pop.support)}):
                a, b = evaluate_cell(pop, n, eps, name, 'natural', out, design, design['trials_per_block'], dataset=name)
                rows += a; blocks += b
                m = float(parameters(n, eps).m); N = int(counts.sum())
                feasible.append(dict(dataset=name, n=n, epsilon=eps, N=N, S=pop.support, m=m,
                                     has_mean_budget=N >= m, budget_to_records=m/N,
                                     poisson_overflow_probability=float(poisson.sf(N, m))))
        print('natural', name, 'method cells', len(rows), flush=True)
    finalize(out, 'natural', rows, blocks); csv_save(out/'natural_feasibility.csv', feasible)


def lifted(out, design):
    rows = []; blocks = []
    for spec in json.loads((out/'datasets.json').read_text()):
        name = spec['dataset']; original = Population.from_counts(load(name))
        for target in design['lifted_target_supports']:
            copies = math.ceil(target/original.support); pop = original.split_categories(copies)
            for eps in design['lifted_epsilons']:
                eff = pop.effective(eps)
                for n in sorted({max(1, eff-1), eff, pop.support}):
                    a, b = evaluate_cell(pop, n, eps, name, 'lifted', out, design,
                                        design['lifted_trials_per_block'], dataset=name, clone_factor=copies,
                                        scope='synthetic category splitting of empirical abundance shape')
                    rows += a; blocks += b
        print('lifted', name, 'method cells', len(rows), flush=True)
    finalize(out, 'lifted', rows, blocks)


def amplification(out, design):
    """Actual independent copies, binary majority and support-bound median."""
    candidates = []
    for n in design['amplification_sizes']:
        for eps in design['amplification_epsilons']:
            candidates += [(name, pop, n, eps, 'synthetic') for name, pop in synthetic_cases(n, eps)
                           if name in design['amplification_scenarios']]
    for spec in json.loads((out/'datasets.json').read_text()):
        pop = Population.from_counts(load(spec['dataset'])); eps = '0.1'
        for n in sorted({max(1, pop.effective(eps)-1), pop.support}):
            candidates.append((spec['dataset'], pop, n, eps, 'natural'))
    rows = []; blocks = []; repeats = design['amplification_repeats']; half = repeats//2
    for name, pop, n, eps, origin in candidates:
        key = hashlib.sha256(repr(('amp', name, n, eps, origin)).encode()).hexdigest()[:20]
        path = out/'cells'/f'amplification_{key}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if sha(out/old['raw_file']) != old['raw_sha256']:
                raise RuntimeError('Amplification raw data changed')
            rows += old['rows']; blocks += old['blocks']; continue
        plan = parameters(n, eps); label, tail, eff = pop.truth(n, eps)
        raw = {}; outcomes = []; local_blocks = []; lows = []; Ds = []
        baseline_outputs = {'count_99': [], 'C1025_L1024': []}
        for block in range(design['seed_blocks']):
            T = design['amplification_trials_per_block']
            fp, _ = sample_profiles(pop, float(plan.m), T*repeats, plan.degree,
                                   stable_rng(design['seeds']['amplification'], key, block))
            lo, hi, _, _ = score_intervals(fp, plan)
            values = enclosed_outcomes(fp, plan, lo, hi, pop.support, eff)
            reject = values['reject'].reshape(T, repeats).sum(axis=1) > half
            lower_lo = np.sort(values['lower_low'].reshape(T, repeats), axis=1)[:, half]
            lower_hi = np.sort(values['lower_high'].reshape(T, repeats), axis=1)[:, half]
            # Median>threshold iff a strict majority of the individual outputs
            # exceed it, so exact certified per-copy events determine coverage.
            over = values['over'].reshape(T, repeats).sum(axis=1) > half
            short = values['short'].reshape(T, repeats).sum(axis=1) > half
            outcomes.append(dict(reject=reject, over=over, short=short, lower_low=lower_lo, lower_high=lower_hi))
            lows.append(fp.low); Ds.append(fp.observed)
            raw[f'copy_reject_{block}'] = values['reject'].reshape(T, repeats)
            raw[f'copy_lower_low_{block}'] = values['lower_low'].reshape(T, repeats)
            raw[f'copy_lower_high_{block}'] = values['lower_high'].reshape(T, repeats)
            for method in baseline_outputs:
                base, _ = sample_profiles(pop, float(budget(n, eps, method)), T, 0,
                        stable_rng(design['seeds']['amplification'], key, method, block))
                baseline_outputs[method].append(count_outcomes(base.observed, n, pop.support, eff))
                raw[f'{method}_observed_{block}'] = base.observed
        raw['low'] = np.concatenate(lows); raw['observed'] = np.concatenate(Ds)
        local_rows = []
        for method, values_list in [('C70_L16_x27', outcomes), *baseline_outputs.items()]:
            merged = {k: np.concatenate([v[k] for v in values_list]) for k in ['reject', 'over', 'short', 'lower_low', 'lower_high']}
            T = len(merged['reject'])
            errors = None if label == 'Gap' else int(np.sum(merged['reject'] if label == 'Yes' else ~merged['reject']))
            raw[f'{method}_reject'] = merged['reject']; raw[f'{method}_lower_errors'] = merged['over'] | merged['short']
            local_rows.append(dict(suite='amplification', cell=key, scenario=name, origin=origin,
                  n=n, epsilon=eps, S=pop.support, truth=label, delta_n_exact=str(tail), effective_support=eff,
                  method=method, m=repeats*float(plan.m) if method == 'C70_L16_x27' else budget(n, eps, method),
                  repeats=repeats if method == 'C70_L16_x27' else 1, trials=T, errors=errors,
                  error_rate=None if errors is None else errors/T, lower_errors=int(np.sum(merged['over'] | merged['short'])),
                  mean_lower_low=float(merged['lower_low'].mean()), mean_lower_high=float(merged['lower_high'].mean()),
                  raw_file=f'raw/amplification_{key}.npz'))
            for block, values in enumerate(values_list):
                err = None if label == 'Gap' else int(np.sum(values['reject'] if label == 'Yes' else ~values['reject']))
                local_blocks.append(dict(suite='amplification', cell=key, method=method, block=block,
                     trials=len(values['reject']), errors=err, lower_errors=int(np.sum(values['over'] | values['short']))))
        raw_file = f'raw/amplification_{key}.npz'; np.savez_compressed(out/raw_file, **raw)
        save(path, dict(population=pop.as_dict(), plan=plan.as_dict(), rows=local_rows, blocks=local_blocks,
                        raw_file=raw_file, raw_sha256=sha(out/raw_file)))
        rows += local_rows; blocks += local_blocks
        print('amplification', origin, name, n, eps, flush=True)
    finalize(out, 'amplification', rows, blocks)
    p = FAILURE
    bound = sum(Fraction(math.comb(repeats, k))*p**k*(1-p)**(repeats-k) for k in range(half+1, repeats+1))
    save(out/'amplification_theory.json', dict(repeats=repeats, single_failure=str(p),
         per_task_failure_exact=str(bound), per_task_failure=float(bound),
         either_output_failure_upper=float(2*bound), expected_coefficient=70*repeats,
         independent_single_run_paths=sum(r['trials']*r['repeats'] for r in rows)))


def field_stress(out, design):
    """Actual finite-file subsampling and explicitly non-iid stress cases."""
    finite = []; duplicate = []; grouped = []; holdout = []; api = []
    for spec in json.loads((out/'datasets.json').read_text()):
        name = spec['dataset']; counts = load(name); N = int(counts.sum()); pop = Population.from_counts(counts)
        for eps in design['finite_file_epsilons']:
            for n in sorted({max(1, pop.effective(eps)-1), pop.support}):
                plan = parameters(n, eps); rng = stable_rng(design['seeds']['finite'], name, eps, n)
                M = rng.poisson(float(plan.m), size=design['finite_file_trials'])
                accepted = M <= N; rejects = 0; successful_api = 0
                recorded = []
                for total in M[accepted]:
                    draw = rng.multivariate_hypergeometric(counts, int(total))
                    D = int(np.sum(draw > 0)); rejects += D > n
                    recorded.append(D)
                    if successful_api < 2:
                        result = test_fingerprint(Fingerprint.from_counts(draw), n, eps)
                        api.append(dict(dataset=name, sampling='without replacement from fixed observed file', **result))
                        successful_api += 1
                finite.append(dict(dataset=name, n=n, epsilon=eps, N=N, S=pop.support, m=float(plan.m),
                    trials=len(M), overflows=int((~accepted).sum()), completed=int(accepted.sum()), rejects=rejects,
                    overflow_probability=float(poisson.sf(N, float(plan.m))),
                    scope='Descriptive fixed-file subsampling; overflow is Inconclusive, never a Yes or silently resampled'))
                np.savez_compressed(out/'raw'/f'finite_{name}_{n}_{eps}.npz', M=M, completed_observed=np.array(recorded, dtype=np.int64))
        # Direct full-fingerprint iid empirical resampling at declared mean.
        for eps in ['0.01', '0.1']:
            for n in sorted({max(1, pop.effective(eps)-1), pop.support}):
                plan = parameters(n, eps)
                for rep in range(3):
                    rng = stable_rng(design['seeds']['finite'], name, 'public_iid', eps, n, rep)
                    draw = rng.poisson(float(plan.m)*counts/N)
                    api.append(dict(dataset=name, sampling='iid finite empirical population',
                                    **test_fingerprint(Fingerprint.from_counts(draw), n, eps)))
    for name in design['group_datasets']:
        matrix = sparse.load_npz(DATA/'prepared'/f'{name}_groups.npz').toarray()
        counts = matrix.sum(axis=0); pop = Population.from_counts(counts)
        for eps in ['0.01', '0.1']:
            n = max(1, pop.effective(eps)-1); plan = parameters(n, eps); label, tail, eff = pop.truth(n, eps)
            for factor in design['duplication_factors']:
                # Poisson(m/factor) fresh labels, each copied factor times.
                fp, _ = sample_profiles(pop, float(plan.m)/factor, design['group_trials'], 0,
                    stable_rng(design['seeds']['groups'], name, eps, factor))
                errors = int(np.sum(fp.observed <= n)) if label == 'No' else None
                duplicate.append(dict(dataset=name, n=n, epsilon=eps, factor=factor, m=float(plan.m),
                      truth=label, trials=len(fp.observed), errors=errors,
                      error_rate=None if errors is None else errors/len(fp.observed)))
                np.savez_compressed(out/'raw'/f'duplicate_{name}_{eps}_{factor}.npz', observed=fp.observed)
            rng = stable_rng(design['seeds']['groups'], name, eps, 'whole_groups')
            # Independent Poisson multiplicities of whole observed groups.
            # Each category's expected count is m*p_i, but within-group labels
            # are dependent and the total count is compound Poisson.
            group_rate = float(plan.m)/int(counts.sum())
            weights = rng.poisson(group_rate, size=(design['group_trials'], len(matrix)))
            totals = weights @ matrix; D = (totals > 0).sum(axis=1)
            errors = int(np.sum(D <= n)) if label == 'No' else None
            grouped.append(dict(dataset=name, n=n, epsilon=eps, truth=label, m=float(plan.m),
                     trials=len(D), errors=errors, error_rate=None if errors is None else errors/len(D),
                     mean_realized_records=float(totals.sum(axis=1).mean()), scope='compound-Poisson whole-group resampling'))
            np.savez_compressed(out/'raw'/f'groups_{name}_{eps}.npz', group_multiplicities=weights, observed=D)
        for split in range(design['group_splits']):
            rng = stable_rng(design['seeds']['groups'], name, 'holdout', split)
            order = rng.permutation(len(matrix)); cut = max(1, int(.7*len(matrix)))
            train = Population.from_counts(matrix[order[:cut]].sum(axis=0))
            test = Population.from_counts(matrix[order[cut:]].sum(axis=0)); eps = '0.1'
            for rule, n in [('train_effective_boundary', max(1, train.effective(eps)-1)), ('train_support', train.support)]:
                plan = parameters(n, eps); label, tail, eff = test.truth(n, eps)
                fp, _ = sample_profiles(test, float(plan.m), design['group_trials'], 0, rng)
                val = count_outcomes(fp.observed, n, test.support, eff)
                errors = None if label == 'Gap' else int(np.sum(val['reject'] if label == 'Yes' else ~val['reject']))
                holdout.append(dict(dataset=name, split=split, rule=rule, n=n, epsilon=eps, S=test.support,
                       truth=label, delta_n_exact=str(tail), m=float(plan.m), train_groups=cut,
                       test_groups=len(order)-cut, trials=len(fp.observed), errors=errors,
                       error_rate=None if errors is None else errors/len(fp.observed),
                       lower_errors=int(np.sum(val['over'] | val['short']))))
                np.savez_compressed(out/'raw'/f'holdout_{name}_{split}_{rule}.npz',
                                    train_groups=order[:cut], test_groups=order[cut:], observed=fp.observed)
        print('field stress', name, flush=True)
    for name, rows in [('finite_files', finite), ('duplication_stress', duplicate),
                       ('group_stress', grouped), ('group_holdout', holdout)]:
        K = sum(r.get('errors') is not None for r in rows)
        for r in rows:
            if r.get('errors') is not None:
                r['error_upper95_simultaneous'] = cp_upper(r['errors'], r['trials'], .05/max(1, K))
        csv_save(out/f'{name}.csv', rows)
    save(out/'public_fingerprint_runs.json', api)
    save(out/'field_stress_complete.json', dict(finite_cells=len(finite), duplicate_cells=len(duplicate),
         group_cells=len(grouped), holdout_cells=len(holdout), public_api_calls=len(api)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'synthetic', 'natural', 'lifted', 'amplification', 'field', 'all'], default='all')
    parser.add_argument('--out', type=Path, default=ROOT/'results_c7016')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    design = smoke_design() if args.smoke else DESIGN
    freeze(args.out, design)
    phases = dict(synthetic=synthetic, natural=natural, lifted=lifted, amplification=amplification, field=field_stress)
    for name, operation in phases.items():
        if args.phase in ['all', name]:
            start = time.perf_counter(); operation(args.out, design)
            save(args.out/f'execution_{name}.json', dict(seconds=time.perf_counter()-start,
                 finished_at=datetime.now(timezone.utc).isoformat(), command=sys.argv))


if __name__ == '__main__':
    main()
