"""Predeclared real-population Poisson experiments, with auditable fingerprints.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.run
This is a conditional empirical-distribution experiment, not a field iid claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import platform

import numpy as np
from scipy.stats import beta
from benchmark_core import Profiles
from c7016.populations import Population
from guaranteed import certified_count_budget
from real_l01 import algorithm as alg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results_real_l01_20261104'
MODES = ['full', 'single', 'count', 'no_count']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def seed_for(*keys):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'little')


def design(populations, trials):
    cases = []
    for record in populations:
        pop = Population.from_counts(record['counts'])
        for epsilon in ['0.01', '0.1', '0.5', '0.9']:
            eff = pop.effective(epsilon)
            candidates = [('below_effective', max(1, eff-1)),
                          ('middle_support', (eff+pop.support)//2),
                          ('at_support', pop.support), ('twice_support', 2*pop.support)]
            seen = set()
            for regime, n in candidates:
                if n in seen:
                    continue
                seen.add(n)
                truth, tail, eff = pop.truth(n, epsilon)
                cases.append(dict(id=f'c{len(cases):04d}', population_id=record['id'],
                                  dataset=record['dataset'], n=n, epsilon=epsilon,
                                  regime=regime, truth=truth, tail_exact=str(tail),
                                  eff=eff, S=pop.support, trials=trials))
    return cases


def profiles_from_draws(draws, degree):
    D = np.count_nonzero(draws, axis=1).astype(np.int64)
    low = np.zeros((len(draws), degree), dtype=np.int64)
    rows, cols = np.nonzero((draws > 0) & (draws <= degree))
    np.add.at(low, (rows, draws[rows, cols]-1), 1)
    return Profiles(D, low)


def exact_lowers(profiles, plan, result, mode):
    output = result['lower_lo'].copy()
    uncertain = np.flatnonzero(output != result['lower_hi'])
    for i in uncertain:
        output[i] = alg.exact_output(profiles.observed[i], profiles.low[i], plan, mode)['lower']
    return output, len(uncertain)


def upper(k, n, alpha=.05):
    return 1. if k == n else float(beta.ppf(1-alpha, k+1, n-k))


def run_case(case, record, out):
    start = time.perf_counter()
    pop = Population.from_counts(record['counts'])
    plan = alg.parameters('70', 1, case['n'], case['epsilon'])
    m = float(plan.m)
    counts = np.array(record['counts'], dtype=np.int64)
    probabilities = counts/counts.sum()
    trials = case['trials']
    all_D, all_low, all_M, block_ids = [], [], [], []
    sampling_seconds = 0.
    seeds = []
    for block, size in enumerate(np.full(4, trials//4) + (np.arange(4) < trials % 4)):
        seed = seed_for('real_l01_v1', case['id'], case['population_id'], block)
        seeds.append(seed)
        rng = np.random.default_rng(seed)
        t = time.perf_counter()
        draws = rng.poisson(m*probabilities, size=(int(size), len(counts)))
        sampling_seconds += time.perf_counter()-t
        fp = profiles_from_draws(draws, plan.degree)
        all_D.append(fp.observed); all_low.append(fp.low); all_M.append(draws.sum(axis=1))
        block_ids.extend([block]*int(size))
    profiles = Profiles(np.concatenate(all_D), np.concatenate(all_low))
    M = np.concatenate(all_M)
    t = time.perf_counter()
    a, b = alg.scale_intervals(profiles, plan)
    scoring_seconds = time.perf_counter()-t
    raw = dict(observed=profiles.observed, low=profiles.low, M=M,
               block=np.array(block_ids, dtype=np.int8))
    stats = {}
    refinements = 0
    lowers = {}
    for mode in MODES:
        result, refined = alg.outcomes(profiles, plan, pop, a, b, mode)
        lower, refined_integer = exact_lowers(profiles, plan, result, mode)
        refinements += refined + refined_integer
        raw[mode+'_lower'] = lower
        for event in ['reject', 'over', 'short', 'binary', 'joint']:
            raw[mode+'_'+event] = result[event]
        lowers[mode] = lower
        stats[mode] = {key: int(np.count_nonzero(result[key]))
                       for key in ['reject', 'over', 'short', 'binary', 'joint']}
        stats[mode].update(mean_lower=float(lower.mean()),
                           q05_lower=float(np.quantile(lower, .05)),
                           q95_lower=float(np.quantile(lower, .95)),
                           mean_unclipped_score_interval=[float(result['lo'].mean()),
                                                         float(result['hi'].mean())],
                           block_joint_failures=[int(result['joint'][np.array(block_ids)==j].sum())
                                                 for j in range(4)])
    # Separately budgeted observed-count baseline, certified at risk <=1/3.
    baseline_m, baseline_cert = certified_count_budget(plan.n, plan.epsilon, Fraction(1, 3))
    baseline_seed = seed_for('real_l01_v1_count_budget', case['id'], case['population_id'])
    t = time.perf_counter()
    baseline_D = np.random.default_rng(baseline_seed).binomial(
        1, -np.expm1(-baseline_m*probabilities), size=(trials, len(counts))).sum(axis=1)
    baseline_seconds = time.perf_counter()-t
    baseline_lower = np.clip(baseline_D, 1, plan.n)
    baseline_reject = baseline_D > plan.n
    baseline_binary = (baseline_reject if case['truth']=='Yes' else ~baseline_reject
                       if case['truth']=='No' else np.zeros(trials, dtype=bool))
    baseline_short = baseline_lower < min(case['eff'], plan.n)
    raw.update(count_budget_observed=baseline_D, count_budget_lower=baseline_lower,
               count_budget_binary=baseline_binary, count_budget_short=baseline_short)
    stats['count_budget'] = dict(binary=int(baseline_binary.sum()), over=0,
                                short=int(baseline_short.sum()),
                                joint=int((baseline_binary|baseline_short).sum()),
                                mean_lower=float(baseline_lower.mean()))
    # L0=14 is exactly count on all these n; explicitly verify rather than rerun.
    legacy = alg.parameters('70', 14, plan.n, plan.epsilon)
    if legacy.scales:
        raise ValueError('L0=14 unexpectedly active: add its genuine paired evaluation')
    raw_path = out/'raw'/f"{case['id']}.npz"
    raw_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    result = dict(**case, C0=70, L0=1, L=float(plan.L), m=m,
                  m_interval=plan.m.str(65), m_over_source_N=m/int(counts.sum()),
                  source_N=int(counts.sum()), degree=plan.degree,
                  scales=len(plan.scales), degrees=[s.k for s in plan.scales],
                  mean_actual_M=float(M.mean()), var_actual_M=float(M.var(ddof=1)),
                  seeds=seeds, baseline_seed=baseline_seed,
                  probability_sum_binary64=float(probabilities.sum()),
                  baseline_m=baseline_m, baseline_tail_certificate=baseline_cert,
                  m_ratio_to_count_budget=m/baseline_m,
                  L0_14_same_budget_equals_count=True,
                  stats=stats, refinements=refinements,
                  full_better_than_count=int(np.sum(lowers['full']>lowers['count'])),
                  full_better_than_single=int(np.sum(lowers['full']>lowers['single'])),
                  mean_gain_over_count=float(np.mean(lowers['full']-lowers['count'])),
                  sampling_seconds=sampling_seconds, scoring_seconds=scoring_seconds,
                  baseline_seconds=baseline_seconds, elapsed_seconds=time.perf_counter()-start,
                  raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    dump(out/'cells'/f"{case['id']}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.trials < 4:
        raise ValueError('Need four independent seed blocks')
    out = args.out
    populations = json.loads((OUT/'prepared/populations.json').read_text())
    lookup = {p['id']: p for p in populations}
    cases = design(populations, args.trials)
    protocol = dict(C0=70, L0=1, poisson_mean='70*n/(epsilon*(1+log(n))) (real, not ceil)',
                    trials=args.trials, blocks=4, cases=cases, modes=MODES,
                    confidence='One-sided exact binomial upper bounds, both per cell and Bonferroni across cells',
                    stopping='Fixed trials in every cell; no outcome-dependent selection or stopping',
                    interpretation='IID Poisson resampling from the fixed empirical count distribution; no claim about unseen field species',
                    population_sha256=hashlib.sha256((OUT/'prepared/populations.json').read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__)
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; choose a distinct output directory')
    else:
        dump(protocol_path, protocol)
    sources = {}
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value:
            p = Path(value).resolve()
            if p.is_relative_to(ROOT) and '.venv' not in p.parts and p.suffix == '.py':
                rel = p.relative_to(ROOT)
                sources[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
                destination = out/'source_snapshot'/rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != p.read_bytes():
                    raise ValueError(f'Frozen source changed: {rel}')
                shutil.copyfile(p, destination)
    dump(out/'source_hashes.json', sources)
    print(f'Frozen {len(cases)} cells, {len(populations)} populations, {args.trials} trials/cell', flush=True)
    results = []
    for index, case in enumerate(cases):
        cell = out/'cells'/f"{case['id']}.json"
        if cell.exists():
            result = json.loads(cell.read_text())
            raw = out/'raw'/f"{case['id']}.npz"
            if hashlib.sha256(raw.read_bytes()).hexdigest() != result['raw_sha256']:
                raise ValueError(f'Raw hash mismatch {case["id"]}')
        else:
            result = run_case(case, lookup[case['population_id']], out)
        results.append(result)
        if index % 20 == 0 or index == len(cases)-1:
            print(f'{index+1}/{len(cases)} full failures={sum(r["stats"]["full"]["joint"] for r in results)} elapsed={sum(r["elapsed_seconds"] for r in results):.1f}s', flush=True)
    for result in results:
        for stats in result['stats'].values():
            stats['joint_upper_95'] = upper(stats['joint'], args.trials)
            stats['joint_upper_simultaneous_95'] = upper(stats['joint'], args.trials, .05/len(cases))
    dump(out/'results.json', results)
    summary = dict(cells=len(results), populations=len(populations),
                   trials_per_cell=args.trials, paths=len(results)*args.trials,
                   truth_counts=dict(Counter(r['truth'] for r in results)),
                   total_refinements=sum(r['refinements'] for r in results),
                   full_failures=sum(r['stats']['full']['joint'] for r in results),
                   max_full_failure_rate=max(r['stats']['full']['joint']/args.trials for r in results),
                   max_full_upper_95=max(r['stats']['full']['joint_upper_95'] for r in results),
                   max_full_upper_simultaneous_95=max(r['stats']['full']['joint_upper_simultaneous_95'] for r in results),
                   gain_paths=sum(r['full_better_than_count'] for r in results),
                   multiscale_gain_paths=sum(r['full_better_than_single'] for r in results),
                   mean_gain=float(np.mean([r['mean_gain_over_count'] for r in results])),
                   m_ratio_range=[min(r['m_ratio_to_count_budget'] for r in results),
                                  max(r['m_ratio_to_count_budget'] for r in results)],
                   m_exceeds_source_N_cells=sum(r['m_over_source_N']>1 for r in results),
                   elapsed_seconds=sum(r['elapsed_seconds'] for r in results))
    dump(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
