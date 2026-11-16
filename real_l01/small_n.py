"""Fixed synthetic small-n diagnostics, kept separate from real-data results.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.small_n
The population family and every trial count are frozen before simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from c7016.populations import Population
from real_l01.run import ROOT, dump, run_case, upper


N_VALUES = [1, 2, 3, 5, 10, 20, 40, 75, 100, 403]
EPSILONS = ['0.1', '0.5', '0.9']
OUT = ROOT/'results_real_l01_20261104/small_n'


def normalized_counts(counts):
    counts = sorted(map(int, counts), reverse=True)
    divisor = reduce(math.gcd, counts)
    return tuple(c//divisor for c in counts)


def design(trials):
    records, cases, seen = {}, [], set()
    duplicates = 0
    for n in N_VALUES:
        for epsilon in EPSILONS:
            eps = Fraction(epsilon)
            candidates = [('uniform', [1]*S)
                          for S in (max(1, n//2), n, 2*n)]
            if n == 1:
                head_counts = [1]
            else:
                # p_head=1-eps/2 and p_tail=eps/(2*(n-1)), exactly.
                head_counts = [(2*eps.denominator-eps.numerator)*(n-1)]
                head_counts += [eps.numerator]*(n-1)
            candidates.append(('heavy_head', head_counts))
            for family, raw_counts in candidates:
                counts = normalized_counts(raw_counts)
                key = n, epsilon, counts
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                digest = hashlib.sha256(repr(counts).encode()).hexdigest()[:16]
                population_id = f'synthetic_{digest}'
                if population_id not in records:
                    records[population_id] = dict(
                        id=population_id, dataset='synthetic_small_n', family=family,
                        counts=list(counts), synthetic=True,
                        interpretation='Constructed exact population; not downloaded biological data',
                    )
                pop = Population.from_counts(counts)
                truth, tail, eff = pop.truth(n, epsilon)
                cases.append(dict(
                    id=f'sn{len(cases):04d}', population_id=population_id,
                    dataset='synthetic_small_n', synthetic=True, family=family,
                    n=n, epsilon=epsilon, regime=f'{family}_S{pop.support}',
                    truth=truth, tail_exact=str(tail), eff=eff, S=pop.support,
                    trials=trials,
                ))
    return records, cases, duplicates


def summarize(results, trials):
    modes = ['full', 'single', 'count', 'no_count', 'count_budget']
    method_summary = {}
    for mode in modes:
        worst = max(results, key=lambda r: r['stats'][mode]['joint'])
        worst_count = worst['stats'][mode]['joint']
        tied = [r['id'] for r in results if r['stats'][mode]['joint'] == worst_count]
        method_summary[mode] = dict(
            failures=sum(r['stats'][mode]['joint'] for r in results),
            maximum_cell_failure_rate=worst_count/trials,
            maximum_cell_upper_95=upper(worst_count, trials),
            maximum_cell_upper_simultaneous_95=upper(worst_count, trials, .05/len(results)),
            worst_cell={key: worst[key] for key in (
                'id', 'family', 'n', 'epsilon', 'S', 'truth', 'eff')},
            worst_cell_failures=worst_count,
            worst_cell_ties=tied,
        )
    return dict(
        synthetic=True, included_in_main_real_data_summary=False,
        interpretation='Finite diagnostic simulation, not a universal probability certificate',
        cells=len(results), populations=len({r['population_id'] for r in results}),
        trials_per_cell=trials, paths=len(results)*trials,
        truth_counts=dict(Counter(r['truth'] for r in results)),
        family_counts=dict(Counter(r['family'] for r in results)),
        confidence='One-sided Clopper-Pearson; Bonferroni across all cells for each fixed method separately',
        method_summary=method_summary,
        full_gain_over_count_paths=sum(r['full_better_than_count'] for r in results),
        full_gain_over_single_paths=sum(r['full_better_than_single'] for r in results),
        total_refinements=sum(r['refinements'] for r in results),
        elapsed_seconds=sum(r['elapsed_seconds'] for r in results),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.trials < 4:
        raise ValueError('Need four independent seed blocks')
    records, cases, duplicates = design(args.trials)
    sources = {}
    for name in ('real_l01/small_n.py', 'real_l01/run.py', 'real_l01/algorithm.py',
                 'followup/core.py', 'parameter_search/algorithm.py',
                 'c7016/populations.py', 'paper_reference.py', 'guaranteed.py'):
        sources[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    protocol = dict(
        synthetic=True, included_in_main_real_data_summary=False,
        C0=70, L0=1, n_values=N_VALUES, epsilons=EPSILONS,
        uniform_supports='max(1,n//2), n, 2*n',
        heavy_head='S=n; p1=1-epsilon/2, others=epsilon/(2*(n-1)); n=1 is the point mass',
        duplicates_removed=duplicates, deduplication='Exact normalized population, n and epsilon',
        trials_per_cell=args.trials, independent_seed_blocks=4,
        stopping='All designs and trials frozen before any output; no outcome-dependent selection',
        confidence='One-sided Clopper-Pearson, 95% simultaneous across cells for each fixed method',
        python=platform.python_version(), numpy=np.__version__,
        source_sha256=sources, populations=list(records.values()), cases=cases,
    )
    protocol_path = args.out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing frozen protocol differs; choose a separate output directory')
    else:
        dump(protocol_path, protocol)
    print(f'Frozen SYNTHETIC diagnostics: {len(cases)} cells, '
          f'{args.trials} trials/cell, {duplicates} duplicates removed', flush=True)
    results = []
    for index, case in enumerate(cases):
        cell = args.out/'cells'/f"{case['id']}.json"
        if cell.exists():
            result = json.loads(cell.read_text())
            raw = args.out/'raw'/f"{case['id']}.npz"
            if hashlib.sha256(raw.read_bytes()).hexdigest() != result['raw_sha256']:
                raise ValueError(f'Raw checksum mismatch: {case["id"]}')
        else:
            result = run_case(case, records[case['population_id']], args.out)
        results.append(result)
        if index % 10 == 0 or index == len(cases)-1:
            print(f'{index+1}/{len(cases)} full failures='
                  f'{sum(r["stats"]["full"]["joint"] for r in results)}', flush=True)
    summary = summarize(results, args.trials)
    dump(args.out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
