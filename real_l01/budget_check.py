"""Post hoc extension: a count baseline with predeclared certified 1% risk.

All original real-data cells are retained. The extension is frozen before its
own independent simulations, and does not alter the original run or results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from c7016.populations import Population
from guaranteed import certified_count_budget
from real_l01.run import ROOT, dump, seed_for, upper


MAIN = ROOT/'results_real_l01_20261104'
OUT = MAIN/'budget_01'
ERROR = Fraction(1, 100)
TRIALS = 2000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    hashes = {}
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.is_relative_to(ROOT) and '.venv' not in path.parts and path.suffix == '.py':
            hashes[str(path.relative_to(ROOT))] = digest(path)
    return hashes


def plan_extension(main_protocol, main_results, populations):
    if len(main_protocol['cases']) != 640:
        raise ValueError('This extension was declared for exactly 640 existing cells')
    records = {record['id']: record for record in populations}
    previous = {result['id']: result for result in main_results}
    cases = []
    for case in main_protocol['cases']:
        pop = Population.from_counts(records[case['population_id']]['counts'])
        truth, tail, eff = pop.truth(case['n'], case['epsilon'])
        assert (truth, str(tail), eff, pop.support) == (
            case['truth'], case['tail_exact'], case['eff'], case['S'])
        m, certificate = certified_count_budget(case['n'], Fraction(case['epsilon']), ERROR)
        cases.append(dict(
            **case, baseline_risk=str(ERROR), baseline_m=m,
            baseline_tail_certificate=certificate,
            seeds=[seed_for('budget_01_extension_v1', case['id'], case['population_id'], block)
                   for block in range(4)],
            c70_m=previous[case['id']]['m'],
            c70_m_interval=previous[case['id']]['m_interval'],
            c70_m_over_baseline_m=previous[case['id']]['m']/m,
        ))
    return records, previous, cases


def run_case(case, record, previous, out):
    started = time.perf_counter()
    counts = np.asarray(record['counts'], dtype=np.int64)
    probabilities = counts/counts.sum()
    occupancy = -np.expm1(-case['baseline_m']*probabilities)
    assert np.all((0 <= occupancy) & (occupancy <= 1))
    blocks = []
    for seed in case['seeds']:
        rng = np.random.default_rng(seed)
        # These independent indicators have exactly the Poisson-occupancy law
        # in real arithmetic; a total Poisson sample size is not reconstructed.
        D = rng.binomial(1, occupancy, size=(TRIALS//4, len(counts))).sum(axis=1)
        blocks.append(D)
    D = np.concatenate(blocks)
    assert len(D) == TRIALS and np.all(D <= case['S'])
    lower = np.clip(D, 1, case['n'])
    reject = D > case['n']
    binary = (reject if case['truth'] == 'Yes' else ~reject
              if case['truth'] == 'No' else np.zeros(TRIALS, dtype=bool))
    over = lower > min(case['S'], case['n'])
    short = lower < min(case['eff'], case['n'])
    joint = binary | over | short
    assert not np.any(over)
    if case['truth'] == 'No':
        assert np.array_equal(binary, D <= case['n'])
    elif case['truth'] == 'Yes':
        assert not np.any(binary)
    stats = {name: int(np.count_nonzero(value)) for name, value in (
        ('reject', reject), ('binary', binary), ('over', over), ('short', short), ('joint', joint))}
    stats.update(
        failure_rate=stats['joint']/TRIALS,
        mean_lower=float(lower.mean()),
        block_joint_failures=[int(x.sum()) for x in joint.reshape(4, TRIALS//4)],
        joint_upper_95=upper(stats['joint'], TRIALS),
        joint_upper_simultaneous_95=upper(stats['joint'], TRIALS, .05/640),
    )
    path = out/'raw'/f"{case['id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observed=D, lower=lower, reject=reject, binary=binary,
                        over=over, short=short, joint=joint,
                        block=np.repeat(np.arange(4, dtype=np.int8), TRIALS//4))
    result = dict(
        **case, extension='post_hoc_predeclared_1pct_count_budget',
        included_in_original_main_summary=False,
        probabilities_sum_binary64=float(probabilities.sum()), stats=stats,
        main_full_stats=previous['stats']['full'],
        comparison='Independent seeds; not a paired comparison to the original C70 run',
        raw_sha256=digest(path), elapsed_seconds=time.perf_counter()-started,
    )
    dump(out/'cells'/f"{case['id']}.json", result)
    return result


def summarize(results):
    worst = max(results, key=lambda result: result['stats']['joint'])
    worst_failures = worst['stats']['joint']
    datasets = {}
    for dataset in sorted({result['dataset'] for result in results}):
        cells = [result for result in results if result['dataset'] == dataset]
        datasets[dataset] = dict(
            cells=len(cells), failures=sum(result['stats']['joint'] for result in cells),
            worst_failure_rate=max(result['stats']['failure_rate'] for result in cells),
            c70_m_over_baseline_m_range=[
                min(result['c70_m_over_baseline_m'] for result in cells),
                max(result['c70_m_over_baseline_m'] for result in cells)],
        )
    return dict(
        extension='Post hoc extension prompted by observed accuracy/budget tradeoff; all extension choices frozen before its own simulations',
        included_in_original_main_summary=False,
        cells=len(results), trials_per_cell=TRIALS, paths=TRIALS*len(results),
        risk_certificate='For each fixed population and cell, the count rule has joint failure risk <= 1/100 under the iid Poisson model',
        confidence='One-sided Clopper-Pearson, Bonferroni across 640 extension cells',
        total_failures=sum(result['stats']['joint'] for result in results),
        worst_failure_rate=worst['stats']['failure_rate'],
        maximum_cell_upper_95=worst['stats']['joint_upper_95'],
        maximum_cell_upper_simultaneous_95=worst['stats']['joint_upper_simultaneous_95'],
        worst_cell={key: worst[key] for key in (
            'id', 'population_id', 'dataset', 'regime', 'n', 'epsilon', 'S', 'eff',
            'truth', 'baseline_m', 'c70_m', 'c70_m_over_baseline_m')},
        worst_cell_failures=worst_failures,
        worst_cell_ties=[result['id'] for result in results
                         if result['stats']['joint'] == worst_failures],
        c70_m_over_baseline_m_range=[
            min(result['c70_m_over_baseline_m'] for result in results),
            max(result['c70_m_over_baseline_m'] for result in results)],
        count_budget_range=[min(result['baseline_m'] for result in results),
                            max(result['baseline_m'] for result in results)],
        truth_counts=dict(Counter(result['truth'] for result in results)),
        datasets=datasets,
        elapsed_seconds=sum(result['elapsed_seconds'] for result in results),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    original_protocol = json.loads((MAIN/'protocol.json').read_text())
    original_results = json.loads((MAIN/'results.json').read_text())
    populations = json.loads((MAIN/'prepared/populations.json').read_text())
    records, previous, cases = plan_extension(original_protocol, original_results, populations)
    sources = source_hashes()
    protocol = dict(
        extension='Post hoc predeclared extension after observing original accuracy/budget tradeoff',
        included_in_original_main_summary=False,
        original_cells_retained='All 640 original cases, without outcome-based selection',
        baseline_risk=str(ERROR), trials_per_cell=TRIALS, independent_seed_blocks=4,
        sampling='For each category independently, Bin(1,1-exp(-m*p_i)); sum gives D. Binary64 numerical sampling.',
        decision='Reject iff D>n; lower=min(max(D,1),n)',
        events='binary uses original exact Yes/No promise; short iff lower<min(eff,n); over iff lower>min(S,n); joint is their union',
        budget='certified_count_budget(n,epsilon,1/100), Arb-certified Poisson left-tail bound',
        stopping='Fixed 2000 trials for every case; no interim-dependent stopping or tuning',
        python=platform.python_version(), numpy=np.__version__,
        source_sha256=sources,
        original_protocol_sha256=digest(MAIN/'protocol.json'),
        original_results_sha256=digest(MAIN/'results.json'),
        populations_sha256=digest(MAIN/'prepared/populations.json'), cases=cases,
    )
    path = args.out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Existing frozen extension differs; choose another output directory')
    else:
        dump(path, protocol)
    dump(args.out/'source_hashes.json', sources)
    print(f'Frozen 1% count-budget extension: {len(cases)} cells x {TRIALS} trials', flush=True)
    results = []
    for index, case in enumerate(cases):
        cell = args.out/'cells'/f"{case['id']}.json"
        if cell.exists():
            result = json.loads(cell.read_text())
            if digest(args.out/'raw'/f"{case['id']}.npz") != result['raw_sha256']:
                raise ValueError(f'Raw checksum mismatch: {case["id"]}')
        else:
            result = run_case(case, records[case['population_id']], previous[case['id']], args.out)
        results.append(result)
        if index % 40 == 0 or index == len(cases)-1:
            print(f'{index+1}/{len(cases)} failures='
                  f'{sum(result["stats"]["joint"] for result in results)}', flush=True)
    dump(args.out/'results.json', results)
    summary = summarize(results)
    dump(args.out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
