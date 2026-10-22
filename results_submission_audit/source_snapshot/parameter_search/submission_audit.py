"""Read-only submission audit of frozen experiments; writes a separate report dataset.

Run from repository root: .venv/bin/python -m parameter_search.submission_audit
No new Monte Carlo trials are generated. All intervals describe the saved grid.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import csv
import hashlib
import json
import math
import platform

import numpy as np
import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results_parameter_search'
OUT = ROOT/'results_submission_audit'


def require(condition, message):
    if not bool(condition):
        raise ArithmeticError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def cp(errors, trials, alpha):
    return 1.0 if errors == trials else float(beta.isf(alpha, errors+1, trials-errors))


def main():
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader((SOURCE/'experiments.csv').open()))
    N = len(rows); K = sum(r['truth'] != 'Gap' for r in rows)
    index = {(r['cell'], r['C'], int(r['L0'])): r for r in rows}
    require(len(index) == N, 'Duplicate CSV method rows')
    protocol = json.loads((SOURCE/'experiment_protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        require(digest(ROOT/path) == expected, f'Working source changed: {path}')
        require(digest(SOURCE/'source_snapshot'/path) == expected, f'Snapshot changed: {path}')
    for name, expected in protocol['data_sha256'].items():
        require(digest(ROOT/'data/c1025/prepared'/name) == expected, f'Data changed: {name}')

    derived, paired, seen = [], [], set()
    for path in sorted((SOURCE/'cells').glob('*.json')):
        cell = json.loads(path.read_text())
        require(digest(SOURCE/cell['raw_file']) == cell['raw_sha256'], f'Raw changed: {path.name}')
        with np.load(SOURCE/cell['raw_file']) as raw:
            for r in cell['rows']:
                key = (r['cell'], r['C'], r['L0']); seen.add(key)
                record = index[key]
                method = f"{r['C']}_{r['L0']}"
                D = raw[f"D_{r['C']}"]
                reject = raw[f'{method}_reject']
                coverage = raw[f'{method}_over'] | raw[f'{method}_short']
                promised = r['truth'] != 'Gap'
                binary = (reject if r['truth'] == 'Yes' else ~reject) if promised else np.zeros(len(D), dtype=bool)
                joint = binary | coverage
                count_binary = ((D > r['n']) if r['truth'] == 'Yes' else (D <= r['n'])) if promised else None
                count_bound = np.clip(D, 1, r['n'])
                count_coverage = (count_bound > r['S']) | (count_bound < min(r['n'], r['effective']))
                require(int(coverage.sum()) == r['coverage_errors'], 'Coverage count mismatch')
                if promised:
                    require(int(binary.sum()) == r['errors'], 'Binary count mismatch')
                    require(math.isclose(cp(r['errors'], r['trials'], .05/N),
                                        float(record['errors_upper95_simultaneous']), abs_tol=1e-12),
                            'Existing binomial interval mismatch')
                require(int((reject & (D <= r['n'])).sum()) == r['extra_rejects'], 'Additional rejection mismatch')
                derived.append(dict(cell=r['cell'], suite=r['suite'], scenario=r['scenario'],
                    C=r['C'], L0=r['L0'], n=r['n'], epsilon=r['epsilon'], truth=r['truth'],
                    trials=len(D), binary_errors=int(binary.sum()) if promised else None,
                    coverage_errors=int(coverage.sum()), joint_errors=int(joint.sum()),
                    joint_upper95_grid=cp(int(joint.sum()), len(D), .05/N),
                    binary_upper95_two_families=cp(int(binary.sum()), len(D), .05/(N+K)) if promised else None,
                    coverage_upper95_two_families=cp(int(coverage.sum()), len(D), .05/(N+K)),
                    count_binary_errors=int(count_binary.sum()) if promised else None,
                    count_coverage_errors=int(count_coverage.sum()), extra_rejects=r['extra_rejects'],
                    improved_lower_outputs=r['improvements'], ambiguous_integer_outputs=r['nonunique_bounds']))
            # Same fingerprints and monotonic max construction imply a paired gain in [0,n].
            candidate = [r for r in cell['rows'] if r['C'] == '70' and r['L0'] == 14
                         and r['suite'] in ['synthetic', 'natural', 'lifted']]
            if candidate:
                r = candidate[0]; n = r['n']
                low = np.maximum(0, (raw['70_14_lower_lo']-raw['70_16_lower_hi'])/n)
                high = np.minimum(1, (raw['70_14_lower_hi']-raw['70_16_lower_lo'])/n)
                require(np.all(low <= high), 'Paired gain enclosure invalid')
                paired.append(dict(cell=r['cell'], suite=r['suite'], scenario=r['scenario'], n=n,
                                   epsilon=r['epsilon'], trials=len(low),
                                   mean_gain_over_n_lower=float(low.mean()),
                                   mean_gain_over_n_upper=float(high.mean())))
    require(seen == set(index), 'CSV and raw cell inventory differ')
    for p in paired:
        # One-sided Hoeffding; family includes all paired main-grid cells, not selected winners.
        p['expected_gain_over_n_lower95_grid'] = max(0.0, p['mean_gain_over_n_lower']-
                math.sqrt(math.log(len(paired)/.05)/(2*p['trials']))-1e-12)
    write_csv('reconstructed_events.csv', derived)
    write_csv('paired_gain_intervals.csv', paired)

    rec = [r for r in derived if r['C'] == '70' and r['L0'] == 14]
    inventory = []
    for suite in sorted({r['suite'] for r in rec}):
        rr = [r for r in rec if r['suite'] == suite]
        inventory.append(dict(suite=suite, cells=len(rr), labels=dict(Counter(r['truth'] for r in rr)),
             coverage_trials=sum(r['trials'] for r in rr),
             promised_binary_trials=sum(r['trials'] for r in rr if r['truth'] != 'Gap'),
             polynomial_trials=sum(int(index[(r['cell'],'70',14)]['trials']) for r in rr
                                   if index[(r['cell'],'70',14)]['branch']=='chebyshev'),
             binary_errors=sum(r['binary_errors'] or 0 for r in rr),
             joint_errors=sum(r['joint_errors'] for r in rr),
             count_binary_errors=sum(r['count_binary_errors'] or 0 for r in rr),
             count_coverage_errors=sum(r['count_coverage_errors'] for r in rr),
             extra_rejects=sum(r['extra_rejects'] for r in rr),
             improved_lower_outputs=sum(r['improved_lower_outputs'] for r in rr),
             ambiguous_integer_outputs=sum(r['ambiguous_integer_outputs'] for r in rr)))
    specs = {r['dataset']:r for r in json.loads((ROOT/'data/c1025/datasets.json').read_text())}
    natural = [r for r in rows if r['C']=='70' and r['L0']=='14' and r['suite']=='natural']
    feasibility = [dict(dataset=r['scenario'], n=int(r['n']), epsilon=r['epsilon'],
                        mean=float(r['mean']), available_records=specs[r['scenario']]['N'],
                        mean_over_records=float(r['mean'])/specs[r['scenario']]['N']) for r in natural]
    write_csv('natural_budget_feasibility.csv', feasibility)
    power = []
    for target in [.0009, .000057]:
        power.append(dict(target=target,
                          zero_error_trials_pointwise95=math.ceil(math.log(.05)/math.log1p(-target)),
                          zero_error_trials_grid95=math.ceil(math.log(.05/N)/math.log1p(-target))))
    collision = []
    for n in [16,100,10000,1000000]:
        root = math.isqrt(4*n)
        m = root+(root*root != 4*n)+1
        alt = Fraction(m*(m-1),2)*(Fraction(1,n**3)+Fraction(1,n**4))
        require(alt < Fraction(1,3), 'Collision example alternative bound')
        require(m*(m-1) >= 4*n, 'Collision example null exponent')
        collision.append(dict(n=n, epsilon_exact=str(1-Fraction(1,n*n)), samples=m,
                     null_error_bound=math.exp(-m*(m-1)/(2*n)),
                     alternative_error_bound_exact=str(alt), alternative_error_bound=float(alt)))
    save('collision_boundary_examples.json', dict(passed=True, examples=collision,
         scope='Numerical examples of the analytic collision-test lemma in theory_scope.tex; not simulation.'))

    summary = dict(passed=True, mode='post-hoc read-only audit; no additional Monte Carlo trials',
        csv_method_rows=N, promised_method_rows=K, raw_population_cells=len(list((SOURCE/'cells').glob('*.json'))),
        C70_L14=dict(coverage_trials=sum(r['trials'] for r in rec),
                    promised_binary_trials=sum(r['trials'] for r in rec if r['truth']!='Gap'),
                    joint_errors=sum(r['joint_errors'] for r in rec),
                    max_joint_upper95_grid=max(r['joint_upper95_grid'] for r in rec),
                    extra_rejects=sum(r['extra_rejects'] for r in rec),
                    ambiguous_integer_outputs=sum(r['ambiguous_integer_outputs'] for r in rec)),
        inventory=inventory, paired_main_cells=len(paired),
        paired_positive_sample_mean_cells=sum(r['mean_gain_over_n_lower']>1e-12 for r in paired),
        paired_positive_expected_gain_cells_hoeffding95=sum(r['expected_gain_over_n_lower95_grid']>0 for r in paired),
        inference_scope='Joint event intervals form one family over all rows. Binary and coverage '
                        'intervals additionally provided as one combined family of N+K endpoints. '
                        'Paired expectation intervals are a separate explicitly identified 95% family; '
                        'dependence across methods does not invalidate Bonferroni.',
        natural_datasets=len(specs), natural_original_records=sum(r['N'] for r in specs.values()),
        natural_support_min=min(r['S'] for r in specs.values()), natural_support_max=max(r['S'] for r in specs.values()),
        natural_cells=len(feasibility), natural_mean_budget_within_records=sum(r['mean_over_records']<=1 for r in feasibility),
        rare_event_power=power,
        sources_sha256={p:digest(ROOT/p) for p in ['parameter_search/submission_audit.py',
                  'results_parameter_search/experiments.csv','results_parameter_search/experiment_protocol.json',
                  'data/c1025/datasets.json','results_parameter_search/conclusions.json']},
        audit_environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                               note='Environment recorded at audit time; not proof of historical execution settings.'))
    save('audit_summary.json', summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
