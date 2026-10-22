"""Independent reconstruction of saved decisions, truth and confidence bounds."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from flint import arb, ctx
from scipy.stats import beta, poisson

from benchmark_core import stable_rng
from c1025.data import ROOT, DATA, save, sha
from c7016.algorithm import parameters, arb_score, ball, score_intervals
from c7016.populations import Population, sample_profiles

COUNTS = Counter()


def require(condition, category):
    if not condition:
        raise ArithmeticError(category)
    COUNTS[category] += 1


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def raw_audit(out):
    protocol = json.loads((out/'protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        require(sha(ROOT/name) == expected, 'frozen source matches working source')
        require(sha(out/'source_snapshot'/name) == expected, 'frozen source snapshot hash')
    for name, expected in protocol['data_hashes'].items():
        require(sha(DATA/'prepared'/name) == expected, 'natural data hash')
    raw_paths = 0; full_scores = 0
    for suite in ['synthetic', 'natural', 'lifted']:
        summary = rows(out/f'{suite}.csv')
        by_key = {(r['cell'], r['method']): r for r in summary}
        K = sum(r['errors'] != '' for r in summary)
        for path in sorted((out/'cells').glob(f'{suite}_*.json')):
            cell = json.loads(path.read_text()); spec = cell['population']
            pop = Population(tuple(spec['multiplicities']), tuple(Fraction(p) for p in spec['probabilities_exact']))
            require(sha(out/cell['raw_file']) == cell['raw_sha256'], 'raw fingerprint hash')
            raw_paths += 1
            with np.load(out/cell['raw_file']) as raw:
                for row in cell['rows']:
                    actual = by_key[(row['cell'], row['method'])]
                    n = row['n']; index = row['raw_index']; T = row['trials']
                    label, tail, eff = pop.truth(n, row['epsilon'])
                    require((label, str(tail), eff) == (row['truth'], row['delta_n_exact'], row['effective_support']), 'exact promise truth')
                    D = raw[f'observed_{index}']; reject = raw[f'reject_{index}']
                    over = raw[f'over_{index}']; short = raw[f'short_{index}']
                    low = raw[f'lower_low_{index}']; high = raw[f'lower_high_{index}']
                    slo = raw[f'score_low_{index}']; shi = raw[f'score_high_{index}']
                    require(len(D) == T and len(reject) == T, 'saved trial count')
                    require(np.all(slo <= shi) and np.all(slo >= D), 'valid statistic enclosure')
                    require(np.all((1 <= low) & (low <= high) & (high <= n)), 'valid integer-output enclosure')
                    # Unresolved intervals were refined before being stored. All
                    # decisions whose interval already separates the threshold
                    # must agree without relying on the recorded decision.
                    require(np.all(reject[slo > n]) and not np.any(reject[shi <= n]), 'decision agrees with enclosure')
                    if label != 'Gap':
                        count = int(np.sum(reject if label == 'Yes' else ~reject))
                        require(count == row['errors'] == int(actual['errors']), 'binary errors reconstructed')
                        ub = 1. if count == T else float(beta.isf(.05/K, count+1, T-count))
                        require(abs(ub-float(actual['error_upper95_simultaneous'])) < 1e-14, 'binary simultaneous confidence bound')
                    else:
                        require(row['errors'] is None and actual['errors'] == '', 'no invented Gap error label')
                    failures = int(np.sum(over | short))
                    require(failures == row['lower_errors'] == int(actual['lower_errors']), 'lower-bound errors reconstructed')
                    ub = 1. if failures == T else float(beta.isf(.05/len(summary), failures+1, T-failures))
                    require(abs(ub-float(actual['lower_error_upper95_simultaneous'])) < 1e-14, 'lower simultaneous confidence bound')
                    require(np.all(over[low > pop.support]) and not np.any(over[high <= pop.support]), 'upper coverage agrees with enclosure')
                    target = min(eff, n)
                    require(np.all(short[high < target]) and not np.any(short[low >= target]), 'lower coverage agrees with enclosure')
                    require(row['rejects'] == int(reject.sum()), 'reject count')
                    if row['method'] == 'C70_L16':
                        plan = parameters(n, row['epsilon'])
                        require(raw['low'].shape == (T, plan.degree), 'all full-degree frequencies retained')
                        require(cell['plan']['mean_rounding'] == 'real expression; no integer ceiling', 'real mean recorded')
                        require(plan.branch == row['branch'], 'branch matches theorem')
                        if plan.scales:
                            # An additional independently chosen middle profile,
                            # beyond the endpoints audited in the runner.
                            i = T//3
                            exact = arb_score(D[i], raw['low'][i], n, bits=1024)
                            require(exact >= arb(float(slo[i])) and exact <= arb(float(shi[i])), '1024-bit saved-profile cross-check')
                            require(bool(exact > n) == bool(reject[i]), '1024-bit decision cross-check')
                            full_scores += 1
                    else:
                        if row['branch'] == 'observed_count':
                            require(np.array_equal(reject, D > n), 'literal count decision')
                for block in cell['blocks']:
                    i = next(r['raw_index'] for r in cell['rows'] if r['method'] == block['method'])
                    start = block['block']*block['trials']; stop = start+block['trials']
                    require(int(np.sum(raw[f'over_{i}'][start:stop] | raw[f'short_{i}'][start:stop])) == block['lower_errors'], 'seed-block coverage count')
        complete = json.loads((out/f'{suite}_complete.json').read_text())
        require(len({r['cell'] for r in summary}) == complete['base_cells'], 'suite cell count')
        require(sum(int(r['trials']) for r in summary if r['method'] not in ['count_same_budget', 'scale0_same_budget'])
                == complete['independent_sampling_paths'], 'independent path accounting')
    for path in sorted((out/'cells').glob('amplification_*.json')):
        cell = json.loads(path.read_text())
        require(sha(out/cell['raw_file']) == cell['raw_sha256'], 'amplification raw hash')
        with np.load(out/cell['raw_file']) as raw:
            direct = np.concatenate([raw[f'copy_reject_{b}'].sum(axis=1) > protocol['design']['amplification_repeats']//2
                                     for b in range(protocol['design']['seed_blocks'])])
            require(np.array_equal(direct, raw['C70_L16_x27_reject']), 'independent-copy majority reconstructed')
            for r in cell['rows']:
                failure = raw[f'{r["method"]}_lower_errors']
                require(int(failure.sum()) == r['lower_errors'], 'amplified coverage reconstructed')
                if r['truth'] != 'Gap':
                    reject = raw[f'{r["method"]}_reject']
                    errors = reject if r['truth'] == 'Yes' else ~reject
                    require(int(errors.sum()) == r['errors'], 'amplified binary errors reconstructed')
        raw_paths += 1
    summary = rows(out/'amplification.csv')
    complete = json.loads((out/'amplification_complete.json').read_text())
    require(sum(int(r['trials'])*int(r['repeats']) for r in summary)
            == complete['independent_sampling_paths'], 'independent amplified path accounting')
    require(sum(int(r['trials']) for r in summary if r['method'].startswith('C70_L16'))
            == complete['primary_trials'], 'aggregated outputs not confused with individual runs')
    return dict(raw_files=raw_paths, additional_full_arb_scores=full_scores)


def sampler_audit(out):
    """Checks independent of the multinomial construction and its normalization."""
    result = []
    with ctx.workprec(512):
        for lam in ['0.00000000000004', '0.000001', '0.01', '0.1', '1', '5', '20', '80', '160', '500']:
            value = float(lam); exact = ball(Fraction.from_float(value))
            max_error = 0.
            for k in [0, 1, 2, 5, 20, 80, 160]:
                oracle = (-exact).exp()*exact**k/math.factorial(k)
                approximate = float(poisson.pmf(k, value))
                error = abs(float(oracle)-approximate)
                require(error <= 3e-13*max(float(oracle), 1e-300)+1e-310, 'Poisson PMF against Arb')
                max_error = max(max_error, error)
            result.append(dict(kind='poisson_probability', mean=value, max_absolute_error=max_error))
    # A huge, ultra-diffuse population is sensitive to erroneously assigning
    # the multinomial normalization residual to a tiny-probability last bin.
    p = Population((10**15,), (Fraction(1, 10**15),)); T = 50000
    fp, defect = sample_profiles(p, 40., T, 12, stable_rng(70169921))
    require(abs(fp.observed.mean()-40) < .15, 'huge diffuse sampler mean')
    require(abs(fp.observed.var()-40) < 1., 'huge diffuse sampler variance')
    require(np.max(fp.low[:, 1:]) <= 1, 'no normalization-induced high-frequency ghosts')
    result.append(dict(kind='huge_diffuse', support=p.support, mean=40, trials=T,
                       observed_mean=float(fp.observed.mean()), observed_variance=float(fp.observed.var()),
                       max_bin_mass_defect=defect))
    save(out/'sampler_validation.json', result)


def proof_and_tests(out):
    target = out/'proof_recheck'; target.mkdir(exist_ok=True)
    source = ROOT/'artifacts/joint_constants_submission'
    for name in ['exact_certificate.py', 'rational_interval.py', 'audit_exact.py', 'submission_proof.tex']:
        shutil.copy2(source/name, target/name)
    logs = []
    for command in [[sys.executable, 'exact_certificate.py'], [sys.executable, '-O', 'audit_exact.py']]:
        result = subprocess.run(command, cwd=target, check=True, capture_output=True, text=True)
        logs.append(result.stdout)
    (target/'recheck.log').write_text('\n'.join(logs))
    for name in ['exact_certificate.json', 'certificate_table.tex', 'audit_exact.json']:
        require((target/name).read_bytes() == (source/name).read_bytes(), 'proof recomputation byte match')
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True, capture_output=True, text=True)
    (out/'tests_final.txt').write_text(test.stdout+test.stderr)
    return dict(certificate=json.loads((target/'exact_certificate.json').read_text())['checks'],
                audit=json.loads((target/'audit_exact.json').read_text())['checks'], test_output=test.stdout.strip())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'results_c7016')
    p.add_argument('--proof-and-tests', action='store_true')
    args = p.parse_args(argv)
    start = time.perf_counter()
    detail = raw_audit(args.out); sampler_audit(args.out)
    if args.proof_and_tests:
        detail['proof_and_tests'] = proof_and_tests(args.out)
    save(args.out/'validation.json', dict(passed=True, checks=sum(COUNTS.values()), categories=COUNTS,
          **detail, seconds=time.perf_counter()-start,
          scope='Saved data and numerical/analytic certificate checks; not a proof-assistant formalization'))
    print('PASS', sum(COUNTS.values()), 'checks', detail, flush=True)


if __name__ == '__main__':
    main()
