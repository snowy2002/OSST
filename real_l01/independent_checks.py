"""Independent finite implementation audit for the unchanged C=70,L0=1 statistic.

This checks arithmetic and formulas, not a distribution-free probability theorem.
The positive closed-form expansion is independent of reference_weights' Chebyshev
recurrence. The original paper's real Poisson mean is retained.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flint import arb, ctx, fmpq
import numpy as np
from benchmark_core import Profiles
from followup.core import exact_output, scale_intervals
from paper_reference import reference_weights


def rational(value):
    q = Fraction(value)
    return arb(fmpq(q.numerator, q.denominator))


def closed_weights(A, B, degree):
    """Return f_l using the exact finite expansion of T_k(1+z)."""
    z = 2 * A / (B - A)
    beta = 2 / (B - A)
    coefficients = [
        rational(Fraction(degree * 2**r * math.factorial(degree + r - 1),
                          math.factorial(degree - r) * math.factorial(2*r)))
        for r in range(degree + 1)
    ]
    normalizer = sum((c * z**r for r, c in enumerate(coefficients)), arb(0))
    weights = [arb(0)]
    for ell in range(1, degree + 1):
        derivative = sum((coefficients[r] * math.comb(r, ell) * z**(r-ell)
                          for r in range(ell, degree + 1)), arb(0))
        magnitude = beta**ell * math.factorial(ell) * derivative / normalizer
        weights.append(1 + (-1)**(ell + 1) * magnitude)
    return tuple(weights)


def independent_geometry(n):
    L = 1 + arb(n).log()
    J = int(((256 * L**2 / 70).log() / arb(2).log()).floor().unique_fmpz())
    scales = []
    for j in range(J + 1):
        A, B = 70 * arb(2)**j / (8 * L), 64 * L
        delta = Fraction(1, 2**(j+2))
        degree = int(((B / A).sqrt() * rational(2 / delta).log())
                     .ceil().unique_fmpz())
        scales.append((j, A, B, delta, degree))
    return L, scales


def output_checks(implementation):
    """Check valid fingerprints with integer and noninteger score boundaries."""
    checked, skipped = 0, []
    for n in (1, 2, 3, 7, 8, 10, 50, 100):
        try:
            plan = implementation.parameters('70', 1, n, Fraction('0.9'), 512)
        except Exception as exc:
            skipped.append(dict(n=n, error=str(exc)))
            continue
        with ctx.workprec(512):
            _, scales = independent_geometry(n)
            weights = [closed_weights(A, B, k) for _, A, B, _, k in scales]
            D_values, low_values = [], []
            for D in sorted(set((0, 1, max(0, n-1), n, n+1))):
                D_values.append(D)
                low_values.append([0]*plan.degree)
            for ell in (1, 2, 3, plan.degree):
                for D in (1, n, n+1):
                    low = [0]*plan.degree
                    low[ell-1] = D
                    D_values.append(D)
                    low_values.append(low)
            profiles = Profiles(np.asarray(D_values, dtype=np.int64),
                                np.asarray(low_values, dtype=np.int64))
            interval_lo, interval_hi = scale_intervals(profiles, plan)
            for idx, (D, low) in enumerate(zip(D_values, low_values)):
                score = arb(D)
                for (_, _, _, delta, k), w in zip(scales, weights):
                    raw = arb(D) + sum((low[ell-1] * (w[ell]-1)
                                       for ell in range(1, k+1)), arb(0))
                    score = score.max((raw-rational(delta)*n)/(1+rational(delta)))
                assert arb(float(interval_lo[idx].max())) <= score, (n, D, low)
                assert arb(float(interval_hi[idx].max())) >= score, (n, D, low)
                decision = 'No' if score > n else 'Yes'
                if score > n:
                    lower = n
                elif score <= 1:
                    lower = 1
                else:
                    integer = score.ceil().unique_fmpz()
                    assert integer is not None, (n, D, low)
                    lower = max(1, min(n, int(integer)))
                actual = exact_output(D, low, plan)
                assert (actual['decision'], actual['lower']) == (decision, lower), (n, D, low)
                checked += 1
    return dict(fingerprints_checked=checked, skipped=skipped,
                cases='Empty, high-frequency D=1,n-1,n,n+1; F1,F2,F3,FK-only profiles',
                oracle='Independent closed-form coefficients and Arb ceil/threshold',
                tested_functions='followup.core.exact_output and scale_intervals')


def run(module_name):
    implementation = importlib.import_module(module_name)
    coefficient_count = 0
    polynomial_count = 0
    scale_count = 0
    parameter_failures = []
    geometry_failures = []
    endpoint_above_one = []
    maximum_degree = 0
    sampled = []
    with ctx.workprec(512):
        for n in range(1, 101):
            L, expected = independent_geometry(n)
            try:
                actual_L, actual = implementation.geometry('70', 1, n, 512)
            except Exception as exc:
                geometry_failures.append(dict(n=n, error=str(exc)))
                continue
            assert actual_L.overlaps(L)
            assert len(expected) == len(actual), n
            for (j, A, B, delta, k), scale in zip(expected, actual):
                assert (scale.j, scale.k, scale.delta) == (j, k, delta), (n, j)
                assert scale.A.overlaps(A) and scale.B.overlaps(B), (n, j)
                independent = closed_weights(A, B, k)
                recurrence = reference_weights(A, B, k)
                for ell, (u, v) in enumerate(zip(independent, recurrence)):
                    assert u.overlaps(v), (n, j, ell)
                    if ell:
                        assert (u - 1).overlaps(scale.corrections[ell-1]), (n, j, ell)
                        assert (-1)**(ell+1) * (u - 1) > 0, (n, j, ell)
                    coefficient_count += 1
                alpha, beta = (B+A)/(B-A), 2/(B-A)
                for y in (A/2, A, (A+B)/2, B, 2*B):
                    expansion = sum(((independent[ell] - 1) * y**ell /
                                     math.factorial(ell) for ell in range(k+1)), arb(0))
                    direct = -(alpha-beta*y).chebyshev_t(k) / alpha.chebyshev_t(k)
                    assert expansion.overlaps(direct), (n, j, str(y))
                    polynomial_count += 1
                assert 0 < A and A <= B/2
                assert (B*(B-A)).sqrt() >= k
                assert 1 / alpha.chebyshev_t(k) <= rational(delta**2/2)
                scale_count += 1
                maximum_degree = max(maximum_degree, k)
            if n in (1, 2, 3, 7, 8, 10, 50, 100):
                sampled.append(dict(n=n, J=len(actual)-1, degrees=[s.k for s in actual]))
            for eps_text in ('0.001', '0.01', '0.1', '0.5', '0.9'):
                epsilon = Fraction(eps_text)
                mean = 70 * n / (rational(epsilon) * L)
                endpoint = 64 * L / mean
                if endpoint > 1:
                    endpoint_above_one.append(dict(n=n, epsilon=eps_text, b=float(endpoint)))
                try:
                    plan = implementation.parameters('70', 1, n, epsilon, 512)
                    assert plan.m.overlaps(mean), (n, epsilon)
                except Exception as exc:
                    parameter_failures.append(dict(n=n, epsilon=eps_text,
                                                   exception=type(exc).__name__, error=str(exc)))
    outputs = output_checks(implementation)
    return dict(
        status='passed_formula_checks',
        scope='Finite implementation audit; no distribution-free success guarantee is certified.',
        module=module_name, precision_bits=512, C=70, L0=1,
        n_range=[1, 100], epsilons=['0.001', '0.01', '0.1', '0.5', '0.9'],
        parameter_grid_size=500, scales_checked=scale_count,
        coefficients_checked=coefficient_count, polynomial_values_checked=polynomial_count,
        maximum_degree=maximum_degree, geometry_failures=geometry_failures,
        parameter_failures=parameter_failures, b_above_one=endpoint_above_one,
        output_checks=outputs,
        representative_geometries=sampled,
        formula_notes={
            'mean': 'm=70*n/(epsilon*(1+log(n))), real-valued, no ceiling',
            'J': 'floor(log2(256*L^2/70))',
            'scaled_interval': 'A_j=70*2^j/(8*L); B=64*L',
            'degree': 'ceil(sqrt(B/A_j)*log(2/delta_j))',
            'correction': '(Z_j-delta_j*n)/(1+delta_j)',
            'score': 'max(D, max_j corrected_j)',
            'lower_bound_output': 'min(max(ceil(score),1),n)',
            'binary_output': 'Yes if score<=n, No otherwise',
            'b_le_1': 'Old Lemma 2.3 premise; not needed to define the polynomial statistic',
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='parameter_search.algorithm')
    parser.add_argument('--output', type=Path,
                        default=Path('results_real_l01_20261104/implementation_checks.json'))
    args = parser.parse_args()
    result = run(args.module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result[key] for key in (
        'status', 'module', 'scales_checked', 'coefficients_checked',
        'polynomial_values_checked', 'maximum_degree', 'geometry_failures',
        'parameter_failures')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
