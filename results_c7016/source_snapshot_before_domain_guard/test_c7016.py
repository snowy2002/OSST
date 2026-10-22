from fractions import Fraction
import math

import numpy as np
import pytest
from flint import arb, ctx

from benchmark_core import Profiles, stable_rng
from c7016.algorithm import (parameters, score_intervals, arb_score, compare_scores,
                              test_fingerprint as run_fingerprint, compiled)
from c7016.populations import Population, sample_profiles, synthetic_cases, expected_profiles
from paper_reference import reference_parameters
from support_size import Fingerprint


def test_branch_switch_and_real_mean():
    below = parameters(3269017, '.1'); above = parameters(3269018, '.1')
    assert below.branch == 'observed_count' and above.branch == 'chebyshev'
    assert above.degree > 80 and len(above.scales) > 5
    with ctx.workprec(512):
        target = 70*arb(3269018)/(arb('0.1')*(1+arb(3269018).log()))
        assert above.m.overlaps(target)
        assert not above.m.contains(above.m.ceil())
        for s in above.scales:
            assert s.A.overlaps(70*arb(2)**s.j/(8*above.L))
            assert s.rho <= arb(str(float(s.delta*s.delta/2)))
    # Existing published baselines retain their original branch/defaults.
    assert reference_parameters(3269018, '.1').branch == 'observed_count'
    assert reference_parameters(1000, '.1').sampling_constant == 2**26


@pytest.mark.parametrize('n', [1, 1000, 3269018, 10**9, 10**12])
def test_dimensionless_weights_independent_of_epsilon(n):
    p = parameters(n, '.001'); q = parameters(n, '.25')
    assert p.scales is q.scales
    with ctx.workprec(512):
        assert (p.m/q.m).contains(250)


@pytest.mark.parametrize('n', [3269018, 10**9, 10**12])
def test_fast_enclosures_against_independent_high_precision_sums(n):
    plan = parameters(n, '.1'); rng = stable_rng(701699, n)
    low = rng.integers(0, max(2, n//plan.degree), size=(25, plan.degree), dtype=np.int64)
    # Include fingerprints dominated by a single coefficient, zero counts,
    # exact integer score, and signed-weight cancellation.
    low[0] = 0; low[1] = 0; low[1, 0] = n
    D = low.sum(axis=1)+rng.integers(0, 100, size=len(low))
    fp = Profiles(D, low)
    a, b, c, d = score_intervals(fp, plan)
    for i in range(len(D)):
        for single, left, right in [(False, a, b), (True, c, d)]:
            exact = arb_score(D[i], low[i], n, bits=1024, single=single)
            assert exact >= arb(float(left[i])) and exact <= arb(float(right[i]))
    # Every uncertain comparison must use high precision, including exact equality.
    fp0 = Profiles(np.array([n]), np.zeros((1, plan.degree), dtype=np.int64))
    decision, refined = compare_scores(fp0, n, np.array([n-1.]), np.array([n+1.]), n)
    assert decision.tolist() == [False] and refined == 1


def test_exact_natural_boundary_and_synthetic_truth():
    pop = Population.from_counts([9, 1])
    assert pop.tail(1) == Fraction(1, 10)
    assert pop.truth(1, '.1')[0] == 'No'
    assert pop.effective('.1') == 1
    for n in [1, 10, 3269018, 10**12]:
        for ep in ['.001', '.25']:
            for _, p in synthetic_cases(n, ep):
                assert sum(g*q for g, q in zip(p.multiplicities, p.probabilities)) == 1


def test_sampler_full_label_law_and_poisson_means():
    pop = Population((10, 20), (Fraction(1, 20), Fraction(1, 40)))
    m, T, K = 40, 50000, 8
    fp, defect = sample_profiles(pop, m, T, K, stable_rng(777))
    ed, el = expected_profiles(pop, m, K)
    assert abs(fp.observed.mean()-ed) < .05
    assert np.max(np.abs(fp.low.mean(axis=0)-el)) < .06
    assert defect < 1e-12
    rng = stable_rng(778)
    explicit = rng.poisson(m*np.repeat([.05, .025], [10, 20]), size=(T, 30))
    assert abs(fp.observed.mean()-(explicit > 0).sum(axis=1).mean()) < .06
    for k in range(1, K+1):
        assert abs(fp.low[:, k-1].mean()-(explicit == k).sum(axis=1).mean()) < .06


def test_public_fingerprint_and_validation():
    result = run_fingerprint(Fingerprint({2: 1001}), 1000, '.1')
    assert result['decision'] == 'No' and result['support_lower_bound'] == 1000
    assert result['M'] == 2002 and result['C0'] == 70 and result['L0'] == 16
    result = run_fingerprint(Fingerprint({1: 0}), 3269018, '.1')
    assert result['decision'] == 'Yes' and result['support_lower_bound'] == 1
    plan = parameters(3269018, '.1')
    with pytest.raises(ValueError):
        score_intervals(Profiles(np.array([1]), np.ones((1, plan.degree), dtype=np.int64)), plan)


def test_no_fake_tail_observation_count():
    pop = Population((10,), (Fraction(1, 10),))
    fp, _ = sample_profiles(pop, 1000, 10, 5, stable_rng(779))
    assert np.all(fp.observed == 10) and np.all(fp.low == 0)
    # Profiles deliberately has no total/M: counts above K are not K+1 counts.
    assert not hasattr(fp, 'total')
