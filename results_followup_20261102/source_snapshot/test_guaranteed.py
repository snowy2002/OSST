"""组合检验的保证组件、分阶段联合模拟、数值证书与数据边界。"""
from fractions import Fraction
import itertools
import math

from flint import arb, ctx, fmpq
import numpy as np
import pytest
from scipy.stats import binom, poisson

from benchmark_core import Population, Profiles, stable_rng
from guaranteed import (NoCertificate, ball, certified_count_budget, make_plan, polynomial_reject,
                         run_labels, validation_cutoff, validation_gate)
from guaranteed_experiments import batch_no, decide_paths, simulate_trajectory
from paper_reference import reference_weights
from support_size import Fingerprint


@pytest.mark.parametrize("n,epsilon", [(1, Fraction(1, 4)), (1000, Fraction(1, 10)), (1000000, Fraction(1, 20))])
def test_count_fallback_is_certified(n, epsilon):
    error = Fraction(1, 40)
    m, interval = certified_count_budget(n, epsilon, error)
    with ctx.workprec(256):
        assert arb(interval) <= ball(error)
        assert (m*ball(epsilon)).gamma_upper(n+1, regularized=1).overlaps(arb(interval))
    assert poisson.cdf(n, m*float(epsilon)) <= float(error)


@pytest.mark.parametrize("r,epsilon,error", [(40, Fraction(1, 10), Fraction(1, 40)),
                                            (100, Fraction(1, 4), Fraction(1, 100)),
                                            (7, Fraction(1, 10), Fraction(1, 100))])
def test_validation_threshold_against_exact_binomial_fraction(r, epsilon, error):
    k, interval = validation_cutoff(r, epsilon, error)
    exact = sum(Fraction(math.comb(r, j))*epsilon**j*(1-epsilon)**(r-j) for j in range(k+1))
    assert exact <= error
    assert float(exact) == pytest.approx(0 if k < 0 else binom.cdf(k, r, float(epsilon)))
    # No 时 q_A>=epsilon，左尾概率随 q 减小而增加。
    for q in [float(epsilon), (1+float(epsilon))/2, .999]:
        assert (0 if k < 0 else binom.cdf(k, r, q)) <= float(error)+1e-14


def test_error_spending_and_validation_overhead():
    for n in [10, 1000, 1000000]:
        plan = make_plan(n, "0.1")
        assert [l.m for l in plan.looks] == sorted({l.m for l in plan.looks})
        assert all(l.m < plan.final_m for l in plan.looks)
        assert sum(l.validation_r for l in plan.looks) <= plan.final_m//5
        assert plan.baseline_m <= plan.final_m
        with ctx.workprec(256):
            # 同时控制：No 的各阶段 Yes 与最终 Yes；Yes 的 No 单独分配 delta。
            bound = sum(arb(l.validation_bound) for l in plan.looks)+arb(plan.final_bound)
            assert bound <= ball(plan.delta)


def test_polynomial_certificates_bound_the_exact_formula():
    plan = make_plan(1000, "0.1")
    H = len(plan.looks)
    with ctx.workprec(256):
        for look in plan.looks:
            B = (ball(plan.config["b_factor"])*(arb(plan.n).log()+1)).min(arb(look.m))
            for cert in look.certificates:
                A = ball(Fraction(look.m, 8*plan.n)*plan.epsilon*2**cert.j)
                assert (B*(B-A)).sqrt() >= cert.degree and B <= look.m
                weights = reference_weights(A, B, cert.degree)
                rho = 1/((B+A)/(B-A)).chebyshev_t(cert.degree)
                width = max([arb(1)]+[w.upper() for w in weights])-min([arb(0)]+[w.lower() for w in weights])
                exact_threshold = plan.n*(1+rho)+width*(arb(plan.n)/2*(H*len(look.certificates)/ball(plan.delta)).log()).sqrt()
                assert arb(fmpq(cert.threshold, cert.denominator)) >= exact_threshold
                for numerator, w in zip(cert.coefficients, weights[1:]):
                    assert arb(fmpq(numerator, cert.denominator)) <= w-1


def test_integer_rejection_distinguishes_values_above_double_precision():
    n = 10**20
    cert = NoCertificate(0, 1, (1,), 2, 2*n, (.5,), float(n), "0", "1")
    assert not cert.rejects(n, {})
    assert cert.rejects(n, {1: 1})
    assert not cert.rejects(n-1, {1: 1})


def test_batch_rejection_matches_exact_scalar_interface():
    plan = make_plan(1000, "0.1")
    rng = np.random.default_rng(321)
    counts = [rng.poisson(.5, size=2000) for _ in range(100)]
    fps = [Fingerprint.from_counts(c) for c in counts]
    profiles = Profiles(np.array([f.observed for f in fps]), np.array(
        [[f.frequencies.get(l, 0) for l in range(1, plan.max_degree+1)] for f in fps]))
    for look in plan.looks:
        actual = batch_no(profiles, look, plan.n)
        expected = [polynomial_reject(fp, look, plan.n) for fp in fps]
        assert np.array_equal(actual, expected)


def test_joint_sampler_preserves_cross_time_law_and_missing_mass():
    pop = Population(np.array([3, 7, 20]), np.array([.10, .05, .0175]))
    trials = 60000
    stages = simulate_trajectory(pop, [10, 37], trials, 5, stable_rng(5917))
    assert np.all(stages[10].profiles.observed <= stages[37].profiles.observed)
    assert np.all(stages[10].missing >= stages[37].missing-1e-15)
    for m, stage in stages.items():
        q0 = np.exp(-m*pop.probabilities)
        observed_mean = float(pop.multiplicities @ (1-q0))
        observed_var = float(pop.multiplicities @ (q0*(1-q0)))
        assert abs(stage.profiles.observed.mean()-observed_mean) < 6*math.sqrt(observed_var/trials)
        mass_mean = float(pop.multiplicities @ (pop.probabilities*q0))
        mass_var = float(pop.multiplicities @ (pop.probabilities**2*q0*(1-q0)))
        assert abs(stage.missing.mean()-mass_mean) < 6*math.sqrt(mass_var/trials)
        assert np.all(stage.profiles.low.sum(axis=1) <= stage.profiles.observed)
    expected_cov = float(pop.multiplicities @ ((1-np.exp(-10*pop.probabilities))*np.exp(-37*pop.probabilities)))
    actual_cov = np.cov(stages[10].profiles.observed, stages[37].profiles.observed, ddof=0)[0, 1]
    assert actual_cov == pytest.approx(expected_cov, abs=.08)


def test_independent_validation_not_plug_in_missing_mass_and_exhaustion():
    plan = make_plan(1000, "0.1")
    result = run_labels(itertools.repeat("same"), plan)
    assert result["decision"] == "Yes" and result["reason"] == "independent_validation"
    assert result["validation_samples"] > 0
    assert result["samples_consumed"] == result["training_samples"]+result["validation_samples"]
    shortage = run_labels(["same"], plan)
    assert shortage["decision"] == "Inconclusive"
    # 门控通过时，坏的验证结果仍不许提前 Yes。
    T = 2
    profiles = {m: Profiles(np.ones(T, dtype=int), np.zeros((T, plan.max_degree), dtype=int))
                for m in {l.m for l in plan.looks}|{plan.final_m, plan.baseline_m}}
    outside = {l.m: np.full(T, l.validation_r) for l in plan.looks if l.validation_r}
    outputs = decide_paths(plan, profiles, outside)
    assert all(r == "fallback" for r in outputs["hybrid"]["reasons"])
    assert np.all(outputs["hybrid"]["training_mean"] == plan.final_m)


def test_holdout_false_accept_control_for_every_small_candidate_set():
    p = [Fraction(1, 5)]*5
    n, epsilon, r, error = 2, Fraction(1, 2), 30, Fraction(1, 40)
    assert sum(sorted(p, reverse=True)[n:]) >= epsilon
    k, _ = validation_cutoff(r, epsilon, error)
    for size in range(n+1):
        for indices in itertools.combinations(range(5), size):
            q = 1-sum(p[i] for i in indices)
            probability = sum(Fraction(math.comb(r, j))*q**j*(1-q)**(r-j) for j in range(k+1))
            assert probability <= error
