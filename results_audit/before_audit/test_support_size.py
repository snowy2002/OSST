"""针对统计含义、数值公式和数据清洗的测试，不依赖网络和随机通过阈值。"""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_chebyt
from scipy.stats import binom, poisson

from fish_data import load_counts_csv, prepare_fish_data
from support_size import *


def test_paper_plan_matches_manuscript_and_large_n_branch():
    plan = paper_plan(1000, .1)
    assert plan["m"] == 84864619140
    assert plan["branch"] == "observed_count"
    huge = paper_plan(10**445, .1)
    assert huge["branch"] == "chebyshev" and huge["scales"]


@pytest.mark.parametrize("sampling", ["fixed", "poisson"])
def test_count_budget_is_sufficient_and_minimal(sampling):
    plan = count_plan(1000, .1, .05, sampling)
    assert plan["miss_bound"] <= .05
    assert count_miss_bound(1000, .1, plan["m"]-1, sampling) > .05
    assert plan["m"] < plan["markov_m"] / 10


def test_count_domination_against_exact_uniform_occupancy():
    # 固定样本：独立地用马尔可夫递推计算真实 D 的分布。
    for S, n, M in [(6, 2, 8), (10, 4, 20), (5, 1, 5)]:
        epsilon = (S-n) / S
        pmf = np.zeros(S+1)
        pmf[0] = 1
        for _ in range(M):
            nxt = np.zeros_like(pmf)
            for d in range(S+1):
                nxt[d] += pmf[d] * d/S
                if d < S:
                    nxt[d+1] += pmf[d] * (1-d/S)
            pmf = nxt
        assert pmf[:n+1].sum() <= count_miss_bound(n, epsilon, M, "fixed") + 1e-14
        # 泊松均匀分布下 D 精确服从 Bin(S,1-exp(-m/S))。
        actual = binom.cdf(n, S, -math.expm1(-M/S))
        assert actual <= count_miss_bound(n, epsilon, M, "poisson") + 1e-14


def test_weights_degree_one_closed_form():
    A, B = .1, 7.
    assert chebyshev_weights(A, B, 1) == pytest.approx([0, 1+2/(A+B)])


@pytest.mark.parametrize("k", [1, 3, 8, 12])
def test_poisson_expectation_identity(k):
    A, B = .15, 20.
    weights = chebyshev_weights(A, B, k)
    for lam in [0, .03, A, (A+B)/2, B, 30.]:
        # 使用概率质量直接求 E f(N)，与独立的 Chebyshev 求值比较。
        lhs = poisson.sf(k, lam) + sum(weights[l]*poisson.pmf(l, lam) for l in range(1, k+1))
        rhs = 1 - math.exp(-lam)*eval_chebyt(k, (B+A-2*lam)/(B-A))/eval_chebyt(k, (B+A)/(B-A))
        assert lhs == pytest.approx(rhs, abs=1e-11)


def test_high_precision_and_aggregation():
    a = chebyshev_weights(.02, 8, 12, 80)
    b = chebyshev_weights(.02, 8, 12, 150)
    assert a == pytest.approx(b, rel=1e-14)
    scale = make_scales(1000, .1, 2530)[0]
    counts = [0, 1, 1, 2, 4, 8, 9, 1000000000000]
    fp = Fingerprint.from_counts(counts)
    expected = math.fsum(scale.weights[x] if x <= scale.k else 1 for x in counts)
    assert raw_score(fp, scale) == pytest.approx(expected)
    assert fp.total == sum(counts)


def test_practical_expectation_envelope_and_constraints():
    scales = make_scales(1000, .1, 2530)
    assert scales[0].rho > scales[0].delta  # 截断后不可继续假设 rho<=delta。
    for s in scales:
        assert s.k <= math.sqrt(s.B * (s.B-s.A))
        assert max(map(abs, s.weights)) <= 20
        lam = np.r_[0, np.geomspace(1e-8, 2530, 2000)]
        q = poisson.sf(s.k, lam)
        for l in range(1, s.k+1):
            q += s.weights[l] * poisson.pmf(l, lam)
        assert np.max(q) <= 1 + s.rho + 1e-10
        assert np.min(q - (1-s.rho)*np.minimum(lam/s.A, 1)) >= -1e-10


def test_small_A_stability_and_underflow_rejection():
    scales = make_scales(1000, 1e-20, 10)
    assert scales and all(math.isfinite(s.rho) for s in scales)
    with pytest.raises(ValueError, match="下溢"):
        make_scales(1000, 1e-320, 1)


def test_profiles_ignore_unobserved_and_labels_are_irrelevant():
    fp = Fingerprint.from_labels(["A", "A", "B", "C", "C", "C"])
    assert fp.frequencies == {1: 1, 2: 1, 3: 1}
    assert fp == Fingerprint.from_counts([0, 3, 1, 2, 0])


def test_no_m_equals_M_substitution_or_forced_yes():
    fp = Fingerprint.from_counts([2, 3])
    with pytest.raises(ValueError):
        count_test(fp, 3, .1, sampling="poisson")
    result = count_test(fp, 3, .1, sampling="poisson", m=100)
    assert result["budget"] == 100 and fp.total == 5
    assert count_test(fp, 3, .1)["decision"] == "Inconclusive"
    assert count_test(fp, 1, .1)["decision"] == "No"
    assert clipped_paper_lower_bound(50, 10) == 10
    assert clipped_paper_lower_bound(-10, 10) == 1


def test_poissonized_fixed_pool_preserves_budget_and_mean():
    mean = poisson_mean_for_pool(300)
    assert poisson.sf(300, mean) <= .005
    assert poisson.sf(300, mean + .001) > .005
    seed = 87
    expected_M = int(np.random.default_rng(seed).poisson(mean))
    result = poissonized_fixed_test([120, 120, 60], 10, .1, np.random.default_rng(seed))
    assert result["status"] == "ok"
    assert result["m"] == mean and result["used_M"] == expected_M
    assert result["used_M"] <= result["pool_N"]
    assert result["support_lower_bound"] >= 3


def test_poissonized_pool_overflow_does_not_retry_or_truncate():
    class OverflowRNG:
        def __init__(self):
            self.calls = 0

        def poisson(self, mean):
            self.calls += 1
            return 31

        def multivariate_hypergeometric(self, *args):
            raise AssertionError("overflow 后不能继续子抽样")

    rng = OverflowRNG()
    result = poissonized_fixed_test([10, 20], 5, .1, rng)
    assert rng.calls == 1
    assert result["status"] == "overflow" and result["used_M"] == 0
    assert result["decision"] == "Inconclusive"
    assert result["score"] is None and result["support_lower_bound"] == 2


def test_support_and_chao_edge_cases():
    assert delta_n([.5, .3, .2], 2) == pytest.approx(.2)
    assert effective_support([.5, .3, .2], .2) == 2
    assert effective_support([.5, .3, .2], .19) == 3
    assert chao1(Fingerprint({1: 3})) == 6
    assert chao1(Fingerprint({})) == 0
    assert describe_sample(Fingerprint({}))["good_turing_coverage"] is None
    for p in [[.1, .1], [-.1, 1.1], [float("nan")], []]:
        with pytest.raises(ValueError):
            normalize_probabilities(p)


@pytest.mark.parametrize("bad", [[-1], [1.5], [float("nan")], [True]])
def test_bad_counts_rejected(bad):
    with pytest.raises(ValueError):
        Fingerprint.from_counts(bad)


def test_csv_aggregates_only_positive_integer_individual_counts(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("species,count\nA,3\nB,0\nA,2\nC,1\n")
    assert load_counts_csv(path, count_column="count") == {"A": 5, "C": 1}
    path.write_text("species,count\nA,1.5\n")
    with pytest.raises(ValueError):
        load_counts_csv(path, count_column="count")


def test_fish_reference_and_no_crayfish(tmp_path):
    # 使用随项目保存的固定版本，不需要联网。
    local = Path(__file__).parent / "data" / "InchLake1.csv"
    (tmp_path / "InchLake1.csv").write_bytes(local.read_bytes())
    records = prepare_fish_data(tmp_path)
    assert len(records) == 4887
    assert len({x["species"] for x in records}) == 9
    assert not any(x["species"] == "Rusty Crayfish" for x in records)
    seine = [x for x in records if x["year"] == 2007 and x["gear"] == "seine"]
    assert len(seine) == 1180
    assert len({x["species"] for x in seine}) == 8
    # 相同网次/种类/体长的行不能无依据去重。
    assert len({x["record_id"] for x in records}) == len(records)
