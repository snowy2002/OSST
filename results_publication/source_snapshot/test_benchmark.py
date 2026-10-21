"""独立核对压缩模拟的分布、原有公式的一致性，以及真实数据审计。"""
import json
import math

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import poisson

from benchmark_core import (Population, Profiles, headtail, sample_profiles, score_profiles,
                            stable_rng, synthetic_cases)
from benchmark_data import DATA, SOURCES, retail_reason, sha256
from large_experiments import clustered_profiles, real_views
from support_size import (Fingerprint, chao1, delta_n, effective_support,
                          hoeffding_statistic, make_scales, paper_statistic)


def test_group_truth_matches_explicit_enumeration():
    pop = Population.from_counts([1, 1, 3, 4, 7, 7, 20])
    p = np.repeat(pop.probabilities, pop.multiplicities)
    for n in range(1, 10):
        assert pop.tail_mass(n) == pytest.approx(delta_n(p, n), abs=1e-14)
    for epsilon in [.001, .037, .1, .27, .8]:
        assert pop.effective_support(epsilon) == effective_support(p, epsilon)
    limited = pop.restrict_top(4)
    assert limited.support == 4
    assert np.sort(np.repeat(limited.probabilities, limited.multiplicities)) == pytest.approx(np.sort(p)[-4:]/np.sort(p)[-4:].sum())


def test_grouped_sampler_matches_analytic_occupancy_moments():
    pop = Population(np.array([3, 7, 20]), np.array([.10, .05, .0175]))
    T, K, m = 100000, 5, 37.
    profiles = sample_profiles(pop, m, T, K, stable_rng(1234, "moments"))
    lam = m*pop.probabilities
    occupancy = -np.expm1(-lam)
    means = [float(pop.multiplicities @ occupancy)]
    variances = [float(pop.multiplicities @ (occupancy*(1-occupancy)))]
    for l in range(1, K+1):
        q = poisson.pmf(l, lam)
        means.append(float(pop.multiplicities @ q))
        variances.append(float(pop.multiplicities @ (q*(1-q))))
    samples = np.column_stack((profiles.observed, profiles.low))
    assert np.all(np.abs(samples.mean(axis=0)-means) < 6*np.sqrt(np.asarray(variances)/T))
    # D 与 F1 的协方差 = sum_i q1_i * P(N_i=0)，不同类别独立。
    expected_cov = float(pop.multiplicities @ (poisson.pmf(1, lam)*np.exp(-lam)))
    actual_cov = np.cov(samples[:, :2], rowvar=False, ddof=0)[0, 1]
    assert actual_cov == pytest.approx(expected_cov, abs=.045)


def test_hundred_million_categories_do_not_collapse_support():
    pop = headtail(.8, 100000000)
    fp = sample_profiles(pop, 10000, 100, 16, stable_rng(91))
    expected = float(pop.multiplicities @ (-np.expm1(-10000*pop.probabilities)))
    assert fp.observed.mean() == pytest.approx(expected, abs=20)
    assert fp.observed.min() > 1000
    assert np.all(fp.low.sum(axis=1) <= fp.observed)


def test_vectorized_scores_match_scalar_implementation():
    data = [[1, 1, 2, 3, 13, 50], [1]*300+[2]*600+[3]*40, [0, 18, 19]]
    fps = [Fingerprint.from_counts(x) for x in data]
    batch = Profiles(np.array([fp.observed for fp in fps]),
                     np.array([[fp.frequencies.get(l, 0) for l in range(1, 17)] for fp in fps]))
    n, eps, m = 1000, .1, 2530
    scores = score_profiles(batch, n, eps, m)
    scales = make_scales(n, eps, m)
    naive = make_scales(n, eps, m, mode="naive", b_factor=1, max_degree=8)
    for i, fp in enumerate(fps):
        assert scores["bounded_multiscale"][i] == pytest.approx(hoeffding_statistic(fp, scales, n, m))
        assert scores["bounded_single_scale"][i] == pytest.approx(hoeffding_statistic(fp, scales[:1], n, m))
        assert scores["naive_chebyshev"][i] == pytest.approx(paper_statistic(fp, naive, n))
        assert scores["chao1_point"][i] == pytest.approx(chao1(fp))
    assert np.all(scores["bounded_multiscale"] >= scores["observed_count"])


@pytest.mark.parametrize("n", [1000, 10000, 100000, 1000000])
@pytest.mark.parametrize("epsilon", [.05, .1, .2])
def test_synthetic_promises(n, epsilon):
    for name, pop in synthetic_cases(n, epsilon):
        expected = "Yes" if name.endswith("null") else "Gap" if name.endswith("gap") else "No"
        assert pop.truth(n, epsilon) == expected
        if name == "multiscale_boundary":
            assert pop.tail_mass(n) == pytest.approx(1.01*epsilon, rel=1e-10)


def test_stress_sampler_keeps_poisson_total_and_changes_dependence():
    # 两个纯类别组，等质量。大块相关采样只会看到极少的不同类别。
    matrix = sparse.csr_matrix([[100, 0], [0, 100]])
    fp, actual = clustered_profiles(matrix, 100, 100, 10000, stable_rng(781))
    assert abs(actual-100) < 6*math.sqrt(100/10000)
    assert 1 < fp.observed.mean() < 1.5


def test_retail_filter_does_not_expand_quantity_or_drop_missing_customer():
    row = [536365, "85123A", "description", 100, None, 2.55, None, "UK"]
    assert retail_reason(row) is None
    assert retail_reason(["C536365", *row[1:]]) == "cancellation"
    assert retail_reason([536365, "POST", *row[2:]]) == "nonproduct_stockcode"
    row[3] = -1
    assert retail_reason(row) == "nonpositive_quantity"


@pytest.mark.parametrize("dataset", list(SOURCES))
def test_downloaded_reference_audit(dataset):
    file = DATA / "prepared" / f"{dataset}_summary.json"
    if not file.exists():
        pytest.skip("尚未下载参考数据；运行 benchmark_data.py 后会检查真实审计")
    info = json.loads(file.read_text())
    counts = np.load(file.parent / f"{dataset}_counts.npz")["counts"]
    matrix = sparse.load_npz(file.parent / f"{dataset}_groups.npz")
    assert counts.sum() == info["records"]
    assert len(counts) == info["support"]
    assert np.array_equal(np.asarray(matrix.sum(axis=0)).ravel(), counts)
    assert sha256(file.parent / f"{dataset}_counts.npz") == info["counts_sha256"]
    if SOURCES[dataset].get("sha256"):
        assert sha256(DATA / "raw" / SOURCES[dataset]["filename"]) == SOURCES[dataset]["sha256"]
    pop = Population.from_counts(counts)
    for eps in [.01, .05, .1]:
        for name, view, n in real_views(pop, eps):
            expected = {"native_boundary": "No", "topn_null": "Yes", "native_null": "Yes", "native_gap": "Gap"}[name]
            assert view.truth(n, eps) == expected
    if dataset == "retail":
        assert info["raw_rows"] == info["records"] + sum(info["removed_rows"].values())
