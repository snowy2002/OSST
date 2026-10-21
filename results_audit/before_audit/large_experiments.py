"""预先声明协议的多领域、大规模支持集检验实验。"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.stats import beta

from benchmark_core import (Population, Profiles, sample_profiles, score_profiles,
                            stable_rng, summarize_scores, synthetic_cases)
from benchmark_data import DATA, ROOT, SOURCES, save_json, sha256
from experiments import pyplot, write_csv
from support_size import count_plan, make_scales, wilson_interval

FACTORS = [.5, 1., 2., 4., 8., 16.]
SIZES = [1000, 10000, 100000, 1000000]
EPSILONS = [.05, .1, .2]
REAL_EPSILONS = [.01, .05, .1]
CANDIDATES = [
    {"b_factor": .5, "max_degree": 12, "weight_cap": 10.},
    {"b_factor": .5, "max_degree": 12, "weight_cap": 20.},
    {"b_factor": 1., "max_degree": 12, "weight_cap": 10.},
    {"b_factor": 1., "max_degree": 12, "weight_cap": 20.},
    {"b_factor": 2., "max_degree": 16, "weight_cap": 10.},
    {"b_factor": 2., "max_degree": 16, "weight_cap": 20.},
]
K = 16
METHOD_NAMES = {"observed_count": "Observed categories", "bounded_multiscale": "Bounded multiscale",
                "bounded_single_scale": "Bounded single scale", "naive_chebyshev": "Naive small constants",
                "chao1_point": "Chao1 point estimate"}
COLORS = {"observed_count": "#58667a", "bounded_multiscale": "#157e74", "bounded_single_scale": "#6b67ad",
          "naive_chebyshev": "#c65b3f", "chao1_point": "#b48b27"}


def budgets(n: int, epsilon: float):
    for c in FACTORS:
        yield f"C={c:g}", c, math.ceil(c*n/(epsilon*(1+math.log(n))))
    yield "certified_count", None, count_plan(n, epsilon)["m"]


def calibrate(out: Path, seed: int, trials: int = 200) -> dict:
    """候选和损失函数在测试前定义；仅用开发集的均匀加权漏报率选参数。

    候选全部有相同的数学误报上界 alpha。若开发集误报率>10%，直接排除
    并保留结果；正式测试不再修改参数或阈值。
    """
    rows = []
    for n in [2000, 20000]:
        for epsilon in [.1, .2]:
            for c in [1., 2., 4., 8.]:
                m = math.ceil(c*n/(epsilon*(1+math.log(n))))
                for name, pop in synthetic_cases(n, epsilon, development=True):
                    if name.endswith("gap"):
                        continue
                    fp = sample_profiles(pop, m, trials, K, stable_rng(seed, "development", n, epsilon, c, name))
                    for ci, config in enumerate(CANDIDATES):
                        score = score_profiles(fp, n, epsilon, m, config)["bounded_multiscale"]
                        row = summarize_scores({"bounded_multiscale": score}, fp, pop, n, epsilon,
                                               {"candidate": ci, "scenario": name, "m": m, "factor": c})[0]
                        rows.append(row)
    ranking = []
    for ci, config in enumerate(CANDIDATES):
        rr = [r for r in rows if r["candidate"] == ci]
        max_fp = max(r["error_rate"] for r in rr if r["truth"] == "Yes")
        mean_fn = float(np.mean([r["error_rate"] for r in rr if r["truth"] == "No"]))
        ranking.append({"candidate": ci, **config, "max_false_positive": max_fp,
                        "mean_false_negative": mean_fn, "eligible": max_fp <= .10})
    eligible = [r for r in ranking if r["eligible"]]
    if not eligible:
        raise RuntimeError("所有候选开发误报异常；停止正式评测")
    selected = min(eligible, key=lambda r: (r["mean_false_negative"], r["weight_cap"], r["max_degree"], r["candidate"]))
    config = CANDIDATES[selected["candidate"]]
    write_csv(out / "calibration_results.csv", rows)
    save_json(out / "calibration.json", {"development_seed": seed, "trials_per_cell": trials,
                                        "sizes": [2000, 20000], "epsilons": [.1, .2],
                                        "criterion": "最小化开发集所有 No 场景/预算的等权平均未拒绝率；候选均有 alpha=.05 的误报界；max FP>.10 排除",
                                        "ranking": ranking, "selected_config": config,
                                        "selected_before_test": True})
    print(f"开发集选定参数 {config}，mean FN={selected['mean_false_negative']:.4f}", flush=True)
    return config


def write_protocol(out: Path, config: dict, trials: int, seed: int, stress_trials: int, data: Path):
    calibration = json.loads((out / "calibration.json").read_text())
    if config != calibration["selected_config"]:
        raise ValueError("正式配置必须与冻结的校准结果一致")
    source_files = [ROOT / p for p in ["support_size.py", "benchmark_core.py", "benchmark_data.py",
                                      "large_experiments.py", "run_large_benchmark.py", "report_large.py",
                                      "test_support_size.py", "test_benchmark.py", "requirements-lock.txt"]]
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "version": 1,
                "synthetic_sizes": SIZES, "synthetic_epsilons": EPSILONS, "real_epsilons": REAL_EPSILONS,
                "sample_factors": FACTORS, "additional_budget": "最小整数 m 使 PoissonCDF(n,m*epsilon)<=.05",
                "alpha": .05, "trials_per_test_cell": trials, "stress_trials": stress_trials,
                "test_seed": seed, "development_seed": calibration["development_seed"],
                "config": config, "methods": METHOD_NAMES, "profile_max_count": K,
                "calibration_sha256": sha256(out / "calibration.json"),
                "source_hashes": {p.name: sha256(p) for p in source_files},
                "data_summaries_sha256": sha256(data / "dataset_summary.json"),
                "decision_rule": "score>n 拒绝支持集<=n；No 样本未拒绝记为 failure_to_reject；Gap 不计错",
                "confidence": "各单元给 95% Wilson 区间；另对所有正式 iid 单元/方法给 Bonferroni 95% 同时单侧 Clopper-Pearson 上界",
                "data_use": "真实频数定义有限经验总体；抽样 iid with replacement + Poissonization。原记录从未被直接假定为 iid。",
                "real_views": {"native_boundary": "n=eff_epsilon(p)-1，保留完整经验分布，No",
                               "topn_null": "把同一个完整经验分布限制到 top-n 并归一化，合成 Yes 对照",
                               "native_null": "完整经验分布，查询 n=S，Yes",
                               "native_gap": "n 取 boundary_n 和 S 的内部中点，完整经验分布，Gap"},
                "interpretation": "统计检验和置信下界的实证验证，不是总类别数的一致精确估计，也不能以有限测试代替最优阶证明。"}
    save_json(out / "protocol.json", protocol)
    return protocol


def evaluate(pop, n, epsilon, m, trials, rng, config, metadata):
    start = time.perf_counter()
    fp = sample_profiles(pop, m, trials, K, rng)
    sample_seconds = time.perf_counter()-start
    start = time.perf_counter()
    scores = score_profiles(fp, n, epsilon, m, config)
    score_seconds = time.perf_counter()-start
    rows = summarize_scores(scores, fp, pop, n, epsilon,
                            {**metadata, "m": m, "normalized_budget": m*epsilon*(1+math.log(n))/n,
                             "sampler_groups": len(pop.multiplicities), "sampling_seconds": sample_seconds,
                             "scoring_seconds": score_seconds})
    # 同一批样本上的成对比较，保留只由某种算法检出的数量。
    base = scores["observed_count"] > n
    regular = scores["bounded_multiscale"] > n
    for row in rows:
        row["bounded_only_rejects"] = int(np.sum(regular & ~base))
        row["count_only_rejects"] = int(np.sum(base & ~regular))
    return rows


def run_synthetic_large(out: Path, config: dict, trials: int, seed: int):
    rows = []
    for n in SIZES:
        for epsilon in EPSILONS:
            for name, pop in synthetic_cases(n, epsilon):
                for budget_id, factor, m in budgets(n, epsilon):
                    rows.extend(evaluate(pop, n, epsilon, m, trials,
                                         stable_rng(seed, "test_synthetic", n, epsilon, name, budget_id), config,
                                         {"suite": "synthetic", "dataset": "constructed", "scenario": name,
                                          "sampling": "iid_poisson", "budget_id": budget_id, "factor": factor,
                                          "reference_records": None}))
            write_csv(out / "synthetic_large.csv", rows)  # 可审计的进度检查点，不跳过失败案例。
            print(f"正式合成：n={n:,}, epsilon={epsilon}，11 场景 x 7 预算 x {trials} 次完成", flush=True)
    return rows


def real_views(pop, epsilon):
    boundary_n = max(1, pop.effective_support(epsilon)-1)
    gap_n = min(pop.support-1, max(boundary_n+1, (boundary_n+pop.support)//2))
    return [("native_boundary", pop, boundary_n), ("topn_null", pop.restrict_top(boundary_n), boundary_n),
            ("native_null", pop, pop.support), ("native_gap", pop, gap_n)]


def run_real_large(out: Path, config: dict, trials: int, seed: int, data: Path = DATA):
    rows = []
    for dataset in SOURCES:
        counts = np.load(data / "prepared" / f"{dataset}_counts.npz")["counts"]
        pop = Population.from_counts(counts)
        for epsilon in REAL_EPSILONS:
            for name, view, n in real_views(pop, epsilon):
                expected = {"native_boundary": "No", "topn_null": "Yes", "native_null": "Yes", "native_gap": "Gap"}[name]
                if view.truth(n, epsilon) != expected:
                    raise ValueError(f"{dataset}/{name}: 预定义真值不成立，不能冒充承诺情形")
                for budget_id, factor, m in budgets(n, epsilon):
                    rows.extend(evaluate(view, n, epsilon, m, trials,
                                         stable_rng(seed, "test_real", dataset, epsilon, name, budget_id), config,
                                         {"suite": "real_empirical", "dataset": dataset, "scenario": name,
                                          "sampling": "iid_poisson", "budget_id": budget_id, "factor": factor,
                                          "reference_records": int(counts.sum())}))
            write_csv(out / "real_large.csv", rows)
            print(f"真实频谱：{dataset}, epsilon={epsilon}，4 场景 x 7 预算 x {trials} 次完成", flush=True)
    return rows


def clustered_profiles(matrix, m, block_size, trials, rng):
    """故意违反独立性：先抽 M~Poi(m)，每 block_size 条从同一自然分组取样。

    选组概率与组的记录数成正比，所以每条记录的边缘分布仍为原 p；
    只改变组内相关性。末尾不足一个块按实际余数生成，不补齐或丢弃。
    """
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    group_p = totals/totals.sum()
    low, observed, Ms = [], [], []
    for _ in range(trials):
        M = int(rng.poisson(m))
        number = rng.multinomial(M//block_size, group_p)*block_size
        if M % block_size:
            number[rng.choice(len(group_p), p=group_p)] += M % block_size
        counts = np.zeros(matrix.shape[1], dtype=np.int64)
        for group in np.flatnonzero(number):
            start, end = matrix.indptr[group:group+2]
            cols, cc = matrix.indices[start:end], matrix.data[start:end]
            counts[cols] += rng.multinomial(int(number[group]), cc/totals[group])
        observed.append(np.count_nonzero(counts))
        low.append(np.bincount(np.minimum(counts, K+1), minlength=K+2)[1:K+1])
        Ms.append(M)
    return Profiles(np.asarray(observed), np.asarray(low)), float(np.mean(Ms))


def run_dependence_stress(out: Path, config: dict, trials: int, seed: int, data: Path = DATA):
    rows = []
    for dataset in SOURCES:
        matrix = sparse.load_npz(data / "prepared" / f"{dataset}_groups.npz").tocsr()
        pop = Population.from_counts(np.asarray(matrix.sum(axis=0)).ravel())
        epsilon = .1
        for scenario, n in [("native_boundary", pop.effective_support(epsilon)-1), ("native_null", pop.support)]:
            for factor in [2., 4.]:
                m = math.ceil(factor*n/(epsilon*(1+math.log(n))))
                for block in [1, 20, 100]:
                    rng = stable_rng(seed, "stress", dataset, scenario, factor, block)
                    if block == 1:
                        fp, mean_M = sample_profiles(pop, m, trials, K, rng), None
                    else:
                        fp, mean_M = clustered_profiles(matrix, m, block, trials, rng)
                    scores = score_profiles(fp, n, epsilon, m, config)
                    rows.extend(summarize_scores(scores, fp, pop, n, epsilon,
                                                 {"dataset": dataset, "scenario": scenario, "m": m,
                                                  "factor": factor, "block_size": block,
                                                  "sampling": "iid_poisson" if block == 1 else "correlated_blocks_poisson_total",
                                                  "mean_actual_M": mean_M}))
        write_csv(out / "dependence_stress.csv", rows)
        print(f"自然分组相关性压力实验：{dataset} 完成", flush=True)
    return rows


def run_richness(out: Path, config: dict, trials: int, seed: int, data: Path = DATA):
    rows = []
    for dataset in SOURCES:
        counts = np.load(data / "prepared" / f"{dataset}_counts.npz")["counts"]
        pop = Population.from_counts(counts)
        # 固定的实验查询上限，明确记录；不把它伪装成从样本估出的 S。
        n = 2 ** math.ceil(math.log2(pop.support))
        for fraction in [.001, .005, .01, .05, .1, .25, .5]:
            m = max(10, math.ceil(fraction*counts.sum()))
            fp = sample_profiles(pop, m, trials, K, stable_rng(seed, "richness", dataset, fraction))
            scores = score_profiles(fp, n, .01, m, config)
            estimates = {"observed_count": scores["observed_count"], "chao1_point": scores["chao1_point"],
                         "bounded_lower": np.maximum(fp.observed, np.minimum(n, np.maximum(1, np.ceil(scores["bounded_multiscale"])) ))}
            for method, values in estimates.items():
                rows.append({"dataset": dataset, "method": method, "reference_records": int(counts.sum()),
                             "S": pop.support, "query_n": n, "epsilon": .01, "fraction": fraction,
                             "m": m, "trials": trials, "mean": float(values.mean()),
                             "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975)),
                             "mean_relative_error": float(np.mean(values/pop.support-1)),
                             "relative_rmse": float(np.sqrt(np.mean((values/pop.support-1)**2))),
                             "within_10pct": float(np.mean(np.abs(values/pop.support-1) <= .1)),
                             "not_above_support": float(np.mean(values <= pop.support))})
    write_csv(out / "richness_curves.csv", rows)
    return rows


def simultaneous_intervals(rows):
    eligible = [r for r in rows if r["truth"] != "Gap"]
    H = len(eligible)
    for r in rows:
        if r["truth"] == "Gap":
            r["error_simultaneous_upper"] = None
        elif r["errors"] == r["trials"]:
            r["error_simultaneous_upper"] = 1.
        else:
            r["error_simultaneous_upper"] = float(beta.ppf(1-.05/H, r["errors"]+1, r["trials"]-r["errors"]))
    return H


def sample_efficiency(rows, out):
    result = []
    synthetic = [r for r in rows if r["suite"] == "synthetic" and r["truth"] != "Gap"]
    for n in SIZES:
        for epsilon in EPSILONS:
            for method in METHOD_NAMES:
                rr = [r for r in synthetic if r["n"] == n and r["epsilon"] == epsilon and r["method"] == method]
                by_m = defaultdict(list)
                for r in rr:
                    by_m[r["m"]].append(r)
                passing = [m for m, group in by_m.items() if max(r["error_simultaneous_upper"] for r in group) <= .05]
                best = min(passing) if passing else None
                result.append({"n": n, "epsilon": epsilon, "method": method,
                               "smallest_tested_passing_m": best,
                               "normalized_passing_budget": best*epsilon*(1+math.log(n))/n if best else None,
                               "count_certified_m": count_plan(n, epsilon)["m"],
                               "meaning": "测试网格内所有 10 个承诺场景的同时误差上界<=5%；空白表示未通过，不外推到所有分布。"})
    write_csv(out / "sample_efficiency.csv", result)
    return result


def render_figures(out, synthetic, real, stress, richness, efficiency, data):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, truth in zip(axes, ["Yes", "No"]):
        for method in METHOD_NAMES:
            yy = [max(r["error_rate"] for r in synthetic if r["truth"] == truth and r["method"] == method and r["factor"] == c) for c in FACTORS]
            ax.plot(FACTORS, yy, "o-", color=COLORS[method], label=METHOD_NAMES[method])
        ax.set(xscale="log", ylim=(-.03, 1.03), xlabel="C in m = ceil(C n / (epsilon log(en)))",
               ylabel="Worst empirical error across n, epsilon and shapes",
               title="Null: false positive" if truth == "Yes" else "Alternative: failure to reject")
        ax.axhline(.05, color="#999999", linestyle=":", linewidth=1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"synthetic_worst_errors.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(SOURCES), figsize=(17, 3.9), sharey=True)
    for ax, dataset in zip(axes, SOURCES):
        for method in ["observed_count", "bounded_multiscale", "bounded_single_scale", "naive_chebyshev"]:
            yy = [max(r["error_rate"] for r in real if r["dataset"] == dataset and r["truth"] == "No" and r["method"] == method and r["factor"] == c) for c in FACTORS]
            ax.plot(FACTORS, yy, "o-", label=METHOD_NAMES[method], color=COLORS[method], markersize=3)
        ax.set(xscale="log", xlabel="C", title=dataset, ylim=(-.03, 1.03))
        ax.axhline(.05, color="#999999", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Worst failure to reject across epsilon")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"real_boundary_errors.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in ["observed_count", "bounded_multiscale", "bounded_single_scale"]:
        rr = sorted([r for r in efficiency if r["epsilon"] == .1 and r["method"] == method], key=lambda r: r["n"])
        xx = [r["n"] for r in rr]
        axes[0].plot(xx, [r["smallest_tested_passing_m"] or np.nan for r in rr], "o-", color=COLORS[method], label=METHOD_NAMES[method])
        axes[1].plot(xx, [r["normalized_passing_budget"] or np.nan for r in rr], "o-", color=COLORS[method])
    axes[0].set(xscale="log", yscale="log", xlabel="n", ylabel="Smallest tested passing sample mean", title="epsilon=0.1; simultaneous 95% error bounds <=5%")
    axes[1].set(xscale="log", xlabel="n", ylabel="m epsilon log(en) / n", title="Finite grid evidence, not an asymptotic proof")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"sample_scaling.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for dataset in SOURCES:
        counts = np.load(data / "prepared" / f"{dataset}_counts.npz")["counts"]
        p = np.sort(counts/counts.sum())[::-1]
        axes[0].plot(np.arange(1, len(p)+1), p, label=dataset)
        axes[1].plot(np.arange(1, len(p)+1)/len(p), np.cumsum(p), label=dataset)
    axes[0].set(xscale="log", yscale="log", xlabel="Category rank", ylabel="Empirical probability", title="Reference frequency spectra")
    axes[1].set(xlabel="Fraction of categories retained", ylabel="Covered probability mass", title="Effective support differs from total support")
    axes[0].legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"data_frequency_spectra.{ext}")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(SOURCES), figsize=(17, 3.9), sharey=True)
    for ax, dataset in zip(axes, SOURCES):
        for method, color in [("observed_count", COLORS["observed_count"]), ("chao1_point", COLORS["chao1_point"]), ("bounded_lower", COLORS["bounded_multiscale"])]:
            rr = [r for r in richness if r["dataset"] == dataset and r["method"] == method]
            xx = [r["fraction"] for r in rr]
            ax.plot(xx, [r["mean"]/r["S"] for r in rr], "o-", color=color, label=method, markersize=3)
            ax.fill_between(xx, [r["q025"]/r["S"] for r in rr], [r["q975"]/r["S"] for r in rr], color=color, alpha=.12)
        ax.axhline(1, color="#999999", linestyle=":")
        ax.set(xscale="log", xlabel="m / reference record count", title=dataset, ylim=(0, 1.25))
    axes[0].set_ylabel("Output / known empirical support S")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"richness_curves.{ext}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for dataset in SOURCES:
        rr = [r for r in stress if r["dataset"] == dataset and r["method"] == "bounded_multiscale" and r["truth"] == "No" and r["factor"] == 4.]
        ax.plot([r["block_size"] for r in rr], [r["error_rate"] for r in rr], "o-", label=dataset)
    ax.set(xscale="log", xlabel="Observations sharing one document/user/invoice/quadrat",
           ylabel="Failure to reject", ylim=(-.03, 1.03), title="Same marginal p, different dependence; C=4, epsilon=0.1")
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"dependence_stress.{ext}")
    plt.close(fig)


def finish_results(out, synthetic, real, stress, richness, config, data=DATA):
    H = simultaneous_intervals(synthetic+real)
    write_csv(out / "synthetic_large.csv", synthetic)
    write_csv(out / "real_large.csv", real)
    efficiency = sample_efficiency(synthetic, out)
    render_figures(out, synthetic, real, stress, richness, efficiency, data)
    rows = synthetic+real
    summary = {"config": config, "iid_cells_before_methods": len(rows)//len(METHOD_NAMES),
               "iid_simulated_samples": sum(r["trials"] for r in rows if r["method"] == "observed_count"),
               "iid_method_decisions": sum(r["trials"] for r in rows), "simultaneous_intervals_count": H,
               "max_synthetic_support": max(r["S"] for r in synthetic),
               "max_synthetic_expected_sample_size": max(r["m"] for r in synthetic),
               "formal_false_positive_bound": "bounded_multiscale 和 bounded_single_scale: iid Poisson 模型下 <=.05（精确算术）；不保证缩小常数后的漏报。",
               "baseline": "observed_count: Yes 时误报 0；certified_count 预算下 No 漏报<=.05。",
               "source_pages": {name: spec["page"] for name, spec in SOURCES.items()},
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(out / "summary.json", summary)
    return summary
