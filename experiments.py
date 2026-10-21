"""可复现实验；模拟结果和真实捕获记录回测分开输出，不据测试结果调参。"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from fish_data import ROOT, SOURCE_PAGE, prepare_fish_data
from support_size import (Fingerprint, chao1, count_plan, delta_n, effective_support,
                          hoeffding_statistic, make_scales, paper_plan, paper_statistic,
                          poisson_fingerprint, poissonized_fixed_test, validate_problem, wilson_interval)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def synthetic_populations(n: int, epsilon: float) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(314159)  # 分布的生成种子独立于重复抽样种子。
    uniform = lambda s: np.full(s, 1 / s)
    headtail = lambda head, s: np.r_[head, np.full(s, (1 - head) / s)]
    zipf = np.arange(1, n + 1, dtype=float) ** -1.2
    lognormal = np.exp(rng.normal(0, 2, size=n))
    boundary_s = math.ceil(n / (1 - epsilon))
    return [
        ("uniform_n", uniform(n)),
        ("head_90pct_null", headtail(0.9, n - 1)),
        ("tiny_tail_null", headtail(0.999, n - 1)),
        ("zipf_null", zipf / zipf.sum()),
        ("lognormal_null", lognormal / lognormal.sum()),
        ("uniform_2n", uniform(2 * n)),
        ("uniform_boundary", uniform(boundary_s)),
        ("diffuse_tail_20pct", headtail(0.8, 100 * n)),
        ("diffuse_tail_near_boundary", headtail(1 - 1.02 * epsilon, 100 * n)),
    ]


def promise_label(p, n, epsilon):
    if np.count_nonzero(p) <= n:
        return "Yes"
    return "No" if delta_n(p, n) >= epsilon else "Gap"


def run_synthetic(out: Path, n: int = 1000, epsilon: float = 0.1,
                  trials: int = 500, seed: int = 20261016, alpha: float = 0.05) -> dict:
    validate_problem(n, epsilon, alpha)
    if not 2 <= n <= 10000 or epsilon > 0.2 or trials < 1:
        raise ValueError("合成实验资源范围：2<=n<=10000，epsilon<=0.2，trials>=1")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    tight = count_plan(n, epsilon, alpha, "poisson")
    budgets = sorted({math.ceil(c * n / (epsilon * (1 + math.log(n)))) for c in [2, 4]} | {tight["m"]})
    rows, scale_rows = [], []
    populations = synthetic_populations(n, epsilon)
    for m in budgets:
        # 所有参数事先固定；不用真实 p、实验标签或回测误差选择参数。
        naive = make_scales(n, epsilon, m, mode="naive", b_factor=1, max_degree=8)
        regular = make_scales(n, epsilon, m, mode="practical", b_factor=1, max_degree=12, weight_cap=20)
        for s in regular:
            scale_rows.append({"m": m, "j": s.j, "A": s.A, "B": s.B, "k": s.k,
                               "rho": s.rho, "paper_delta": s.delta, "weight_range": s.width,
                               "max_abs_weight": max(map(abs, s.weights)),
                               "margin": s.width * math.sqrt(n * math.log(len(regular) / alpha) / 2)})
        for name, p in populations:
            label = promise_label(p, n, epsilon)
            rejects = Counter()
            mean_D = 0
            for _ in range(trials):
                fp = poisson_fingerprint(p, m, rng)
                mean_D += fp.observed
                decisions = {"observed_count": fp.observed > n,
                             "naive_chebyshev": paper_statistic(fp, naive, n) > n,
                             "bounded_chebyshev": hoeffding_statistic(fp, regular, n, m, alpha) > n}
                for method, reject in decisions.items():
                    rejects[method] += int(reject)
            for method in ["observed_count", "naive_chebyshev", "bounded_chebyshev"]:
                errors = rejects[method] if label == "Yes" else trials - rejects[method]
                lo, hi = wilson_interval(errors, trials)
                rows.append({"distribution": name, "truth": label, "S": len(p),
                             "delta_n": delta_n(p, n), "n": n, "epsilon": epsilon,
                             "m": m, "trials": trials, "method": method,
                             "rejects": rejects[method], "reject_rate": rejects[method] / trials,
                             "error_metric": "false_positive" if label == "Yes" else "failure_to_reject",
                             "error_rate": errors / trials if label != "Gap" else None,
                             "error_ci95_low": lo if label != "Gap" else None,
                             "error_ci95_high": hi if label != "Gap" else None,
                             "mean_observed": mean_D / trials})
        print(f"合成实验：完成 m={m}，{len(populations)} 种分布 × {trials} 次", flush=True)
    write_csv(out / "synthetic_results.csv", rows)
    write_csv(out / "polynomial_scales.csv", scale_rows)

    # 论文巨大 m 也能在已知有限模拟分布上执行：直接抽计数，而非生成 M 条标签。
    strict_m = paper_plan(n, epsilon)["m"]
    strict_rng = np.random.default_rng(seed + 1)
    strict_runs = []
    for name, p in populations:
        t = time.perf_counter()
        fp = poisson_fingerprint(p, strict_m, strict_rng)
        strict_runs.append({"distribution": name, "m": strict_m, "M": fp.total,
                            "S": len(p), "D": fp.observed, "seconds": time.perf_counter() - t})
    write_csv(out / "paper_parameters_simulated_counts.csv", strict_runs)
    write_json(out / "sample_plans.json", {"paper": paper_plan(n, epsilon), "tight_poisson": tight,
                                          "tight_fixed": count_plan(n, epsilon, alpha, "fixed")})
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    colors = {"observed_count": "#526474", "naive_chebyshev": "#cb6045", "bounded_chebyshev": "#257e73"}
    for ax, truth in zip(axes, ["Yes", "No"]):
        for method, color in colors.items():
            yy = [max(r["error_rate"] for r in rows if r["m"] == m and r["truth"] == truth and r["method"] == method)
                  for m in budgets]
            ax.plot(budgets, yy, "o-", label=method.replace("_", " "), color=color)
        ax.set_title("Worst null false positive rate" if truth == "Yes" else "Worst alternative failure to reject")
        ax.set_xlabel("Expected sample count m")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Rate across independent repetitions")
    axes[0].axhline(alpha, color="gray", linestyle=":", label=f"alpha = {alpha}")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Support testing, n={n}, epsilon={epsilon}; {trials} trials per distribution")
    fig.tight_layout()
    fig.savefig(out / "synthetic_error_rates.png")
    fig.savefig(out / "synthetic_error_rates.pdf")
    plt.close(fig)
    summary = {"n": n, "epsilon": epsilon, "alpha": alpha, "seed": seed, "trials": trials,
               "budgets": budgets, "paper_m": strict_m, "count_poisson_m": tight["m"],
               "seconds": time.perf_counter() - start,
               "parameter_selection": "预先固定 b_factor=1, max_degree=12, weight_cap=20；未按测试结果调优",
               "bounded_method_note": "No/未拒绝；无统一漏报保证，failure_to_reject 不是已认证的 Yes",
               "data": str(out / "synthetic_results.csv"), "plot": str(out / "synthetic_error_rates.png")}
    write_json(out / "synthetic_summary.json", summary)
    return summary


def metric_row(values, truth, base):
    a = np.asarray(values, dtype=float)
    return {**base, "reference_species": truth, "mean": float(a.mean()),
            "median": float(np.median(a)), "mae": float(np.abs(a-truth).mean()),
            "rmse": float(np.sqrt(np.mean((a-truth)**2))),
            "round_exact_rate": float(np.mean(np.rint(a) == truth)),
            "within_one_rate": float(np.mean(np.abs(a-truth) <= 1)),
            "replicate_p025": float(np.quantile(a, .025)),
            "replicate_p975": float(np.quantile(a, .975))}


def run_fish(out: Path, trials: int = 500, seed: int = 20261016) -> dict:
    if trials < 1:
        raise ValueError("trials 必须是正整数")
    out.mkdir(parents=True, exist_ok=True)
    records = prepare_fish_data()
    rng = np.random.default_rng(seed)
    groups = [(2007, "seine"), (2008, "seine"), (2007, "angling"), (2008, "miniWDNR")]
    point_rows, species_rows, test_rows, event_rows, descriptions, lower_rows = [], [], [], [], [], []
    lower_rng = np.random.default_rng(seed + 20)
    for year, gear in groups:
        subset = [r for r in records if r["year"] == year and r["gear"] == gear]
        counter = Counter(r["species"] for r in subset)
        species = sorted(counter)
        counts = np.array([counter[x] for x in species])
        N, S = int(counts.sum()), len(species)
        p = counts / N
        group = f"{year}_{gear}"
        descriptions.append({"year": year, "gear": gear, "records": N, "reference_species": S,
                             "eff_005": effective_support(p, .05), "eff_001": effective_support(p, .01)})
        for sp in species:
            species_rows.append({"year": year, "gear": gear, "species": sp, "count": counter[sp],
                                 "empirical_p": counter[sp] / N})
        # 真正从有限记录里抽子样本；不放回。对照真值仅为该参考记录的支持集。
        budgets = sorted({x for x in [30, 100, 300, 600, N] if x <= N})
        for M in budgets:
            observed, estimated = [], []
            for _ in range(trials):
                sample = rng.multivariate_hypergeometric(counts, M)
                fp = Fingerprint.from_counts(sample)
                observed.append(fp.observed)
                estimated.append(chao1(fp))
            for method, values in [("observed", observed), ("chao1_bc", estimated)]:
                point_rows.append(metric_row(values, S, {"group": group, "M": M, "N": N,
                                                        "trials": trials, "method": method,
                                                        "sampling": "without_replacement_from_records"}))

        # 实际记录子样本 -> 泊松适配器 -> 论文结构的单侧下界。
        # cap=20 在检查真实种数前固定，避免用已知 S 作 cap 造成泄漏。
        # 原始记录并非经过认证的 iid；这里报告经验覆盖，不声称现场置信保证。
        for M in budgets:
            lower_values, observed_values, overflows = [], [], 0
            for _ in range(trials):
                sample = lower_rng.multivariate_hypergeometric(counts, M)
                answer = poissonized_fixed_test(sample, 20, .01, lower_rng)
                lower_values.append(answer["support_lower_bound"])
                observed_values.append(answer["full_observed_species"])
                overflows += int(answer["status"] == "overflow")
            lower_rows.append({"group": group, "M": M, "N": N, "reference_species": S,
                               "n_cap": 20, "epsilon": .01, "alpha": .05, "trials": trials,
                               "mean_observed": float(np.mean(observed_values)),
                               "mean_polynomial_lower_bound": float(np.mean(lower_values)),
                               "reference_overstatement_rate": float(np.mean(np.asarray(lower_values) > S)),
                               "improvement_over_observed_rate": float(np.mean(np.asarray(lower_values) > observed_values)),
                               "overflows": overflows,
                               "model_status": "descriptive_original_catches_not_certified_iid"})

        # 按捕获事件划分，保留网内相关结构；不能把所有鱼洗牌后称为现场预测。
        events = defaultdict(Counter)
        for r in subset:
            events[r["event_id"]][r["species"]] += 1
        event_ids = sorted(events)
        if len(event_ids) >= 2:
            observations, estimates, train_sizes = [], [], []
            for _ in range(trials):
                train_ids = rng.choice(event_ids, max(1, len(event_ids)//2), replace=False)
                train = Counter()
                for event_id in train_ids:
                    train.update(events[event_id])
                fp = Fingerprint.from_counts(train.values())
                observations.append(fp.observed)
                estimates.append(chao1(fp))
                train_sizes.append(fp.total)
            for method, values in [("observed", observations), ("chao1_bc", estimates)]:
                event_rows.append(metric_row(values, S, {"group": group, "events": len(event_ids),
                                                        "mean_train_fish": float(np.mean(train_sizes)),
                                                        "trials": trials, "method": method}))

        # 独立的实验：把真实频率视为已知经验分布 p，用泊松模型有放回模拟。
        # 这里的 iid 是模拟器提供的，不是声称原始捕获过程 iid。
        for epsilon in [.01, .05]:
            for m in [100, 300, 600]:
                scales = {n: make_scales(n, epsilon, m) for n in range(1, S+1)}
                rejects = Counter()
                for _ in range(trials):
                    fp = poisson_fingerprint(p, m, rng)
                    for n in scales:
                        rejects[n, "observed_count"] += int(fp.observed > n)
                        rejects[n, "bounded_chebyshev"] += int(hoeffding_statistic(fp, scales[n], n, m) > n)
                for n in scales:
                    label = promise_label(p, n, epsilon)
                    for method in ["observed_count", "bounded_chebyshev"]:
                        r = rejects[n, method]
                        error = r if label == "Yes" else trials - r
                        ci = wilson_interval(error, trials)
                        test_rows.append({"group": group, "n": n, "epsilon": epsilon, "m": m,
                                          "reference_species": S, "delta_n": delta_n(p, n),
                                          "truth": label, "method": method, "trials": trials,
                                          "reject_rate": r / trials,
                                          "error_rate": error / trials if label != "Gap" else None,
                                          "error_ci95_low": ci[0] if label != "Gap" else None,
                                          "error_ci95_high": ci[1] if label != "Gap" else None,
                                          "sampling": "poisson_from_empirical_distribution"})
        print(f"鱼类回测：完成 {group}，{N} 条捕获记录，记录内 {S} 种鱼", flush=True)

    write_csv(out / "fish_subsample_results.csv", point_rows)
    write_csv(out / "fish_species_counts.csv", species_rows)
    write_csv(out / "fish_testing_results.csv", test_rows)
    write_csv(out / "fish_event_holdout.csv", event_rows)
    write_csv(out / "fish_polynomial_lower_bounds.csv", lower_rows)
    write_csv(out / "fish_groups.csv", descriptions)

    # 随机子样本示例用于 --csv 入口；不是捕获时间顺序。
    sample_rng = np.random.default_rng(seed + 10)
    pool = [r for r in records if r["year"] == 2007 and r["gear"] == "seine"]
    sample_records = [pool[i] for i in sample_rng.choice(len(pool), 300, replace=False)]
    write_csv(ROOT / "data" / "example_300_fish.csv", sample_records)

    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (year, gear) in zip(axes.flat, groups):
        group = f"{year}_{gear}"
        ref = next(x for x in descriptions if x["year"] == year and x["gear"] == gear)
        for method, color in [("observed", "#526474"), ("chao1_bc", "#257e73")]:
            r = [x for x in point_rows if x["group"] == group and x["method"] == method]
            ax.plot([x["M"] for x in r], [x["mean"] for x in r], "o-", color=color, label=method)
            ax.fill_between([x["M"] for x in r], [x["replicate_p025"] for x in r],
                            [x["replicate_p975"] for x in r], color=color, alpha=.13)
        ax.axhline(ref["reference_species"], color="#cb6045", linestyle="--", label="reference records")
        ax.set(title=f"Inch Lake {year}, {gear}", xlabel="Fish records sampled without replacement", ylabel="Species count")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Richness backtest: shaded bands are replicate percentiles, not confidence intervals")
    fig.tight_layout()
    fig.savefig(out / "fish_richness.png")
    fig.savefig(out / "fish_richness.pdf")
    plt.close(fig)
    summary = {"source": SOURCE_PAGE, "seed": seed, "trials": trials, "fish_records": len(records),
               "observed_species_all_records": len({r["species"] for r in records}),
               "groups": descriptions, "data": str(out / "fish_subsample_results.csv"),
               "plot": str(out / "fish_richness.png"),
               "conclusion_scope": "只验证对参考捕获记录的恢复能力；全湖鱼种总数没有独立真值，不能验证"}
    write_json(out / "fish_summary.json", summary)
    return summary
