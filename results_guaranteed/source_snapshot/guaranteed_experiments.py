#!/usr/bin/env python3
"""有双向保证的组合检验：独立开发集、预先冻结协议、正式测试和消融。

模拟器以等概率组内的计数状态递推，精确保留多个检查时刻的联合泊松模型。
验证样本仅在模拟器中用真实缺失质量生成，决策器只得到验证集的事件次数。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from scipy.stats import beta, poisson

from benchmark_core import Population, Profiles, stable_rng, synthetic_cases
from benchmark_data import DATA, ROOT, SOURCES
from experiments import write_csv, pyplot
from guaranteed import DEFAULT_CONFIG, Plan, make_plan, run_labels
from large_experiments import real_views

METHODS = {"count": "计数基线", "count_validation": "计数＋独立验证", "polynomial_count": "多项式＋计数收尾",
           "hybrid": "完整组合方法"}
SIZES = [1000, 10000, 100000, 1000000]
EPSILONS = [.05, .1, .2]
CANDIDATES = [{**DEFAULT_CONFIG, "validation_factor": factor, "validation_gate": gate}
              for factor in [4., 8., 12.] for gate in [.5, .75]]


@dataclass(frozen=True)
class Stage:
    profiles: Profiles
    missing: np.ndarray  # 模拟器真值；禁止传给实际分类接口。


def simulate_trajectory(pop: Population, means, trials: int, degree: int, rng) -> dict[int, Stage]:
    """每个组的 (F0,...,FK,F>K) 在独立泊松增量下转移；高次数状态吸收。

    一组 g 个概率相同的类别中，当前计数为 c 的类别被新 Poi(dt*p) 次数
    分配到 c,c+1,...,K,>K。这样保留检查时刻间的相关性，不能每个时刻重抽。
    """
    if trials < 1 or degree < 2 or list(means) != sorted(set(means)):
        raise ValueError("需要正重复数、degree>=2、严格递增的预算")
    groups = len(pop.multiplicities)
    state = np.zeros((groups, trials, degree+2), dtype=np.int64)
    state[:, :, 0] = pop.multiplicities[:, None]
    stages, previous = {}, 0
    for m in means:
        if m <= 0:
            raise ValueError("泊松均值必须为正数")
        nxt = np.zeros_like(state)
        nxt[:, :, -1] = state[:, :, -1]
        lam = (m-previous)*pop.probabilities
        pmf = poisson.pmf(np.arange(degree+1)[None, :], lam[:, None])
        tails = poisson.sf(np.arange(degree+1)[None, :], lam[:, None])
        # 同一计数状态下所有组一起分配；避免每个组重复调用分布函数。
        for c in range(degree+1):
            active_groups = np.flatnonzero(np.any(state[:, :, c], axis=1))
            if not len(active_groups):
                continue
            cutoff = degree-c
            probabilities = np.column_stack((pmf[active_groups, :cutoff+1], tails[active_groups, cutoff]))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            increment = rng.multinomial(state[active_groups, :, c], probabilities[:, None, :])
            nxt[active_groups, :, c:degree+1] += increment[:, :, :-1]
            nxt[active_groups, :, -1] += increment[:, :, -1]
        state = nxt
        aggregate = state.sum(axis=0)
        profiles = Profiles(pop.support-aggregate[:, 0], aggregate[:, 1:degree+1].copy())
        missing = state[:, :, 0].T @ pop.probabilities
        stages[m] = Stage(profiles, np.clip(missing, 0., 1.))
        previous = m
    return stages


def batch_no(profiles: Profiles, look, n: int) -> np.ndarray:
    """浮点只作候选筛选；每个多项式 No 都必须通过整数证书复核。

    筛选过于保守最多推迟 No，计数收尾仍控制漏报。它不能新增未经认证的 No。
    """
    reject = profiles.observed > n
    observed = profiles.observed.astype(float)
    for cert in look.certificates:
        values = profiles.low[:, :cert.degree]
        coefficients = np.array(cert.correction_lower)
        approximate = observed + values @ coefficients
        magnitude = observed + values @ np.abs(coefficients) + abs(cert.threshold_upper)+1
        guard = 64*np.finfo(float).eps*(cert.degree+1)*magnitude
        candidates = np.flatnonzero(~reject & (approximate+guard > cert.threshold_upper))
        for index in candidates:
            numerator = int(profiles.observed[index])*cert.denominator
            numerator += sum(a*int(f) for a, f in zip(cert.coefficients, values[index]))
            if numerator > cert.threshold:
                reject[index] = True
    return reject


def validation_draws(plan: Plan, stages: dict, rng) -> dict[int, np.ndarray]:
    # 独立的新样本。这里只生成 outside 个数，不能用 missing 本身决定 Yes。
    return {look.m: rng.binomial(look.validation_r, stages[look.m].missing)
            for look in plan.looks if look.validation_r > 0}


def decide_paths(plan: Plan, profiles: dict[int, Profiles], outside: dict[int, np.ndarray]):
    """决策接口无 p、S、missing 或真值标签；只接收观测和固定计划。"""
    T = next(iter(profiles.values())).trials
    poly = {look.m: batch_no(profiles[look.m], look, plan.n) for look in plan.looks}
    outputs = {}
    for method in METHODS:
        active = np.ones(T, dtype=bool)
        rejects = np.zeros(T, dtype=bool)
        training = np.zeros(T, dtype=np.int64)
        validation = np.zeros(T, dtype=np.int64)
        reasons = np.full(T, "", dtype=object)
        allow_validation = method in ("hybrid", "count_validation")
        allow_polynomial = method in ("hybrid", "polynomial_count")
        cap = plan.baseline_m if method == "count" else plan.final_m
        for look in plan.looks:
            if look.m >= cap:
                continue
            observed = profiles[look.m].observed
            direct = active & (observed > plan.n)
            polynomial = active & ~direct & poly[look.m] if allow_polynomial else np.zeros(T, bool)
            no = direct | polynomial
            training[no] = look.m
            rejects[no] = True
            reasons[direct] = "observed"
            reasons[polynomial] = "polynomial"
            active[no] = False
            if allow_validation and look.validation_r:
                # 该门控只节省成本；验证判定依靠整数 k，不依靠浮点估计准确性。
                gate_limit = look.m*plan.epsilon*Fraction(str(plan.config["validation_gate"]))
                gate = active & (profiles[look.m].low[:, 0] <= gate_limit.numerator//gate_limit.denominator)
                validation[gate] += look.validation_r
                yes = gate & (outside[look.m] <= look.validation_k)
                training[yes] = look.m
                reasons[yes] = "validation"
                active[yes] = False
        rejects[active] = profiles[cap].observed[active] > plan.n
        training[active] = cap
        reasons[active] = "fallback"
        outputs[method] = {"rejects": rejects, "training_mean": training,
                           "validation": validation, "cost": training+validation, "reasons": reasons}
    return outputs


def evaluate_paths(pop, n, eps, plan, trials, seed, key):
    means = sorted({l.m for l in plan.looks}|{plan.final_m, plan.baseline_m})
    stages = simulate_trajectory(pop, means, trials, plan.max_degree, stable_rng(seed, "training", key))
    outside = validation_draws(plan, stages, stable_rng(seed, "validation", key))
    outputs = decide_paths(plan, {m: s.profiles for m, s in stages.items()}, outside)
    truth = pop.truth(n, eps)
    baseline_cost = outputs["count"]["cost"]
    rows = []
    for method, output in outputs.items():
        errors = int(output["rejects"].sum() if truth == "Yes" else (~output["rejects"]).sum())
        cost, reasons = output["cost"], output["reasons"]
        rows.append({"method": method, "truth": truth, "n": n, "epsilon": eps, "S": pop.support,
                     "delta_n": pop.tail_mass(n), "trials": trials, "errors": errors if truth != "Gap" else None,
                     "error_rate": errors/trials if truth != "Gap" else None,
                     "reject_rate": float(output["rejects"].mean()), "mean_budget": float(cost.mean()),
                     "mean_training_mean": float(output["training_mean"].mean()),
                     "mean_validation_samples": float(output["validation"].mean()),
                     "budget_q90": float(np.quantile(cost, .90)), "max_budget": int(cost.max()),
                     "count_baseline_mean_budget": float(baseline_cost.mean()),
                     "relative_budget": float(cost.mean()/baseline_cost.mean()),
                     "polynomial_exit_rate": float(np.mean(reasons == "polynomial")),
                     "validation_exit_rate": float(np.mean(reasons == "validation")),
                     "observed_exit_rate": float(np.mean(reasons == "observed")),
                     "fallback_rate": float(np.mean(reasons == "fallback")),
                     "certified_expected_budget_cap": plan.final_m+sum(l.validation_r for l in plan.looks),
                     "paired_budget_difference_se": float(np.std(cost-baseline_cost, ddof=1)/math.sqrt(trials)) if trials>1 else 0.})
    return rows


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def calibrate(out, trials=100, seed=2026101901):
    """所有候选均有相同数学保证；仅以开发集平均相对样本成本选择。"""
    totals = [[] for _ in CANDIDATES]
    cells = []
    for n in [2000, 20000]:
        for eps in [.05, .1, .2]:
            plans = [make_plan(n, eps, config=c) for c in CANDIDATES]
            means = sorted({l.m for p in plans for l in p.looks}|{p.final_m for p in plans}|{p.baseline_m for p in plans})
            for name, pop in synthetic_cases(n, eps, development=True):
                if pop.truth(n, eps) == "Gap":
                    continue
                stages = simulate_trajectory(pop, means, trials, max(p.max_degree for p in plans),
                                             stable_rng(seed, "development", n, eps, name))
                profiles = {m: s.profiles for m, s in stages.items()}
                for i, plan in enumerate(plans):
                    draws = validation_draws(plan, stages, stable_rng(seed, "validation", i, n, eps, name))
                    outputs = decide_paths(plan, profiles, draws)
                    relative = float(outputs["hybrid"]["cost"].mean()/outputs["count"]["cost"].mean())
                    totals[i].append(relative)
                    rejects = outputs["hybrid"]["rejects"]
                    truth = pop.truth(n, eps)
                    cells.append({"candidate": i, "n": n, "epsilon": eps, "scenario": name,
                                  "truth": truth, "trials": trials, "relative_budget": relative,
                                  "errors": int(rejects.sum() if truth == "Yes" else (~rejects).sum())})
            print(f"开发集完成 n={n}, epsilon={eps}", flush=True)
    ranking = sorted([{"candidate": i, "config": c, "mean_relative_budget": float(np.mean(totals[i]))}
                      for i, c in enumerate(CANDIDATES)], key=lambda r: (r["mean_relative_budget"], r["candidate"]))
    selected = ranking[0]["config"]
    save_json(out/"calibration.json", {"seed": seed, "trials_per_cell": trials,
                                      "criterion": "60 个开发场景相对计数基线的平均预算，场景等权；不使用正式测试数据。",
                                      "ranking": ranking, "selected_config": selected})
    save_json(out/"selected_config.json", selected)
    write_csv(out/"calibration_cells.csv", cells)
    return selected


def protocol(out, config, trials, seed):
    sources = [ROOT/p for p in ["guaranteed.py", "guaranteed_experiments.py", "guaranteed_report.py", "test_guaranteed.py",
                                "paper_reference.py", "support_size.py", "benchmark_core.py", "requirements-lock.txt"]]
    saved = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, "seed": seed,
             "trials_per_cell": trials, "sizes": SIZES, "epsilons": EPSILONS,
             "real_epsilons": [.01, .05, .1], "methods": METHODS,
             "dependence_stress": {"datasets": ["retail"], "epsilon": .1, "block_sizes": [1, 10, 100], "trials": 200},
             "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             "calibration_sha256": hashlib.sha256((out/"calibration.json").read_bytes()).hexdigest(),
             "data_summary_sha256": hashlib.sha256((DATA/"dataset_summary.json").read_bytes()).hexdigest(),
             "risk": "每个承诺情形下错误率<=.05；No 的提前 Yes 与最终 Yes 共分配 .05；Yes 的多项式 No 共分配 .05。",
             "budget_metric": "累计训练泊松均值+实际验证条数。有限阶段且增量独立，平均值无偏估计期望实际观测成本；不是每次实际 M。",
             "sampling": "多个阶段共享同一泊松过程的历史；独立验证；原始真实记录只用来定义有限经验分布。",
             "simultaneous_intervals": "全部非 Gap 单元及四种方法，Bonferroni 95% 同时单侧 Clopper–Pearson 上界。",
             "scope": "最坏期望 O(n/epsilon) 的组合算法，不是论文第 3 节的缩常数最优阶证明。"}
    save_json(out/"protocol.json", saved)
    return saved


def run_formal(out, config, trials, seed):
    rows, plans = [], {}
    for n in SIZES:
        for eps in EPSILONS:
            plan = make_plan(n, eps, config=config)
            plans[f"{n}/{eps}"] = plan.as_dict()
            for name, pop in synthetic_cases(n, eps):
                cells = evaluate_paths(pop, n, eps, plan, trials, seed, ("synthetic", n, eps, name))
                rows.extend({"suite": "synthetic", "dataset": "synthetic", "scenario": name, **r} for r in cells)
            print(f"正式合成 n={n}, epsilon={eps}，11 场景完成", flush=True)
    for dataset in SOURCES:
        with np.load(DATA/"prepared"/f"{dataset}_counts.npz") as data:
            pop = Population.from_counts(data["counts"])
        for eps in [.01, .05, .1]:
            for view, current, n in real_views(pop, eps):
                plan = make_plan(n, eps, config=config)
                plans[f"{n}/{eps}"] = plan.as_dict()
                cells = evaluate_paths(current, n, eps, plan, trials, seed, ("real", dataset, eps, view))
                rows.extend({"suite": "real_empirical", "dataset": dataset, "scenario": view, **r} for r in cells)
        print(f"真实频谱 {dataset} 完成", flush=True)
    tested = [r for r in rows if r["errors"] is not None]
    H = len(tested)
    for r in rows:
        k, N = r["errors"], r["trials"]
        r["error_upper95"] = None if k is None else 1. if k == N else float(beta.ppf(.95, k+1, N-k))
        r["error_simultaneous_upper95"] = None if k is None else 1. if k == N else float(beta.ppf(1-.05/H, k+1, N-k))
    write_csv(out/"results.csv", rows)
    save_json(out/"plans.json", plans)
    return rows


def dependence_stress(out, config, trials=200, seed=2026101903):
    """保持每条观测的边际分布，但让相邻 block 条标签相同；明确违反 iid。"""
    with np.load(DATA/"prepared/retail_counts.npz") as data:
        counts = data["counts"]
    probabilities = counts/counts.sum()
    pop = Population.from_counts(counts)
    n, eps = pop.effective_support(.1)-1, .1
    plan = make_plan(n, eps, config=config)
    rows = []
    for block in [1, 10, 100]:
        errors, costs, reason_counts = 0, [], {}
        for trial in range(trials):
            rng = stable_rng(seed, "blocked_labels", block, trial)
            def labels():
                while True:
                    yield from np.repeat(rng.choice(len(counts), size=512, p=probabilities), block).tolist()
            answer = run_labels(labels(), plan, seed=seed+10000*block+trial)
            errors += answer["decision"] != "No"
            costs.append(answer["samples_consumed"])
            reason_counts[answer["reason"]] = reason_counts.get(answer["reason"], 0)+1
        rows.append({"dataset": "retail", "n": n, "epsilon": eps, "S": pop.support,
                     "delta_n": pop.tail_mass(n), "block_size": block, "trials": trials,
                     "errors": errors, "false_negative_rate": errors/trials,
                     "mean_actual_samples": float(np.mean(costs)), "reasons": json.dumps(reason_counts),
                     "guarantee_applies": block == 1})
        print(f"独立性压力测试 block={block}, 漏报={errors}/{trials}", flush=True)
    write_csv(out/"dependence_stress.csv", rows)
    return rows


def figures(out, rows):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, truth in zip(axes, ["Yes", "No"]):
        for method, label in [("count_validation", "Count + validation"), ("polynomial_count", "Polynomial + count"), ("hybrid", "Full hybrid")]:
            means = [np.mean([r["relative_budget"] for r in rows if r["suite"] == "synthetic" and r["n"] == n
                              and r["truth"] == truth and r["method"] == method]) for n in SIZES]
            axis.plot(SIZES, means, "o-", label=label)
        axis.axhline(1, color="gray", linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("Threshold n")
        axis.set_title("True Yes" if truth == "Yes" else "True No")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Mean budget / sequential count baseline")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"sample_efficiency.png")
    fig.savefig(out/"sample_efficiency.pdf")
    plt.close(fig)


def report(out, rows, config):
    hybrid = [r for r in rows if r["method"] == "hybrid" and r["truth"] != "Gap"]
    summary = {"config": config, "cells_per_method": len(rows)//len(METHODS),
               "trials_per_cell": rows[0]["trials"], "method_decisions": sum(r["trials"] for r in rows),
               "max_false_positive": max(r["error_rate"] for r in hybrid if r["truth"] == "Yes"),
               "max_false_negative": max(r["error_rate"] for r in hybrid if r["truth"] == "No"),
               "max_simultaneous_upper95": max(r["error_simultaneous_upper95"] for r in hybrid),
               "mean_relative_budget": float(np.mean([r["relative_budget"] for r in hybrid])),
               "mean_relative_budget_yes": float(np.mean([r["relative_budget"] for r in hybrid if r["truth"] == "Yes"])),
               "mean_relative_budget_no": float(np.mean([r["relative_budget"] for r in hybrid if r["truth"] == "No"])),
               "worst_relative_budget": max(r["relative_budget"] for r in hybrid)}
    save_json(out/"summary.json", summary)
    figures(out, rows)
    # 正文与自包含数学证明由下方独立函数生成，所有数值从本轮 CSV/协议取得。
    render_report(out, rows, summary)
    return summary


def render_report(out, rows, summary):
    # 模板保存在同一项目，避免实验中靠手工筛选数值撰写结论。
    from guaranteed_report import render
    render(out, rows, summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results_guaranteed")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--development-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026101902)
    parser.add_argument("--phase", choices=["all", "calibrate", "evaluate", "report"], default="all")
    args = parser.parse_args(argv)
    if min(args.trials, args.development_trials) < 2:
        parser.error("重复次数至少为 2")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.phase in ("all", "calibrate"):
        config = calibrate(out, args.development_trials)
    else:
        config = json.loads((out/"selected_config.json").read_text())
    if args.phase == "calibrate":
        print(config, flush=True)
        return 0
    if args.phase in ("all", "evaluate"):
        check = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                               text=True, capture_output=True)
        (out/"tests.log").write_text(check.stdout+check.stderr)
        if check.returncode:
            raise RuntimeError("测试未通过；见 tests.log，未开始正式评测")
        protocol(out, config, args.trials, args.seed)
        rows = run_formal(out, config, args.trials, args.seed)
        dependence_stress(out, config)
        save_json(out/"execution.json", {"seconds": time.perf_counter()-start, "python": sys.version})
    else:
        from run_large_benchmark import read_rows
        rows = read_rows(out/"results.csv")
    summary = report(out, rows, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(out/"report.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
