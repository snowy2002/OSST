#!/usr/bin/env python3
"""有限样本双向保证的支持集检验：多项式早拒绝、独立验证早接受、计数收尾。

给定事先固定的 n,epsilon,delta，在 iid、固定分布及承诺问题下：
S<=n 时误报<=delta；Delta_n>=epsilon 时漏报<=delta。
本方法保留论文多项式构造，采用新的停止规则；最坏期望样本界为 O(n/epsilon)，
不声称重新证明了缩小常数后的最优阶。详细证明在 results_guaranteed/report.html。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Iterable

from flint import arb, ctx, fmpq
import numpy as np
from scipy.stats import binom

from paper_reference import exact_inputs, reference_weights
from support_size import Fingerprint, count_plan, make_scales

FACTORS = (Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4), Fraction(8))
DEFAULT_CONFIG = {"b_factor": .5, "max_degree": 12, "weight_cap": 20.,
                  "validation_factor": 8., "validation_gate": .75}
PRECISION = 256


def ball(x):
    x = x if isinstance(x, Fraction) else Fraction(str(x))
    return arb(fmpq(x.numerator, x.denominator))


def directed_float(value, upper: bool) -> float:
    """产生可认证的二进制有理端点；后续判定仅做整数运算。"""
    endpoint = value.upper() if upper else value.lower()
    result = float(endpoint)
    if not math.isfinite(result):
        raise ArithmeticError("参数超过二进制端点的表示范围")
    for _ in range(4):
        exact = ball(Fraction.from_float(result))
        if (exact >= value if upper else exact <= value):
            return result
        result = math.nextafter(result, math.inf if upper else -math.inf)
    raise ArithmeticError("无法认证浮点端点的舍入方向")


@lru_cache(maxsize=256)
def certified_count_budget(n: int, epsilon: Fraction, error: Fraction):
    """SciPy 仅提议预算；最终用 Arb 认证 Poi(m*epsilon) 左尾界。"""
    proposed = count_plan(n, float(epsilon), float(error))["m"]
    if proposed >= 2**52:
        raise ValueError("此实用实现要求预算 <2^52；不缩小预算来绕过限制")
    with ctx.workprec(PRECISION):
        for m in range(proposed, proposed+100):
            bound = (m*ball(epsilon)).gamma_upper(n+1, regularized=1)
            if bound <= ball(error):
                return m, bound.str(60)
    raise ArithmeticError("无法认证计数收尾预算")


@lru_cache(maxsize=512)
def validation_cutoff(r: int, epsilon: Fraction, error: Fraction):
    """选择满足 Pr(Bin(r,epsilon)<=k)<=error 的整数 k；-1 表示禁用。"""
    if r <= 0:
        return -1, "0"
    # 量化点的 CDF 通常 >=error，因此从其前一个整数开始向下认证。
    proposal = min(r-1, int(binom.ppf(float(error), r, float(epsilon)))-1)
    with ctx.workprec(PRECISION):
        for k in range(proposal, -1, -1):
            cdf = (1-ball(epsilon)).beta_lower(r-k, k+1, regularized=1)
            if cdf <= ball(error):
                return k, cdf.str(60)
    return -1, "0"


@dataclass(frozen=True)
class NoCertificate:
    j: int
    degree: int
    coefficients: tuple[int, ...]
    denominator: int
    threshold: int
    correction_lower: tuple[float, ...]
    threshold_upper: float
    rho_interval: str
    range_upper: str

    def rejects(self, observed: int, frequencies) -> bool:
        # Z 的下界与拒绝阈值的上界比较；负权重也正确保留。
        numerator = int(observed)*self.denominator
        numerator += sum(a*int(frequencies.get(l, 0)) for l, a in enumerate(self.coefficients, 1))
        return numerator > self.threshold


def compile_no_certificates(n: int, epsilon: Fraction, m: int, looks: int,
                            error: Fraction, config: dict) -> tuple[NoCertificate, ...]:
    """小常数仅提出候选，Lemma 2.3 条件与权重均由 Arb 重新计算认证。"""
    proposals = make_scales(n, float(epsilon), m, b_factor=config["b_factor"],
                           max_degree=config["max_degree"], weight_cap=config["weight_cap"])
    accepted = []
    with ctx.workprec(PRECISION):
        B = (ball(config["b_factor"])*(arb(n).log()+1)).min(arb(m))
        for proposal in proposals:
            A = ball(Fraction(m, 8*n)*epsilon*2**proposal.j)
            k = proposal.k
            while k >= 1:
                if not (0 < A and A < B and B <= m and (B*(B-A)).sqrt() >= k):
                    k -= 1
                    continue
                weights = reference_weights(A, B, k)
                if all(w.abs_upper() <= ball(config["weight_cap"]) for w in weights):
                    break
                k -= 1
            if k < 1:
                continue
            rho = 1/((B+A)/(B-A)).chebyshev_t(k)
            lower = min([arb(0)]+[w.lower() for w in weights])
            upper = max([arb(1)]+[w.upper() for w in weights])
            width = upper-lower
            accepted.append((proposal.j, k, weights, rho, width))
        compiled = []
        for j, k, weights, rho, width in accepted:
            margin = width*(arb(n)/2*(looks*len(accepted)/ball(error)).log()).sqrt()
            threshold_upper = directed_float(n*(1+rho)+margin, True)
            corrections = tuple(directed_float(w-1, False) for w in weights[1:])
            rational = [Fraction.from_float(x) for x in (*corrections, threshold_upper)]
            denominator = max(x.denominator for x in rational)  # 均为二的幂。
            integers = [x.numerator*(denominator//x.denominator) for x in rational]
            compiled.append(NoCertificate(j, k, tuple(integers[:-1]), denominator, integers[-1],
                                          corrections, threshold_upper, rho.str(60), width.upper().str(60)))
        return tuple(compiled)


@dataclass(frozen=True)
class Look:
    m: int
    certificates: tuple[NoCertificate, ...]
    validation_r: int
    validation_k: int
    validation_bound: str


@dataclass(frozen=True)
class Plan:
    n: int
    epsilon: Fraction
    delta: Fraction
    config: dict
    looks: tuple[Look, ...]
    final_m: int
    final_bound: str
    baseline_m: int
    baseline_bound: str

    @property
    def max_degree(self):
        return max([2]+[c.degree for look in self.looks for c in look.certificates])

    def as_dict(self):
        return {"algorithm": "certified_chebyshev_validation_count", "n": self.n,
                "epsilon": str(self.epsilon), "delta": str(self.delta), "config": self.config,
                "false_positive_bound": str(self.delta), "false_negative_bound": str(self.delta),
                "error_allocation": {"polynomial_total": str(self.delta),
                                     "early_accept_total": str(self.delta/2), "final_count": str(self.delta/2)},
                "final_poisson_mean": self.final_m, "final_count_tail_interval": self.final_bound,
                "baseline_poisson_mean": self.baseline_m, "baseline_count_tail_interval": self.baseline_bound,
                "max_validation_samples": sum(l.validation_r for l in self.looks),
                "expected_sample_upper_bound": self.final_m+sum(l.validation_r for l in self.looks),
                "looks": [{"m": l.m, "scales": len(l.certificates), "degrees": [c.degree for c in l.certificates],
                           "validation_r": l.validation_r, "accept_if_outside_at_most": l.validation_k,
                           "binomial_tail_interval": l.validation_bound} for l in self.looks],
                "guarantee_scope": "固定 n/epsilon/delta、固定分布 iid、采样随机化独立、Yes/No 承诺；不需最小非零概率假设。",
                "complexity": "最坏期望观测量 O((n+log(1/delta))/epsilon)，未证明最优阶的缩常数版本。",
                "numeric_policy": "参数和尾界由 Arb 认证，多项式最终比较使用整数；无法认证则拒绝配置。"}


def make_plan(n=1000, epsilon="0.1", delta="0.05", config=None) -> Plan:
    n, ep = exact_inputs(n, epsilon)
    failure = Fraction(str(delta))
    if not 0 < failure < Fraction(1, 2):
        raise ValueError("delta 必须在 (0,1/2) 内")
    if n > 10**7:
        raise ValueError("实用实现的资源范围 n<=10^7；严格原论文大整数入口另见 paper_reference.py")
    settings = {**DEFAULT_CONFIG, **(config or {})}
    if (set(settings) != set(DEFAULT_CONFIG) or not 0 < settings["validation_gate"] <= 1
            or not math.isfinite(settings["validation_factor"]) or settings["validation_factor"] <= 0):
        raise ValueError("无效的实用配置")
    final_m, final_bound = certified_count_budget(n, ep, failure/2)
    baseline_m, baseline_bound = certified_count_budget(n, ep, failure)
    means = sorted({max(1, math.ceil(float(c)*n/(float(ep)*(1+math.log(n))))) for c in FACTORS})
    means = [m for m in means if 5*m < 4*final_m]
    H = len(means)
    looks = []
    if H:
        r_proposal = math.ceil(settings["validation_factor"]*math.log(float(2*H/failure))/float(ep))
        # 总验证开销至多最终训练均值的 20%；r 只影响功效，不影响已认证的界。
        r = min(r_proposal, final_m//(5*H))
        cutoff, bound = validation_cutoff(r, ep, failure/(2*H))
        if cutoff < 0:
            r = 0
        for m in means:
            certs = compile_no_certificates(n, ep, m, H, failure, settings)
            looks.append(Look(m, certs, r, cutoff, bound))
    return Plan(n, ep, failure, settings, tuple(looks), final_m, final_bound, baseline_m, baseline_bound)


def polynomial_reject(fp: Fingerprint, look: Look, n: int) -> bool:
    if fp.observed > n:
        return True
    return any(c.rejects(fp.observed, fp.frequencies) for c in look.certificates)


def validation_gate(fp: Fingerprint, look: Look, plan: Plan) -> bool:
    """F1/m 只决定要不要花验证样本，绝不直接充当概率证书。"""
    return (fp.observed <= plan.n and look.validation_r > 0 and
            Fraction(fp.frequencies.get(1, 0), look.m) <=
            Fraction(str(plan.config["validation_gate"]))*plan.epsilon)


def run_labels(labels: Iterable, plan: Plan, *, seed: int = 20261019) -> dict:
    """顺序读取真正的 iid 标签；训练和验证均计入成本，不重复使用验证集。

    训练数为独立 Poisson 增量，验证集固定大小、只验证当时冻结的已见集合。
    数据耗尽返回 Inconclusive；不能把一份不足预算的 CSV 强行判为 Yes。
    """
    stream, rng, counts = iter(labels), np.random.default_rng(seed), Counter()
    actual, training, validation, previous_m = 0, 0, 0, 0
    history = []

    def result(decision, reason, mean):
        return {"decision": decision, "reason": reason, "samples_consumed": actual,
                "training_samples": training, "validation_samples": validation,
                "training_poisson_mean": mean, "observed_training_categories": len(counts),
                "n": plan.n, "epsilon": str(plan.epsilon), "delta": str(plan.delta),
                "history": history,
                "guarantee_scope": "仅当预定参数和独立同分布采样设计成立；未完成时不保证给出二元结果。"}

    for look in [*plan.looks, Look(plan.final_m, (), 0, -1, "0")]:
        requested = int(rng.poisson(look.m-previous_m))
        try:
            for _ in range(requested):
                label = next(stream)
                actual += 1
                training += 1
                counts[label] += 1
        except StopIteration:
            return result("No" if len(counts)>plan.n else "Inconclusive", "data_exhausted", previous_m)
        previous_m = look.m
        fp = Fingerprint.from_counts(counts.values())
        entry = {"m": look.m, "M_training": training, "D": fp.observed, "F1": fp.frequencies.get(1, 0)}
        history.append(entry)
        if fp.observed > plan.n:
            return result("No", "observed_more_than_n", look.m)
        if look.m == plan.final_m:
            return result("Yes", "certified_count_fallback", look.m)
        if polynomial_reject(fp, look, plan.n):
            return result("No", "certified_polynomial", look.m)
        if validation_gate(fp, look, plan):
            outside = 0
            try:
                for _ in range(look.validation_r):
                    label = next(stream)
                    actual += 1
                    validation += 1
                    outside += int(label not in counts)
            except StopIteration:
                return result("Inconclusive", "validation_data_exhausted", look.m)
            entry.update({"validation_r": look.validation_r, "outside": outside, "cutoff": look.validation_k})
            if outside <= look.validation_k:
                return result("Yes", "independent_validation", look.m)
    raise AssertionError("计划缺少终止步骤")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "demo", "csv"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--epsilon", default="0.1")
    parser.add_argument("--delta", default="0.05")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--column", default="species")
    parser.add_argument("--config", type=Path, help="JSON 配置；必须在看本次样本前选定")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else None
        plan = make_plan(args.n, args.epsilon, args.delta, config)
        if args.action == "plan":
            output = plan.as_dict()
        elif args.action == "demo":
            rng = np.random.default_rng(args.seed+1)
            def labels(S):
                while True:
                    yield from rng.integers(0, S, size=4096).tolist()
            output = {"plan": plan.as_dict(), "runs": [
                {"true_support": S, **run_labels(labels(S), plan, seed=args.seed)} for S in [args.n, 2*args.n]]}
        else:
            import csv
            if args.file is None:
                raise ValueError("csv 需要 --file，每行是一次观测，保留训练/验证的顺序分割")
            with args.file.open(newline="") as file:
                rows = csv.DictReader(file)
                if args.column not in (rows.fieldnames or []):
                    raise ValueError("CSV 中没有指定类别列")
                output = run_labels((r[args.column] for r in rows), plan, seed=args.seed)
            output["data_model_status"] = "采样独立性无法由 CSV 验证；自然分组/时间漂移需另行处理。"
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError, OSError) as e:
        parser.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
