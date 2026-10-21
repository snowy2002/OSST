"""支持集大小检验：论文公式、有限样本修正和可执行的计数基线。

统计模型中的 m 始终是事先指定的泊松均值，不是观测到的样本数 M。
所有多项式参数必须在看数据之前确定。详见 README.md 中的保证范围。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from functools import lru_cache
import math
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import binom, poisson


def validate_problem(n: int, epsilon: float, alpha: float = 0.05) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n 必须是正整数")
    if not 0 < epsilon < 1 or not 0 < alpha < 1:
        raise ValueError("epsilon 和 alpha 必须在 (0, 1) 内")


def paper_plan(n: int, epsilon: float) -> dict:
    """精确遵循第 3 节的参数；只生成计划，不抽取海量样本。"""
    validate_problem(n, epsilon)
    with localcontext() as ctx:
        ctx.prec = max(80, len(str(n)) + 50)
        dn, de = Decimal(int(n)), Decimal(str(epsilon))
        L = 1 + dn.ln()
        m = int((Decimal(2**26) * dn / (de * L)).to_integral_value(rounding=ROUND_CEILING))
        a0 = de / (8 * dn)
        b = 64 * L / m
        scales = []
        if L >= 1024:
            a, j = a0, 0
            while 2 * a <= b:
                delta = Decimal(2) ** (-j - 2)
                k = int(((b / a).sqrt() * (2 / delta).ln()).to_integral_value(rounding=ROUND_CEILING))
                scales.append({"j": j, "a": str(a), "b": str(b), "delta": str(delta), "k": k})
                a *= 2
                j += 1
        return {"n": int(n), "epsilon": epsilon, "L": str(L), "m": m,
                "branch": "chebyshev" if scales else "observed_count",
                "C0": 2**26, "L0": 1024, "scales": scales,
                "note": "论文参数计划；不自动采样。"}


def count_miss_bound(n: int, epsilon: float, sample_size: float,
                     sampling: str = "poisson") -> float:
    """No 情形下 P(D<=n) 的上界；固定样本用二项分布，泊松样本用泊松分布。"""
    validate_problem(n, epsilon)
    if not math.isfinite(sample_size) or sample_size < 0:
        raise ValueError("样本量必须是有限非负数")
    if sampling == "poisson":
        return float(poisson.cdf(n, sample_size * epsilon))
    if sampling == "fixed":
        if int(sample_size) != sample_size:
            raise ValueError("固定样本量必须是整数")
        return float(binom.cdf(n, int(sample_size), epsilon))
    raise ValueError("sampling 只能是 poisson 或 fixed")


def count_plan(n: int, epsilon: float, alpha: float = 0.05,
               sampling: str = "poisson") -> dict:
    """二分查找满足尾界的最小整数预算；不使用宽松的 Markov 界。"""
    validate_problem(n, epsilon, alpha)
    if sampling not in ("poisson", "fixed"):
        raise ValueError("sampling 只能是 poisson 或 fixed")
    low, high = 0, max(1, math.ceil((n + 1) / epsilon))
    while count_miss_bound(n, epsilon, high, sampling) > alpha:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if count_miss_bound(n, epsilon, mid, sampling) <= alpha:
            high = mid
        else:
            low = mid
    return {"n": int(n), "epsilon": epsilon, "alpha": alpha, "sampling": sampling,
            "m": high, "miss_bound": count_miss_bound(n, epsilon, high, sampling),
            "markov_m": math.ceil((n + 1) / (epsilon * alpha)),
            "guarantee": "iid 和 Yes/No 承诺下：误报=0，漏报<=alpha"}


@dataclass(frozen=True)
class Fingerprint:
    """F_l：出现 l 次的类别数。完全忽略未知的 F_0，空间不依赖总样本数。"""
    frequencies: Mapping[int, int]

    def __post_init__(self):
        clean = {}
        for key, value in self.frequencies.items():
            if (isinstance(key, bool) or isinstance(value, bool)
                    or int(key) != key or int(value) != value or key < 1 or value < 0):
                raise ValueError("频数直方图要求 l>=1，F_l>=0，且均为整数")
            if value:
                clean[int(key)] = int(value)
        object.__setattr__(self, "frequencies", clean)

    @property
    def observed(self) -> int:
        return sum(self.frequencies.values())

    @property
    def total(self) -> int:
        return sum(k * v for k, v in self.frequencies.items())

    @classmethod
    def from_counts(cls, counts: Iterable[int]) -> "Fingerprint":
        frequencies = Counter()
        for x in counts:
            if isinstance(x, bool) or not math.isfinite(x) or int(x) != x or x < 0:
                raise ValueError("类别计数必须是非负整数")
            if x:
                frequencies[int(x)] += 1
        return cls(frequencies)

    @classmethod
    def from_labels(cls, labels: Iterable) -> "Fingerprint":
        return cls.from_counts(Counter(labels).values())


def chebyshev_weights(A: float, B: float, k: int, precision: int = 80) -> tuple[float, ...]:
    """直接递推 u[t,l]=l! [y^l] T_t((B+A-2y)/(B-A))，A=ma，B=mb。

    避免单独生成 c_l、l!、m^l；返回 f(0),...,f(k)。f(l>k)=1。
    Decimal 用于系数生成，最终转换为 double 聚合，不声称区间算术认证。
    """
    if not (math.isfinite(A) and math.isfinite(B) and 0 < A < B):
        raise ValueError("需要 0<A<B 且二者有限")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 256:
        raise ValueError("多项式阶数应为 1..256（计算资源限制）")
    if precision < 30:
        raise ValueError("Decimal precision 至少 30")
    with localcontext() as ctx:
        ctx.prec = precision
        a, b = Decimal(str(A)), Decimal(str(B))
        alpha, beta = (b + a) / (b - a), 2 / (b - a)
        prev = [Decimal(1)] + [Decimal(0)] * k
        cur = [alpha, -beta] + [Decimal(0)] * (k - 1)
        for t in range(1, k):
            nxt = [Decimal(0)] * (k + 1)
            for ell in range(t + 2):
                nxt[ell] = 2 * alpha * cur[ell] - prev[ell]
                if ell:
                    nxt[ell] -= 2 * beta * ell * cur[ell - 1]
            prev, cur = cur, nxt
        weights = (0.0,) + tuple(float(1 - cur[l] / cur[0]) for l in range(1, k + 1))
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("权重溢出，请降低阶数或扩大 B")
    return weights


@dataclass(frozen=True)
class Scale:
    j: int
    A: float
    B: float
    k: int
    delta: float
    rho: float
    weights: tuple[float, ...]

    @property
    def width(self) -> float:
        return max(1.0, *self.weights) - min(0.0, *self.weights)


def make_scales(n: int, epsilon: float, m: float, *, mode: str = "practical",
                b_factor: float = 1.0, max_degree: int = 12,
                weight_cap: float = 20.0, precision: int = 80) -> list[Scale]:
    """practical 保持多尺度结构，降低阶数以限制权重幅度，同时保留 Lemma 2.3 条件。

    naive 为研究对照：缩小 b 常数、截断 k，却继续沿用论文 delta 修正。
    paper 仅使用论文实际启用的分支。practical 用实际 rho，不能假定 rho<=delta。
    """
    validate_problem(n, epsilon)
    if not math.isfinite(m) or m <= 0:
        raise ValueError("m 必须是事先指定的有限正数")
    if mode not in ("paper", "naive", "practical"):
        raise ValueError("未知多项式模式")
    if not 1 <= max_degree <= 256 or not math.isfinite(b_factor) or b_factor <= 0:
        raise ValueError("需要 1<=max_degree<=256 和 b_factor>0")
    if weight_cap < 1 or not math.isfinite(weight_cap):
        raise ValueError("weight_cap 必须是有限数且 >=1")
    L = 1 + math.log(n)
    if mode == "paper" and L < 1024:
        return []
    try:
        A0 = m * epsilon / (8 * n)
    except OverflowError as e:
        raise ValueError("参数超出浮点计算范围；巨整数参数请使用 paper_plan") from e
    if not math.isfinite(A0) or A0 <= 0:
        raise ValueError("A0 溢出或下溢；请调整参数，或使用 paper_plan")
    B = 64 * L if mode == "paper" else min(b_factor * L, m)
    if B < 2 * A0:
        return []
    J = math.floor(math.log2(B) - math.log2(A0) - 1)
    if J > 256:
        raise ValueError("尺度数过多，请检查 epsilon 和 m")
    result = []
    for j in range(J + 1):
        A, delta = (2**j) * A0, 2.0 ** (-j - 2)
        # 等价于 acosh((B+A)/(B-A))；A/B 很小时避免先舍入成 acosh(1)。
        theta = 2 * math.atanh(math.sqrt(A / B))
        if mode == "practical":
            # 精确求达到 rho<=delta 所需的最小阶数，再施加有限样本限制。
            target = math.ceil(math.acosh(1 / delta) / theta)
            k = min(target, max_degree, math.floor(math.sqrt(B * (B - A))))
        else:
            target = math.ceil(math.sqrt(B / A) * math.log(2 / delta))
            if mode == "paper" and target > 256:
                raise ValueError("论文阶数超过本实现计算限制；请使用 paper_plan")
            k = target if mode == "paper" else min(target, max_degree)
        if k < 1:
            continue
        weights = chebyshev_weights(A, B, k, precision)
        if mode == "practical":
            while k > 1 and max(map(abs, weights)) > weight_cap:
                k -= 1
                weights = chebyshev_weights(A, B, k, precision)
            if max(map(abs, weights)) > weight_cap:
                continue
        z = k * theta
        rho = 2 * math.exp(-z) / (1 + math.exp(-2 * z))
        result.append(Scale(j, A, B, k, delta, rho, weights))
    return result


def raw_score(fp: Fingerprint, scale: Scale) -> float:
    return math.fsum([float(fp.observed)] + [
        fp.frequencies.get(l, 0) * (scale.weights[l] - 1) for l in range(1, scale.k + 1)])


def paper_statistic(fp: Fingerprint, scales: list[Scale], n: int) -> float:
    """原文修正公式；传入缩小参数的 scales 时，这只是实验统计量。"""
    return max([float(fp.observed)] + [
        (raw_score(fp, s) - s.delta * n) / (1 + s.delta) for s in scales])


def hoeffding_statistic(fp: Fingerprint, scales: list[Scale], n: int,
                        m: float, alpha: float = 0.05) -> float:
    """有限样本误报控制：S<=n 时，P(score>n)<=alpha（泊松化 iid）。

    每个类别的贡献范围宽度 R 包括 f(0)=0、f(l>k)=1。
    Lemma 2.3 给 E(raw)<=(1+rho)n；Hoeffding 给
    t=R sqrt(n log(K/alpha)/2)。K 个尺度用 union bound，无需相互独立。
    这里只保证误报，不能因此宣称 No 情形的漏报也小。
    """
    validate_problem(n, .5, alpha)
    if not math.isfinite(m) or m <= 0:
        raise ValueError("m 必须为有限正数")
    if not scales:
        return float(fp.observed)
    scores = [float(fp.observed)]
    for s in scales:
        if s.B > m * (1 + 1e-14) or math.sqrt(s.B * (s.B - s.A)) + 1e-12 < s.k:
            raise ValueError("该尺度不满足 Lemma 2.3：b<=1，m sqrt(b(b-a))>=k")
        margin = s.width * math.sqrt(n * math.log(len(scales) / alpha) / 2)
        scores.append((raw_score(fp, s) - margin) / (1 + s.rho))
    return max(scores)


def clipped_paper_lower_bound(score: float, n: int) -> int:
    """论文数值输出；不要用这个已截断到 n 的量做 >n 检验。"""
    return min(max(math.ceil(score), 1), n)


def chao1(fp: Fingerprint) -> float:
    """偏差修正的 Chao1 点估计；不是逐次成立的下界，也不是置信界。"""
    f1, f2 = fp.frequencies.get(1, 0), fp.frequencies.get(2, 0)
    return fp.observed + f1 * (f1 - 1) / (2 * (f2 + 1))


def describe_sample(fp: Fingerprint) -> dict:
    return {"M": fp.total, "observed_species": fp.observed,
            "singletons": fp.frequencies.get(1, 0), "doubletons": fp.frequencies.get(2, 0),
            "chao1_bc": chao1(fp),
            "good_turing_coverage": (1 - fp.frequencies.get(1, 0) / fp.total) if fp.total else None,
            "point_estimate_note": "Chao1 和覆盖率是描述性估计；不能证明湖泊全部鱼种数"}


def count_test(fp: Fingerprint, n: int, epsilon: float, *, alpha: float = 0.05,
               sampling: str = "fixed", m: float | None = None) -> dict:
    validate_problem(n, epsilon, alpha)
    if sampling == "poisson":
        if m is None:
            raise ValueError("泊松模式必须显式传入预先指定的 m，不能用 M 替代")
        budget = m
    elif sampling == "fixed":
        budget = fp.total
    else:
        raise ValueError("sampling 只能是 poisson 或 fixed")
    bound = count_miss_bound(n, epsilon, budget, sampling)
    decision = "No" if fp.observed > n else ("Yes" if bound <= alpha else "Inconclusive")
    return {"decision": decision, "n": n, "epsilon": epsilon, "alpha": alpha,
            "sampling": sampling, "budget": budget, "observed": fp.observed,
            "worst_case_false_negative_bound": bound,
            "interpretation": "Yes/No 是承诺问题的判断；Yes 不排除支持集更大但尾部质量<epsilon"}


@lru_cache(maxsize=256)
def poisson_mean_for_pool(N: int, overflow_prob: float = .005) -> float:
    """对预定大小为 N 的 iid 样本池，选择 Pr(Poi(m)>N)<=overflow_prob 的 m。

    N 必须是事先计划的观测量，不能是根据鱼种出现情况选择的停止时间。
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not 1 <= N < 10**9:
        raise ValueError("泊松子抽样需要 1<=N<10^9（NumPy 超几何采样限制）")
    if not 0 < overflow_prob < .5:
        raise ValueError("overflow_prob 必须在 (0,0.5) 内")
    low, high = 0.0, float(N)
    for _ in range(70):
        mid = (low + high) / 2
        if poisson.sf(N, mid) <= overflow_prob:
            low = mid
        else:
            high = mid
    return low


def poissonized_fixed_test(counts: Iterable[int], n: int, epsilon: float,
                           rng: np.random.Generator, *, alpha: float = .05,
                           overflow_prob: float = .005) -> dict:
    """固定样本池的泊松化适配器，不通过重抽 M 强行满足 M<=N。

    从固定 N 条 iid 观测中独立随机选择 M 条（M<=N）等价于无限 iid
    序列的前 M 条；M>N 时只能返回 Inconclusive。对整个过程的单侧误报
    保证仍为 alpha。原始数据不是 iid 时，只能作描述性分析。
    """
    validate_problem(n, epsilon, alpha)
    original = list(counts)
    fp = Fingerprint.from_counts(original)
    N = fp.total
    m = poisson_mean_for_pool(N, overflow_prob)
    M = int(rng.poisson(m))
    result = {"pool_N": N, "m": m, "requested_M": M,
              "overflow_probability_bound": overflow_prob,
              "full_observed_species": fp.observed, "n": n, "epsilon": epsilon,
              "alpha": alpha,
              "guarantee": "整个流程在固定 N、iid 原始样本下单侧误报<=alpha；非 iid 时仅作描述性分析"}
    if M > N:
        # 已经看到 >n 种仍是确定性证据；其余情形不报告多项式结论。
        return {**result, "status": "overflow", "used_M": 0, "score": None,
                "decision": "No" if fp.observed > n else "Inconclusive",
                "support_lower_bound": fp.observed, "polynomial_scales": 0}
    subset = rng.multivariate_hypergeometric(np.asarray(original, dtype=np.int64), M)
    sample_fp = Fingerprint.from_counts(subset)
    scales = make_scales(n, epsilon, m)
    score = hoeffding_statistic(sample_fp, scales, n, m, alpha)
    return {**result, "status": "ok", "used_M": sample_fp.total,
            "subset_observed_species": sample_fp.observed, "score": score,
            "decision": "No" if max(fp.observed, score) > n else "Inconclusive",
            "support_lower_bound": max(fp.observed, min(n, max(1, math.ceil(score)))),
            "polynomial_scales": len(scales)}


def normalize_probabilities(p: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(p) if not isinstance(p, np.ndarray) else p, dtype=float)
    if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("概率必须是非负、有限、一维数组")
    if not math.isclose(float(arr.sum()), 1, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("概率和必须为 1；不会默默归一化错误输入")
    return arr / arr.sum()


def poisson_fingerprint(p: Iterable[float], m: float, rng: np.random.Generator) -> Fingerprint:
    """模拟 N_i~Poi(mp_i)，与先抽 M 再抽标签完全同分布。

    采样 O(S)，np.unique 的排序聚合至多 O(S log S)，空间 O(S)。

    这是已知模拟分布时的加速；真实采样仍需取得相应数量的观测。
    不用 bincount(counts)，避免巨大次数值导致分配数十亿长度数组。
    """
    p = normalize_probabilities(p)
    if not math.isfinite(m) or m <= 0 or m * float(p.max()) > 9e18:
        raise ValueError("m 超过 NumPy 泊松采样的数值范围")
    values, repeats = np.unique(rng.poisson(m * p), return_counts=True)
    return Fingerprint({int(v): int(r) for v, r in zip(values, repeats) if v > 0})


def delta_n(p: Iterable[float], n: int) -> float:
    p = normalize_probabilities(p)
    if n < 1:
        raise ValueError("n>=1")
    return float(np.sort(p)[::-1][n:].sum())


def effective_support(p: Iterable[float], epsilon: float) -> int:
    if not 0 < epsilon < 1:
        raise ValueError("epsilon 必须在 (0,1) 内")
    p = normalize_probabilities(p)
    # 按尾部质量计算，避免 1 - cumsum 在接近边界时相消。
    tail = np.cumsum(np.sort(p))
    return max(1, len(p) - int(np.searchsorted(tail, epsilon, side="right")))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """二项比例的 95% Wilson 区间；用于重复实验，不用于真实物种数。"""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("无效的实验计数")
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z*z / trials
    center = (p + z*z / (2 * trials)) / den
    radius = z * math.sqrt(p * (1-p) / trials + z*z / (4 * trials*trials)) / den
    return max(0.0, center - radius), min(1.0, center + radius)
