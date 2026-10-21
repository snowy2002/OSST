"""大规模实验核心：按等概率类别分组，精确模拟检验所需的充分统计量。

例如一亿个等概率尾部类别只需一个分组。没有将尾部删去、合并成一个
类别或近似为无碰撞样本。仅不生成算法完全不用的类别名称和高频计数。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.stats import poisson

from support_size import make_scales, wilson_interval


@dataclass(frozen=True)
class Population:
    multiplicities: np.ndarray
    probabilities: np.ndarray  # 每个类别的概率，不是整个组的概率质量。

    def __post_init__(self):
        g0 = np.asarray(self.multiplicities)
        p = np.asarray(self.probabilities, dtype=float)
        if (g0.ndim != 1 or p.shape != g0.shape or not len(p)
                or np.any(g0 <= 0) or np.any(g0 != np.floor(g0))
                or np.any(p <= 0) or not np.all(np.isfinite(p))):
            raise ValueError("组大小必须为正整数，组内每类概率必须为有限正数")
        g = g0.astype(np.int64)
        if not math.isclose(float(g @ p), 1, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("概率质量之和必须为 1")
        order = np.argsort(p)[::-1]
        object.__setattr__(self, "multiplicities", g[order])
        object.__setattr__(self, "probabilities", p[order] / float(g @ p))

    @property
    def support(self):
        return int(self.multiplicities.sum())

    @classmethod
    def from_counts(cls, counts):
        counts = np.asarray(counts)
        if counts.ndim != 1 or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("计数必须是一维非负整数")
        values, repeats = np.unique(counts[counts > 0], return_counts=True)
        return cls(repeats, values / float(counts.sum()))

    def tail_mass(self, n: int) -> float:
        """直接加总被排除类别的质量，避免 1-topmass 的相消。"""
        if n < 0:
            raise ValueError("n>=0")
        start = np.r_[0, np.cumsum(self.multiplicities)[:-1]]
        excluded = self.multiplicities - np.clip(n-start, 0, self.multiplicities)
        return float(excluded @ self.probabilities)

    def effective_support(self, epsilon: float) -> int:
        # 可丢弃的最小概率类别数，用尾部质量稳定计算。
        remainder, discard = epsilon, 0
        for g, p in zip(self.multiplicities[::-1], self.probabilities[::-1]):
            if g*p <= remainder:
                discard += int(g)
                remainder -= float(g*p)
            else:
                discard += min(int(g), int(math.floor(remainder / p)))
                break
        return max(1, self.support-discard)

    def restrict_top(self, n: int) -> "Population":
        if not 1 <= n <= self.support:
            raise ValueError("需要 1<=n<=S")
        start = np.r_[0, np.cumsum(self.multiplicities)[:-1]]
        g = np.clip(n-start, 0, self.multiplicities)
        keep = g > 0
        p = self.probabilities[keep]
        return Population(g[keep], p / float(g[keep] @ p))

    def truth(self, n: int, epsilon: float) -> str:
        return "Yes" if self.support <= n else "No" if self.tail_mass(n) >= epsilon else "Gap"


@dataclass(frozen=True)
class Profiles:
    observed: np.ndarray
    low: np.ndarray  # 第 l-1 列是 F_l；高于 K 的次数合并，但其类别数保留。

    @property
    def trials(self):
        return len(self.observed)


def stable_rng(seed: int, *keys) -> np.random.Generator:
    """按实验键产生固定子流；增删其他场景不会改变已有场景的样本。"""
    digest = hashlib.sha256(repr((seed, *keys)).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))


def sample_profiles(pop: Population, m: float, trials: int, K: int,
                    rng: np.random.Generator) -> Profiles:
    """精确分组法：组内 (F0,...,FK,F>K) ~ Multinomial(g, Poi(mp) 的分箱概率)。

    不同组独立。与逐类别独立抽 Poisson(mp_i) 后统计 D,F1,...,FK 同分布，
    时间 O(组数 * trials * K)，空间 O(trials*K + 组数*K)。
    不保留实际总抽样数 M；不能把 F>K 当作 F_{K+1} 来冒充 M。
    """
    if not math.isfinite(m) or m <= 0 or trials < 1 or K < 2:
        raise ValueError("需要 m>0, trials>=1, K>=2")
    lam = m * pop.probabilities
    q = np.column_stack((poisson.pmf(np.arange(K+1)[None, :], lam[:, None]), poisson.sf(K, lam)))
    if not np.all(np.isfinite(q)):
        raise ValueError("泊松概率计算失败")
    q /= q.sum(axis=1, keepdims=True)
    low = np.zeros((trials, K), dtype=np.int64)
    unseen = np.zeros(trials, dtype=np.int64)
    for g, probabilities in zip(pop.multiplicities, q):
        block = rng.multinomial(int(g), probabilities, size=trials)
        unseen += block[:, 0]
        low += block[:, 1:K+1]
    return Profiles(pop.support-unseen, low)


DEFAULT_CONFIG = {"b_factor": 1., "max_degree": 12, "weight_cap": 20.}


def score_profiles(profiles: Profiles, n: int, epsilon: float, m: float,
                   config: dict = DEFAULT_CONFIG, alpha: float = .05) -> dict[str, np.ndarray]:
    """仅接收观测统计量和事先固定的参数；不接收真实分布或真值标签。"""
    observed = profiles.observed.astype(float)
    regular = make_scales(n, epsilon, m, **config)
    naive = make_scales(n, epsilon, m, mode="naive", b_factor=1, max_degree=8)
    scores = {"observed_count": observed.copy(), "bounded_multiscale": observed.copy(),
              "bounded_single_scale": observed.copy(), "naive_chebyshev": observed.copy()}
    for name, scales in [("bounded_multiscale", regular), ("bounded_single_scale", regular[:1]),
                         ("naive_chebyshev", naive)]:
        for scale in scales:
            if scale.k > profiles.low.shape[1]:
                raise ValueError("需要的多项式阶数超过已经生成的统计量")
            raw = observed + profiles.low[:, :scale.k] @ (np.asarray(scale.weights[1:])-1)
            if name == "naive_chebyshev":
                adjusted = (raw-scale.delta*n) / (1+scale.delta)
            else:
                if scale.B > m*(1+1e-14) or scale.k > math.sqrt(scale.B*(scale.B-scale.A))+1e-12:
                    raise ValueError("有限样本误报界的前提不成立")
                margin = scale.width * math.sqrt(n*math.log(len(scales)/alpha)/2)
                adjusted = (raw-margin) / (1+scale.rho)
            scores[name] = np.maximum(scores[name], adjusted)
    f1, f2 = profiles.low[:, 0].astype(float), profiles.low[:, 1].astype(float)
    scores["chao1_point"] = observed + f1*(f1-1)/(2*(f2+1))
    return scores


def summarize_scores(scores: dict, profiles: Profiles, pop: Population,
                     n: int, epsilon: float, metadata: dict) -> list[dict]:
    truth = pop.truth(n, epsilon)
    eff = pop.effective_support(epsilon)
    trials = profiles.trials
    rows = []
    for method, score in scores.items():
        reject = score > n
        failures = int(reject.sum()) if truth == "Yes" else int((~reject).sum())
        lower = np.minimum(n, np.maximum(1, np.ceil(score)))
        low, high = wilson_interval(failures, trials)
        rows.append({**metadata, "method": method, "truth": truth, "S": pop.support,
                     "n": n, "epsilon": epsilon, "delta_n": pop.tail_mass(n), "effective_support": eff,
                     "trials": trials, "rejects": int(reject.sum()), "reject_rate": float(reject.mean()),
                     "error_metric": "false_positive" if truth == "Yes" else "failure_to_reject" if truth == "No" else "not_defined",
                     "errors": failures if truth != "Gap" else None,
                     "error_rate": failures/trials if truth != "Gap" else None,
                     "error_ci95_low": low if truth != "Gap" else None,
                     "error_ci95_high": high if truth != "Gap" else None,
                     "mean_observed": float(profiles.observed.mean()), "mean_score": float(score.mean()),
                     "support_lower_coverage": float(np.mean(lower <= pop.support)),
                     "effective_lower_success": float(np.mean((lower >= min(eff, n)) & (lower <= pop.support))),
                     "clipped_lower_mean": float(lower.mean()),
                     "score_relative_rmse": float(np.sqrt(np.mean((score/pop.support-1)**2))),
                     "note": "No 的未拒绝率按漏报统计；有限预算实际接口应返回 Inconclusive，不能当作已证实 Yes。"})
    return rows


def uniform(S: int) -> Population:
    return Population(np.array([S]), np.array([1/S]))


def headtail(head: float, tail_count: int) -> Population:
    return Population(np.array([1, tail_count]), np.array([head, (1-head)/tail_count]))


def rank_blocks(S: int, exponent: float, groups: int = 96) -> Population:
    """明确构造的分段等概率幂律分布；不冒充精确逐类别 Zipf 分布。"""
    edges = np.unique(np.r_[0, np.geomspace(1, S, groups).round().astype(np.int64), S])
    sizes = np.diff(edges)
    weights = ((edges[1:]+edges[:-1]+1)/2) ** (-exponent)
    return Population(sizes, weights / float(sizes @ weights))


def shaped_boundary(n: int, epsilon: float, seed: int) -> Population:
    """多尺度重尾：用二分固定尾质量恰为 1.01 epsilon，真值仅用于构造。"""
    tail = rank_blocks(100*n, .75 + .05*(seed % 5), 64)
    def mix(t):
        return Population(np.r_[1, tail.multiplicities], np.r_[1-t, t*tail.probabilities])
    low, high = 1e-6, .8
    for _ in range(60):
        mid = (low+high)/2
        if mix(mid).tail_mass(n) < 1.01*epsilon:
            low = mid
        else:
            high = mid
    return mix(high)


def synthetic_cases(n: int, epsilon: float, *, development: bool = False):
    rng = stable_rng(610 if development else 7721, n, epsilon, "shape")
    # lognormal 分组的大小相近，类别概率为 lognormal 随机权重；定义完整且固定。
    edges = np.linspace(0, n, min(n, 96)+1).astype(np.int64)
    sizes = np.diff(edges)
    weights = np.exp(rng.normal(0, 1.8 if development else 2.3, len(sizes)))
    diffuse_factor = 40 if development else 100
    near_multiplier = 1.06 if development else 1.02
    cases = [
        ("uniform_null", uniform(n)),
        ("head90_null", headtail(.90 if not development else .85, n-1)),
        ("rare_tail_null", headtail(.999, n-1)),
        ("rank_power_null", rank_blocks(n, 1.17 if development else 1.31)),
        ("lognormal_blocks_null", Population(sizes, weights / float(sizes @ weights))),
        ("uniform_boundary", uniform(math.ceil(n/(1-epsilon))+1)),
        ("uniform_clear", uniform(2*n)),
        ("diffuse_boundary", headtail(1-near_multiplier*epsilon, diffuse_factor*n)),
        ("diffuse_clear", headtail(1-2*epsilon, diffuse_factor*n)),
        ("multiscale_boundary", shaped_boundary(n, epsilon, 31 if development else 57)),
        ("tiny_mass_gap", headtail(1-.5*epsilon, diffuse_factor*n)),
    ]
    for name, pop in cases:
        expected = "Yes" if name.endswith("null") else "Gap" if name.endswith("gap") else "No"
        if pop.truth(n, epsilon) != expected:
            raise ValueError(f"构造场景标签错误：{name}")
    return cases
