#!/usr/bin/env python3
"""第 3 节原算法的独立参考实现：原常数、原阶数、原修正、原输出。

使用 Arb 球区间认证 m/J/k 的取整和最终判定。区间跨越判定边界时加精度；
达到资源上限仍不能确定就报错，不改常数、不截断阶数、不猜 Yes/No。
这认证给定次数统计量的数值计算，无法从数据自动验证 iid 采样假设。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import operator
from pathlib import Path

from flint import arb, ctx, fmpq

from support_size import Fingerprint, clipped_paper_lower_bound, paper_correction

C0 = 2**26
L0 = 2**10


class NumericalCertificationError(ArithmeticError):
    """有限资源下区间未能确定离散输出；不是论文检验的 Yes/No 结果。"""


class ReferenceResourceError(ValueError):
    """拒绝超预算计算，不以较小阶数替代原算法。"""


def exact_inputs(n, epsilon) -> tuple[int, Fraction]:
    if isinstance(n, bool):
        raise ValueError("n 必须为正整数")
    try:
        n = operator.index(n)
        ep = epsilon if isinstance(epsilon, Fraction) else Fraction(str(epsilon))
    except (ValueError, TypeError, ZeroDivisionError) as e:
        raise ValueError("n 为正整数；epsilon 应为有限小数或分数") from e
    if n < 1 or not 0 < ep < 1:
        raise ValueError("需要 n>=1 且 0<epsilon<1")
    return n, ep


def _as_arb(x: Fraction):
    return arb(fmpq(x.numerator, x.denominator))


def certified_integer(value: arb, rounding: str) -> int:
    """floor/ceil 的区间只含唯一整数时才能返回该整数。"""
    if rounding not in ("floor", "ceil"):
        raise ValueError("rounding 必须为 floor 或 ceil")
    rounded = value.floor() if rounding == "floor" else value.ceil()
    integer = rounded.unique_fmpz()
    if integer is None:
        raise NumericalCertificationError(f"{rounding} 结果未被当前精度唯一确定")
    return int(integer)


@dataclass(frozen=True)
class ReferenceScale:
    j: int
    a: arb
    b: arb
    A: arb
    B: arb
    k: int
    delta: Fraction
    rho: arb


@dataclass(frozen=True)
class ReferencePlan:
    n: int
    epsilon: Fraction
    L: arb
    m: int
    a0: arb
    b: arb
    J: int | None
    scales: tuple[ReferenceScale, ...]
    precision_bits: int
    sampling_constant: int = 2**26

    @property
    def branch(self):
        return "chebyshev" if self.scales else "observed_count"

    def as_dict(self):
        with ctx.workprec(self.precision_bits):
            return {"algorithm": "paper_section3_reference", "n": self.n,
                    "epsilon": str(self.epsilon),
                    "C0": self.sampling_constant, "L0": L0, "L": self.L.str(60), "m": self.m,
                    "a0": self.a0.str(60), "b": self.b.str(60), "J": self.J,
                    "branch": self.branch, "precision_bits": self.precision_bits,
                    "parameter_rounding_certified": True,
                    "scales": [{"j": s.j, "a": s.a.str(60), "b": s.b.str(60),
                                "A": s.A.str(60), "B": s.B.str(60), "k": s.k,
                                "delta": str(s.delta), "rho": s.rho.str(60)} for s in self.scales],
                    "note": "保留第 3 节公式与分支；C0 显式记录，1025 对应新证明。区间文本包含舍入误差。"}


def _plan_at_precision(n: int, epsilon: Fraction, bits: int, c0=None) -> ReferencePlan:
    """调用者必须已进入 ctx.workprec(bits)。"""
    if isinstance(c0, bool):
        raise ValueError("c0 必须是正整数")
    c0 = C0 if c0 is None else operator.index(c0)
    if c0 < 1:
        raise ValueError("c0 必须是正整数")
    ep = _as_arb(epsilon)
    L = arb(n).log()+1
    if n == 1:
        # L=1 是精确值；避免恰好为整数的有理表达式在区间除法后跨越 ceil。
        ratio = Fraction(c0*n, 1)/epsilon
        m = (ratio.numerator+ratio.denominator-1)//ratio.denominator
    else:
        m = certified_integer(c0*arb(n)/(ep*L), "ceil")
    a0, b = ep/(8*n), 64*L/m
    scales = []
    if L < L0:
        J = None
    elif L >= L0:
        J = certified_integer((b/(2*a0)).log()/arb(2).log(), "floor")
        if J < 0:
            raise NumericalCertificationError("第 3 节大 L 分支不应没有尺度")
        for j in range(J+1):
            a = a0*2**j
            delta = Fraction(1, 2**(j+2))
            k = certified_integer((b/a).sqrt()*_as_arb(2/delta).log(), "ceil")
            A, B = m*a, m*b
            rho = 1/((B+A)/(B-A)).chebyshev_t(k)
            # 第 4 节使用的前提必须能用区间验证；不以容差放过失败。
            if not (0 < a and a <= b/2 and b <= 1 and (B*(B-A)).sqrt() >= k and rho <= _as_arb(delta)):
                raise NumericalCertificationError("当前精度不能认证 Lemma 2.3 / 4.1 的参数条件")
            scales.append(ReferenceScale(j, a, b, A, B, k, delta, rho))
    else:
        raise NumericalCertificationError("L 与 L0 的比较未确定")
    return ReferencePlan(n, epsilon, L, m, a0, b, J, tuple(scales), bits, c0)


def reference_parameters(n: int, epsilon, *, max_bits: int = 65536, c0=None) -> ReferencePlan:
    """逐项对应第 3 节。精度是计算资源，不是统计参数。"""
    n, epsilon = exact_inputs(n, epsilon)
    bits = max(192, n.bit_length()+epsilon.denominator.bit_length()+128)
    while bits <= max_bits:
        with ctx.workprec(bits):
            try:
                return _plan_at_precision(n, epsilon, bits, c0)
            except NumericalCertificationError:
                bits *= 2
    raise NumericalCertificationError("无法在 max_bits 内认证论文参数；未改用近似取整")


def reference_weights(A: arb, B: arb, k: int) -> tuple[arb, ...]:
    """等价于式 (2)(3) 的阶乘系数递推；保持当前 Arb 精度，不转 float。

    u[t,l]=l![y^l]T_t(alpha-beta*y), y=mx,
    u[t+1,l]=2 alpha u[t,l]-2 beta l u[t,l-1]-u[t-1,l].
    f(0)=0，f(l)=1-u[k,l]/u[k,0]，f(l>k)=1。
    """
    if not (0 < A and A < B) or k < 1:
        raise ValueError("需要 0<A<B 和 k>=1")
    alpha, beta = (B+A)/(B-A), 2/(B-A)
    previous = [arb(1)]+[arb(0) for _ in range(k)]
    current = [alpha, -beta]+[arb(0) for _ in range(k-1)]
    for t in range(1, k):
        following = [arb(0) for _ in range(k+1)]
        for ell in range(t+2):
            following[ell] = 2*alpha*current[ell]-previous[ell]
            if ell:
                following[ell] -= 2*beta*ell*current[ell-1]
        previous, current = current, following
    return (arb(0),)+tuple(1-current[ell]/current[0] for ell in range(1, k+1))


def reference_statistic(fp: Fingerprint, plan: ReferencePlan, *, max_degree: int = 4096):
    """原文 tilde S；包括所有规定尺度，degree 上限超出就报错，不截断。"""
    with ctx.workprec(plan.precision_bits):
        score = arb(fp.observed)
        diagnostics = []
        for scale in plan.scales:
            if scale.k > max_degree:
                raise ReferenceResourceError(f"论文需要 k={scale.k}，超出资源上限 {max_degree}；未截断阶数")
            weights = reference_weights(scale.A, scale.B, scale.k)
            raw = arb(fp.observed)
            for ell in range(1, scale.k+1):
                if fp.frequencies.get(ell, 0):
                    raw += fp.frequencies[ell]*(weights[ell]-1)
            adjusted = paper_correction(raw, _as_arb(scale.delta), plan.n)
            score = score.max(adjusted)
            diagnostics.append({"j": scale.j, "k": scale.k,
                                "raw_interval": raw.str(60), "adjusted_interval": adjusted.str(60)})
        return score, diagnostics


def reference_test(fp: Fingerprint, n: int, epsilon, *, m: int, sampling: str = "poisson",
                   max_bits: int = 65536, max_degree: int = 4096, c0=None) -> dict:
    """m 必须显式给出且等于论文预算，防止用小样本冒充严格原版。"""
    if sampling != "poisson":
        raise ValueError("论文原版要求 iid 泊松模型；固定样本适配器属于另一个方法")
    if isinstance(m, bool) or not isinstance(m, int):
        raise ValueError("请以精确整数传入事先指定的 m，不接受已经舍入的 float")
    n, epsilon = exact_inputs(n, epsilon)
    plan = reference_parameters(n, epsilon, max_bits=max_bits, c0=c0)
    if m != plan.m:
        raise ValueError(f"输入 m 与第 3 节预算不一致：需要 {plan.m}；不能以 M 或缩小预算代替")
    bits = max(plan.precision_bits, fp.observed.bit_length()+128)
    while bits <= max_bits:
        with ctx.workprec(bits):
            try:
                plan = _plan_at_precision(n, epsilon, bits, c0)
                score, details = reference_statistic(fp, plan, max_degree=max_degree)
                if score > n:
                    decision, lower = "No", n
                elif score <= n:
                    decision = "Yes"
                    lower = clipped_paper_lower_bound(certified_integer(score, "ceil"), n)
                else:
                    raise NumericalCertificationError("tilde S 区间跨过 n，不能猜测判定")
                return {"algorithm": "paper_section3_reference", "branch": plan.branch,
                        "n": n, "epsilon": str(epsilon), "C0": plan.sampling_constant,
                        "L0": L0, "m": plan.m, "M": fp.total,
                        "observed": fp.observed, "score_interval": score.str(60),
                        "decision": decision, "support_lower_bound": lower,
                        "numeric_decision_certified": True, "precision_bits": bits,
                        "scale_diagnostics": details,
                        "model_requirement": "数据须来自预先指定此 m 的 iid 泊松采样；函数不能仅凭频数证明采样模型。",
                        "guarantee_scope": ("原论文常数" if plan.sampling_constant == 2**26 else
                                            "C0=1025 新证明的条件性保证" if plan.sampling_constant == 1025 else
                                            "实验常数，未声称统一理论保证") + "；须满足 iid 泊松模型。"}
            except NumericalCertificationError:
                bits *= 2
    raise NumericalCertificationError("达到精度上限仍不能认证原公式输出；未返回猜测的 Yes/No")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "test", "demo"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--epsilon", default="0.1", help="精确十进制或分数，不先转换为 float")
    parser.add_argument("--c0", type=int, default=C0, help="采样系数：原版 67108864；新证明 1025")
    parser.add_argument("--fingerprint", type=Path, help='JSON：{"1":F1,"2":F2,...}，不含 F0')
    parser.add_argument("--m", type=int, help="预先指定的精确整数泊松均值")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args(argv)
    try:
        plan = reference_parameters(args.n, args.epsilon, c0=args.c0)
        if args.action == "plan":
            result = plan.as_dict()
        elif args.action == "test":
            if args.fingerprint is None or args.m is None:
                raise ValueError("test 需要 --fingerprint 和 --m")
            values = json.loads(args.fingerprint.read_text())
            fp = Fingerprint({int(k): v for k, v in values.items()})
            result = reference_test(fp, args.n, args.epsilon, m=args.m, c0=args.c0)
        else:
            import numpy as np
            from support_size import poisson_fingerprint
            if args.n > 1000000 or plan.m > 10**18:
                raise ReferenceResourceError("demo 模拟器上限 n<=10^6、m<=10^18；未降低论文预算")
            rng = np.random.default_rng(args.seed)
            result = {"note": "已知有限分布的原参数等价泊松计数模拟；随机发生器使用双精度，不是在现实中获取这些观测。",
                      "plan": plan.as_dict(), "runs": []}
            for S in [args.n, 2*args.n]:
                fp = poisson_fingerprint(np.full(S, 1/S), plan.m, rng)
                result["runs"].append({"true_S": S, **reference_test(fp, args.n, args.epsilon, m=plan.m, c0=args.c0)})
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
