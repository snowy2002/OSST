"""论文原算法的独立核对：有理数闭式系数、期望恒等式、原参数及输出边界。"""
from fractions import Fraction
import json
import math

from flint import arb, ctx, fmpq
import pytest

from paper_reference import (C0, NumericalCertificationError, ReferenceResourceError, ReferencePlan, ReferenceScale,
                             certified_integer, exact_inputs, main, reference_parameters,
                             reference_statistic, reference_test, reference_weights)
from support_size import Fingerprint, chebyshev_weights, make_scales


def rational_chebyshev_polynomial(a, b, k):
    """独立 oracle：用 T_k 的有限闭式展开，直接生成 x 的有理数系数。

    不使用生产代码的 T 递推、u 递推或浮点求值。
    T_k(z)=k/2 sum_r (-1)^r (k-r-1)!/(r!(k-2r)!) (2z)^(k-2r)。
    """
    alpha, beta = (b+a)/(b-a), -2/(b-a)
    coefficients = [Fraction(0) for _ in range(k+1)]
    for r in range(k//2+1):
        power = k-2*r
        factor = Fraction(k*(-1)**r*math.factorial(k-r-1)*2**power,
                          2*math.factorial(r)*math.factorial(power))
        for ell in range(power+1):
            coefficients[ell] += factor*math.comb(power, ell)*alpha**(power-ell)*beta**ell
    # P=-T(affine)/T(alpha)，常数项准确等于 -1。
    return [-c/coefficients[0] for c in coefficients]


@pytest.mark.parametrize("k", [1, 2, 3, 8, 12, 20])
def test_weights_against_independent_exact_polynomial(k):
    a, b, m = Fraction(1, 100), Fraction(1, 2), 23
    coefficients = rational_chebyshev_polynomial(a, b, k)
    assert coefficients[0] == -1
    with ctx.workprec(512):
        weights = reference_weights(arb(fmpq(m*a.numerator, a.denominator)),
                                    arb(fmpq(m*b.numerator, b.denominator)), k)
        assert weights[0] == 0
        for ell in range(1, k+1):
            exact = 1+coefficients[ell]*math.factorial(ell)/m**ell
            assert weights[ell].contains(fmpq(exact.numerator, exact.denominator))
        # 原工程权重在可计算范围内也应是同一多项式；只是没有区间认证。
        assert chebyshev_weights(float(m*a), float(m*b), k) == pytest.approx(
            [float(w) for w in weights], rel=2e-13, abs=1e-13)


@pytest.mark.parametrize("k", [1, 6, 12])
def test_poisson_expectation_identity_in_interval_arithmetic(k):
    with ctx.workprec(512):
        A, B = arb("0.15"), arb(20)
        weights = reference_weights(A, B, k)
        for lam in [arb(0), arb("0.03"), A, (A+B)/2, B, arb(30)]:
            probabilities = [(-lam).exp()*lam**ell/math.factorial(ell) for ell in range(k+1)]
            direct = 1-sum(probabilities)+sum(weights[ell]*probabilities[ell] for ell in range(1, k+1))
            formula = 1-(-lam).exp()*((B+A-2*lam)/(B-A)).chebyshev_t(k)/((B+A)/(B-A)).chebyshev_t(k)
            assert direct.overlaps(formula)
            assert (direct-formula).abs_upper() < arb("1e-100")


def test_corrected_maximum_against_exact_rational_formula_with_polynomial_winner():
    # 独立检查统计量算子，刻意选让多项式胜过 D 的小参数。
    # 此 fixture 不是 §3 参数方案；原方案由其他测试单独核对。
    n, m, k = 20, 23, 5
    a, b, delta = Fraction(1, 100), Fraction(1, 2), Fraction(1, 4)
    coefficients = rational_chebyshev_polynomial(a, b, k)
    oracle = [Fraction(0)]+[1+coefficients[l]*math.factorial(l)/m**l for l in range(1, k+1)]
    with ctx.workprec(512):
        aa, bb = arb(fmpq(a.numerator, a.denominator)), arb(fmpq(b.numerator, b.denominator))
        scale = ReferenceScale(0, aa, bb, m*aa, m*bb, k, delta, 1/((bb+aa)/(bb-aa)).chebyshev_t(k))
        plan = ReferencePlan(n, Fraction(1, 10), arb(4), m, aa, bb, 0, (scale,), 512)
        for counts in [[], [1]*19, [20]*15, [1]*11+[2, 4, 4, 20]]:
            fp = Fingerprint.from_counts(counts)
            exact_raw = sum(oracle[c] if c <= k else Fraction(1) for c in counts)
            exact_score = max(Fraction(fp.observed), (exact_raw-delta*n)/(1+delta))
            score, _ = reference_statistic(fp, plan)
            assert score.contains(fmpq(exact_score.numerator, exact_score.denominator))
            if counts == [1]*19:
                assert score > n and exact_score > n  # 防止一直被 D 分支遮住修正公式中的错误。


@pytest.mark.parametrize("n,epsilon", [(1, "0.1"), (1, "3/7"), (1000, "0.1"), (10**445, "0.1")])
def test_original_parameters_and_minimal_ceiling(n, epsilon):
    plan = reference_parameters(n, epsilon)
    with ctx.workprec(plan.precision_bits):
        ep = Fraction(epsilon)
        ratio = C0*arb(n)/(arb(fmpq(ep.numerator, ep.denominator))*(arb(n).log()+1))
        assert ratio > plan.m-1
        assert ratio <= plan.m or (n == 1 and Fraction(C0, 1)/ep == plan.m)
        for s in plan.scales:
            assert s.a.overlaps(plan.a0*2**s.j)
            assert s.b.overlaps(64*plan.L/plan.m)
            assert s.delta == Fraction(1, 2**(s.j+2))
            target = (s.b/s.a).sqrt()*arb(fmpq(2*s.delta.denominator, s.delta.numerator)).log()
            assert target > s.k-1 and target <= s.k
            assert s.rho <= arb(fmpq(s.delta.numerator, s.delta.denominator))
        if plan.scales:
            assert plan.scales[-1].a <= plan.b/2
            assert plan.scales[-1].a*2 > plan.b/2
            assert len(plan.scales) == plan.J+1
    if n == 1000:
        assert plan.m == 84864619140 and plan.branch == "observed_count"


def test_L0_branch_at_consecutive_integers_around_exp_1023():
    with ctx.workprec(2048):
        boundary = certified_integer(arb(1023).exp(), "floor")
    assert reference_parameters(boundary, "0.1").branch == "observed_count"
    assert reference_parameters(boundary+1, "0.1").branch == "chebyshev"


@pytest.mark.parametrize("D,decision,lower", [(0, "Yes", 1), (999, "Yes", 999),
                                               (1000, "Yes", 1000), (1001, "No", 1000)])
def test_original_small_branch_and_unclipped_decision(D, decision, lower):
    plan = reference_parameters(1000, "0.1")
    result = reference_test(Fingerprint({3: D}), 1000, "0.1", m=plan.m)
    assert (result["decision"], result["support_lower_bound"]) == (decision, lower)
    assert result["m"] == plan.m and result["M"] == 3*D
    assert result["numeric_decision_certified"] and not result["scale_diagnostics"]


def test_giant_original_branch_evaluates_every_scale_and_aggregates_correctly():
    # F_l 压缩表示允许执行真正的大 L 分支，无需物化 10^445 条类别记录。
    # 这是确定性公式核对，不是声称采集了如此多的样本或验证了成功概率。
    n = 10**445
    plan = reference_parameters(n, "0.1")
    assert [s.k for s in plan.scales] == [6, 6, 5]
    fp = Fingerprint.from_counts([0, 1, 1, 2, 4, 8, 10**445])
    score, details = reference_statistic(fp, plan)
    with ctx.workprec(plan.precision_bits):
        for s, detail in zip(plan.scales, details):
            weights = reference_weights(s.A, s.B, s.k)
            direct = sum(weights[c] if c <= s.k else arb(1) for c in [0, 1, 1, 2, 4, 8, 10**445])
            assert direct.overlaps(arb(detail["raw_interval"]))
    for observed, answer in [(n, "Yes"), (n+1, "No")]:
        result = reference_test(Fingerprint({20: observed}), n, "0.1", m=plan.m)
        assert result["decision"] == answer and result["support_lower_bound"] == n
        assert len(result["scale_diagnostics"]) == plan.J+1
        assert result["numeric_decision_certified"]
    assert score >= fp.observed


def test_wrong_budget_fixed_model_and_float_m_are_rejected():
    fp = Fingerprint({1: 100})
    for invalid in [2530, 100, 84864619140.0, True]:
        with pytest.raises(ValueError):
            reference_test(fp, 1000, "0.1", m=invalid)
    with pytest.raises(ValueError, match="泊松"):
        reference_test(fp, 1000, "0.1", m=84864619140, sampling="fixed")
    with pytest.raises(ValueError, match="精确整数论文 m"):
        make_scales(1000, .1, 2530, mode="paper")
    assert make_scales(1000, .1, 84864619140, mode="paper") == []
    big = reference_parameters(10**445, "0.1")
    with pytest.raises(ValueError, match="任意精度"):
        make_scales(big.n, .1, big.m, mode="paper")


def test_resource_limits_raise_instead_of_altering_the_algorithm():
    plan = reference_parameters(10**445, "0.1")
    with pytest.raises(ReferenceResourceError, match="未截断"):
        reference_test(Fingerprint({1: 12}), plan.n, "0.1", m=plan.m, max_degree=5)
    with pytest.raises(NumericalCertificationError):
        reference_parameters(plan.n, "0.1", max_bits=128)
    for operation in ["ceil", "floor"]:
        with pytest.raises(NumericalCertificationError):
            certified_integer(arb("5 +/- 0.1"), operation)


@pytest.mark.parametrize("n,eps", [(True, ".1"), (0, ".1"), (1.2, ".1"),
                                   (1000, "nan"), (1000, "0"), (1000, "1")])
def test_invalid_reference_inputs(n, eps):
    with pytest.raises(ValueError):
        exact_inputs(n, eps)


def test_cli_preserves_exact_epsilon_and_checks_budget(tmp_path, capsys):
    source = tmp_path / "fingerprint.json"
    source.write_text(json.dumps({"1": 1001}))
    assert main(["test", "--n", "1000", "--epsilon", "1/10", "--m", "84864619140",
                 "--fingerprint", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "No" and result["support_lower_bound"] == 1000
    with pytest.raises(SystemExit):
        main(["test", "--n", "1000", "--epsilon", "1/10", "--m", "1001",
              "--fingerprint", str(source)])
