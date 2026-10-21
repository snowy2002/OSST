#!/usr/bin/env python3
"""Generate the Chinese audit report from completed checks."""
from pathlib import Path
import html
import json

HERE=Path(__file__).resolve().parent
esc=html.escape


def main():
    result=json.loads((HERE/'results.json').read_text())
    exact=json.loads((HERE/'exact_endpoint_certificates.json').read_text())
    fractional=json.loads((HERE/'fractional_output.json').read_text())
    regression=(HERE/'regression_tests.log').read_text()
    assert result['passed'] and exact['passed'] and fractional['passed'] and '101 passed' in regression
    detail=result['details']
    names={
      'universal_bound_endpoints_and_exact_algebra':'统一界端点、几何级数和判定修正代数',
      'independent_finite_distribution_count_probabilities':'独立有限分布计数概率',
      'exact_rational_coefficients_and_expectations':'精确有理数多项式系数与期望恒等式',
      'integer_branch_boundaries_and_parameters':'相邻整数分支边界和全部尺度参数',
      'large_degree_independent_weights_and_pipeline':'完整阶数权重、期望和最终输出',
      'explicit_finite_counterexample_for_C1024':'C₀=1024 的显式有限分布反例',
      'deliberate_error_controls':'有意注入错误的对照检查',
    }
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C₀=1025 证明全链路复核</title>
<style>body{max-width:1050px;margin:36px auto;padding:0 24px;font:16px/1.8 system-ui,sans-serif;color:#243345}h1,h2{line-height:1.4}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd5df;padding:9px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px;font-size:13px}a{color:#006d77}.box{padding:18px;background:#edf7f3;border-left:4px solid #147d64}</style><body>
<h1>C₀=1025：证明、计算与实现的全链路复核</h1>
<p>对 2026-10-21 推导的后续复核。本文明确区分解析证明、精确代数、有限数值检查和实现检查。</p>
<div class="box"><b>结论：</b>在保持 L₀=1024、原 Section 3 构造和 iid 泊松采样模型的条件下，未发现推翻 C₀=1025 充分性的数学错误。补全论证细节并修正反例取整增量的表述后，证明链闭合。当前证据不是 Lean/Isabelle 形式化证明，也不是第三方独立审稿，不能据此宣称“绝对不存在任何问题”。</div>
<h2>核验对象与结论范围</h2>
<p>结论是原算法的两类错误概率分别小于 1%，以及支持大小下界联合成功率至少 98%。只把 C₀ 从 2²⁶ 改为 1025；b=64L/m、各尺度、各阶数、修正项和分支门槛保持原定义。C₀=1025 是保留该 1% 保证时的最小正整数常数，不是所有算法或仅要求 2/3 成功率时的最优常数。原中间界 ‖f‖∞≤4n^(1/16) 被替换，没有沿用未经重新证明的旧界。</p>
<h2>解析证明逐环节检查</h2>
<table><tr><th>环节</th><th>检查与依据</th><th>结果</th></tr>
<tr><td>模型与概率口径</td><td>M~Poi(m)，标签 iid；Nᵢ 独立。m 是均值，D 是观察支持数；ε 是承诺间隔，不是失败概率。</td><td>条件明确</td></tr>
<tr><td>计数耦合</td><td>已见 k≤r 类时，未见质量 qₖ≥ε。用新独立 Eₖ~Exp(1) 构造等待时间 Eₖ/qₖ≤Eₖ/ε；得到 P(D≤r)≤P(Poi(mε)≤r)。</td><td>补充了显式耦合，未假设真实等待时间彼此独立</td></tr>
<tr><td>所有小 L</td><td>n&lt;10⁸ 与 n≥10⁸ 两段覆盖全部 L&lt;1024。Chernoff 指数分别大于 46 和 47，均超过 ln100。</td><td>有解析覆盖；有限网格不承担全称证明</td></tr>
<tr><td>大 L 参数</td><td>J≥0，0&lt;aⱼ≤b/2&lt;1，kⱼ&lt;1.5L，m√[b(b−aⱼ)]≥kⱼ，ρⱼ≤δⱼ。包括 floor/ceil 边界。</td><td>符合期望引理全部前提</td></tr>
<tr><td>新权重界</td><td>Tₖ 在 (−1,1) 内的根保证平移系数正性；阶乘积分给出绝对系数和；双曲函数差界推出 ‖f‖∞≤4exp(2k²/[m(b−a)])。</td><td>无假设系数非负的偷换；原 f 本身可正可负</td></tr>
<tr><td>方差和多尺度</td><td>Var Zⱼ≤‖fⱼ‖∞² ED；Σδⱼ⁻²≤(4/3)L⁴。最终失败界 (128e/3)L⁴exp(−23L/32) 在 L≥1024 上递减。</td><td>覆盖全部尺度，无遗漏并集因子</td></tr>
<tr><td>可数无限支持</td><td>遗漏尾部统计量绝对值≤‖f‖∞M_tail，二阶矩≤‖f‖∞²[mτ+(mτ)²]→0。</td><td>补充了 L² 收敛依据</td></tr>
<tr><td>最终 Yes/No</td><td>分别处理 S≤n、ED&gt;2n，以及 p_(r)≥b/2、p_(r)&lt;a₀、a₀≤p_(r)&lt;b/2。最后一类严格余量为 δ[(14−16δ)n−2r]/(1+δ)&gt;0。</td><td>分支穷尽，边界归属明确</td></tr>
<tr><td>下界输出与取整</td><td>整数 S 保证 ceil 不破坏上界；取 r=min(effε,n)−1 控制下界；只在这两个事件之间用并集界。</td><td>联合成功率≥98%</td></tr>
<tr><td>均值向上取整</td><td>计数分支预算增加；多项式分支等价于 C′∈[C,C+εL/n)，仍处在证明的合法常数范围。</td><td>不依赖“所有多项式检验随样本数单调”的错误假设</td></tr>
</table>
<h2>代码证据</h2><p>下表为同一次完整运行；断言数量只用于可追溯，不代表证明强度或独立数据量。</p>
<table><tr><th>检查组</th><th>通过断言数</th></tr>''']
    for key,count in result['checks_by_group'].items():
        parts.append(f'<tr><td>{esc(names[key])}</td><td>{count:,}</td></tr>')
    parts.append(f'</table><p>另有 {exact["total_checks"]} 项仅使用 Python 整数和 Fraction 的精确证书，以及 {esc(regression.strip().splitlines()[-1])}。</p>')
    for key in ['independent_finite_distribution_count_probabilities','exact_rational_coefficients_and_expectations','integer_branch_boundaries_and_parameters']:
        short={k:v for k,v in detail[key].items() if k not in ['boundary_n','selected_cases']}
        parts.append(f'<p><b>{esc(names[key])}</b></p><pre>{esc(json.dumps(short,ensure_ascii=False,indent=2))}</pre>')
    large=detail['large_degree_independent_weights_and_pipeline']
    parts.append('<h2>实现精度：不能把低层一次计算当成可靠判定</h2><p>完整阶数的独立正系数展开与生产代码的有符号递推逐系数比较。低层 reference_statistic 不会自行提高精度；入口 reference_test 有自动增精。这里只在独立进程的测试夹具中暂设 C0=1025，未改写生产文件。</p>')
    parts.append('<pre>'+esc(json.dumps({k:large[k] for k in ['weight_coefficients_compared','expectation_points','pipeline_fixtures','precision_events','public_adaptive_calls']},ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<p>初始低层计算曾出现不可判定的 Arb 区间（记录中为 nan）；这不是合法判定结果。公共入口没有猜测输出，而是提高精度后正确返回。另补充一个多项式胜过 D、但统计量小于 n 且不是整数的案例，独立计算完整的向上取整结果，并与公共入口返回的 443 位整数逐位一致：</p><pre>'+esc(json.dumps(fractional,ensure_ascii=False,indent=2))+'</pre>')
    parts.append('''<p>上面的巨大 n 测试是压缩频数与公式核对，没有声称实际采集 10⁴⁴⁴ 个样本。有限的数值测试也不能取代证明对所有 n、ε 和分布的量化。</p>
<h2>发现与修订</h2>
<ol><li>上次 14 个端点检查不足以验证系数计算、计数耦合与输出实现。本轮增加独立实现及精确概率检查。</li>
<li>反例的取整说明原来将尾部均值增量直接写成小于 ε；有限 K 下应为小于 Kq。现明确取 K=100n²+n−1，给出完整有限反例和正确的取整界。此处修订不影响 C₀=1025 的充分性证明。</li>
<li>计数等待时间的独立性构造、无限支持下的二阶收敛、1025 的独立两段计数证明已补写。</li>
<li>新验证器开发时，正系数递推曾多写一个因子 2；精确有理数交叉检查立即失败。修正后重新运行全部检查。该错误不在被审查的解析证明或生产算法中。</li>
<li>大阶数下，原参考实现初始精度不足以获得窄区间；高精度公式检查与公共入口增精检查分别记录，未把宽区间或未决结果当作通过。</li></ol>
<h2>明确的有限分布反例</h2><p>令 n=⌈e¹⁰²³⌉−1，K=100n²+n−1，ε=1/4，q=1/(400n²)，p₀=1−Kq，另有 K 个概率为 q 的类别。总质量精确为 1，Δₙ 精确等于 ε。C₀=1024 的尾部计数均值仅略大于 n，中心矩与独立泊松耦合给出错误率下界。无随机模拟、无正态近似。</p>''')
    parts.append('<pre>'+esc(json.dumps(detail['explicit_finite_counterexample_for_C1024'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append('''<h2>为什么基本常数不依赖浮点数正确性</h2>
<p>附带的 Fraction 证书采用正项 Taylor 部分和证明 ln(51.25)&lt;4、ln100&lt;5、ln2&lt;0.7。用 1.414²&lt;2 证明阶数界。大 L 集中界在端点可进一步放松为 2⁻⁶⁸⁹&lt;1/100（使用 2&lt;e&lt;3），因此即使不引用极小小数 2.916…×10⁻³⁰⁶，也足够完成 1% 证明。单调性的导数分子在 L=1024+x、x≥0 下系数全负。</p>
<h2>仍然存在的验证边界</h2>
<p>这是逐式解析审查、两条系数计算路径、精确概率计算、区间证书和独立整数/有理数端点证书的组合。尚未将概率论、实分析和完整算法编码到证明助手核内；也没有第三方数学审阅。不能以“全部测试通过”代替“绝对无误”的承诺。iid、正确的泊松采样及精确/认证数值计算仍是适用前提。本结论不会自动认证其他实用混合变体或固定样本适配器。</p>
<h2>文件与复现</h2>
<ul><li><a href="../c0_reduction_proof.tex">修订后的完整证明</a></li>
<li><a href="c0_reduction_proof_before_audit.tex">复核前的证明快照</a></li>
<li><a href="../audit_c0_1025.py">独立核验程序</a></li>
<li><a href="results.json">完整核验结果及源码哈希</a></li>
<li><a href="exact_endpoint_certificates.py">纯有理数证书程序</a> / <a href="exact_endpoint_certificates.json">结果</a></li>
<li><a href="verify_fractional_output.py">非整数输出的完整取整核验</a> / <a href="fractional_output.json">结果</a></li>
<li><a href="regression_tests.log">回归测试日志</a></li></ul>
<pre>cd /home/snowy/OSST
.venv/bin/python artifacts/audit_c0_1025.py
.venv/bin/python artifacts/c0_1025_audit/exact_endpoint_certificates.py
.venv/bin/python artifacts/c0_1025_audit/verify_fractional_output.py
.venv/bin/python -m pytest -q
.venv/bin/python artifacts/c0_1025_audit/render_report.py</pre>
<p>切比雪夫标准恒等式与零点资料：<a href="https://dlmf.nist.gov/18.5.E1">NIST DLMF 18.5</a>、<a href="https://dlmf.nist.gov/18.16">NIST DLMF 18.16</a>。</p>
</body></html>''')
    (HERE/'report.html').write_text('\n'.join(parts))
    print(HERE/'report.html')


if __name__=='__main__':main()
