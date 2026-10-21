"""从冻结实验输出生成中文结论、数学保证和文献比较。"""
import base64
import html
import json
from pathlib import Path

import numpy as np

from benchmark_data import DATA


REFERENCES = [
    {"title": "Wang: Optimal Support Size Testing", "url": "https://arxiv.org/abs/2610.18723",
     "finding": "原文多尺度切比雪夫构造及 Lemma 2.3 期望界是本实现的基础；原文常数没有被直接套到新规则。"},
    {"title": "Ferreira Pinto Jr. & Harms: Testing Support Size More Efficiently Than Learning Histograms",
     "url": "https://arxiv.org/abs/2410.18915",
     "finding": "同一个支持集承诺检验问题，解释多项式逼近方法；不能据渐近阶直接得到实用常数。"},
    {"title": "Wu & Yang: Chebyshev polynomials, moment matching, and optimal estimation of the unseen",
     "url": "https://arxiv.org/abs/1504.01227",
     "finding": "有理论与实证基础的多项式估计；其总支持集估计模型要求最小非零概率下界，不能直接套用到这里的无该假设问题。"},
    {"title": "Chien & Milenkovic: Support Estimation via Regularized and Weighted Chebyshev Approximations",
     "url": "https://arxiv.org/html/1901.07506v2",
     "finding": "联合优化偏差与方差有助于改善实用参数；论文仍采用最小非零概率条件。本次没有把离散网格优化当作连续域保证。"},
    {"title": "Howard et al.: Time-uniform, nonparametric, nonasymptotic confidence sequences",
     "url": "https://arxiv.org/abs/1810.08240",
     "finding": "反复查看数据需要对整个停止过程控制错误。本次采用有限个预定检查点与误差预算求和，未实现该论文的完整 confidence sequence。"},
    {"title": "McAllester & Ortiz: Concentration Inequalities for the Missing Mass and for Histogram Rule Error",
     "url": "https://www.jmlr.org/papers/v4/mcallester03a.html",
     "finding": "缺失概率质量有专门的集中界；本实现用独立验证绕开直接认证 Good–Turing 缺失质量，F1/m 仅用于决定是否花验证成本。"},
    {"title": "SciPy 官方文档：单侧精确二项检验", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
     "finding": "验证集的 outside 次数对应单侧二项检验；本实现进一步用 Arb 认证所采用的尾概率阈值。"},
    {"title": "python-flint 官方文档：Arb 实数区间", "url": "https://python-flint.readthedocs.io/en/latest/arb.html",
     "finding": "用于认证解析前提、尾概率、权重和阈值的上下界；最终 No 比较转换为整数算术。"},
]


def table(headers, rows):
    return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(x))}</th>" for x in headers)+"</tr></thead><tbody>"+"".join(
        "<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table>"


def render(out: Path, rows, summary):
    from run_large_benchmark import read_rows
    selected = summary["config"]
    protocol = json.loads((out/"protocol.json").read_text())
    calibration = json.loads((out/"calibration.json").read_text())
    datasets = json.loads((DATA/"dataset_summary.json").read_text())
    stress = read_rows(out/"dependence_stress.csv")
    png = base64.b64encode((out/"sample_efficiency.png").read_bytes()).decode()
    pct = lambda value: f"{100*value:.2f}%"
    chunks = [f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>兼顾正确性和样本效率的支持集检验</title><style>
body{{font:16px/1.75 system-ui,sans-serif;color:#233448;max-width:1120px;margin:36px auto;padding:0 24px}}
h1,h2,h3{{line-height:1.4}}h2{{margin-top:42px}}a{{color:#086b86}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #d4e1e6;padding:10px;text-align:left;vertical-align:top}}th{{background:#edf5f6}}
pre{{padding:16px;background:#eef4f7;white-space:pre-wrap;overflow-wrap:anywhere}}code{{background:#edf3f6;padding:2px 5px}}
.finding{{background:#edf9f4;border-left:5px solid #198274;padding:16px}}.limit{{background:#fff7e8;padding:16px}}
img{{width:100%;height:auto}}nav a{{margin-right:18px}}details{{margin:20px 0}}input,select{{padding:8px;margin:8px}}.scroll{{overflow:auto}}
</style><body><p>Optimal Support Size Testing · 有限样本保证与独立实验</p>
<h1>保留多项式方法，补齐两种误判的保证</h1>
<div class="finding"><b>已经实现：</b>多项式尽早判断 No，独立新样本验证 Yes，困难情况由计数检验收尾。
在固定分布 iid 和论文的承诺条件下，两种误判分别不超过 5%。保证来自下方的数学推导；实验用于检查实用效果。</div>
<p>正式实验覆盖 <b>{summary['cells_per_method']}</b> 个总体与参数组合，每种方法每单元 <b>{summary['trials_per_cell']}</b> 次，
四种方法共 <b>{summary['method_decisions']:,}</b> 次判定。合成阈值最大 100 万，含均匀、偏斜、稀有尾部、逼近边界及 Gap；另有五套真实频数定义的经验分布。</p>
<p><b>完整组合方法：</b>最大实测误报 {pct(summary['max_false_positive'])}，最大实测漏报 {pct(summary['max_false_negative'])}；
各单元同时单侧 95% 置信上界的最大值为 {pct(summary['max_simultaneous_upper95'])}。
所有承诺单元等权平均的预算为计数基线的 <b>{pct(summary['mean_relative_budget'])}</b>，
最不利单元为基线的 {pct(summary['worst_relative_budget'])}，改善并非每个分布都一样。</p>
<nav><a href="#algorithm">方法</a><a href="#proof">保证证明</a><a href="#experiments">实验</a>
<a href="#research">文献调研</a><a href="#conditions">前提</a><a href="#run">复现</a></nav>
<div class="limit"><b>准确定位：</b>这是新增的组合检验，不是论文第 3 节原算法，也没有证明缩小常数后仍有最优阶样本复杂度。
目前的最坏期望预算为 O((n+log(1/δ))/ε)。它解决的是带概率质量间隔的 Yes/No 检验，不能保证精确恢复任意分布的全部类别数。</div>
<h2 id="algorithm">用简单的话解释流程</h2>
<ol><li>分批获取样本，累计看见超过 n 类即可输出 No。</li>
<li>用论文的多尺度切比雪夫构造寻找尚未看见的类别；只在经过认证的阈值被越过时提前输出 No。</li>
<li>如果主要概率质量可能已被现有类别覆盖，冻结当前已见集合，再取一批独立样本。落在集合之外的次数足够少，才提前输出 Yes。</li>
<li>仍不能判断就继续，到有保证的计数预算后输出最终 Yes/No。输入文件提前耗尽则输出 Inconclusive。</li></ol>
<p>缩小常数是允许的，因为新流程重新控制了自己的误差。没有把“实验里没出错”当成原论文保证。</p>
<p>开发集选定 B={selected['b_factor']}log(en)、最多 {selected['max_degree']} 阶、权重幅度上限 {selected['weight_cap']}，
验证系数 {selected['validation_factor']}、验证门控 {selected['validation_gate']}。
门控中的 F₁/m 只决定是否投入验证成本，最终 Yes 必须依靠独立验证样本。</p>
<h2 id="proof">为什么两种错误都受控制</h2>
<p><b>设定。</b>n、ε、δ 在看本次样本前固定。训练集在预定均值 m₁&lt;…&lt;m_H&lt;m* 下由独立泊松增量累积。
验证样本来自同一分布的新 iid 观测。记 D 为训练集已见类别数，Fₗ 为出现 l 次的类别数。
所有配置可以在独立开发集上选择，但本次测试过程中不能据结果重新挑选 n、ε、权重或错误预算。</p>
<h3>1．提前输出 No：控制误报</h3>
<p>第 t 个时刻第 j 个尺度仍用原文的 f 函数与 Z=Σᵢf(Nᵢ)。设 A=ma、B=mb，逐尺度验证 0&lt;A&lt;B≤m 和 √(B(B−A))≥k。
Lemma 2.3 给出 E[f(Nᵢ)]≤1+ρ，其中 ρ=1/Tₖ((B+A)/(B−A))。使用实际 ρ，不要求截断阶数后仍有 ρ≤原文 δⱼ。</p>
<pre>R = max{{0,1,f(1),…,f(k)}} − min{{0,1,f(1),…,f(k)}}
u = R √[(n/2) log(H Kₜ / δ)]
只有 Z &gt; (1+ρ)n + u 时，才根据这个尺度输出 No。</pre>
<p>S≤n 时，补入贡献恒为 0 的虚拟类别，可看作 n 个有界独立变量。Hoeffding 界给出这个尺度误报≤δ/(H Kₜ)。
对每个时刻的 Kₜ 个尺度以及 H 个时刻求和，整个流程误报≤δ。各尺度、各检查时刻之间不需要独立。
直接看到 D&gt;n 的分支在 S≤n 时不可能发生。验证和收尾不会额外输出基于多项式的 No。</p>
<h3>2．提前输出 Yes：控制漏报的一半</h3>
<p>冻结当前已见集合 A，且 |A|≤n。在 No 情形下，任何这样的集合之外都至少有 ε 概率质量。
因此 q=P(X∉A)≥ε。用 r 个全新 iid 样本验证，V=Σ1{{X∉A}} 条件于此前历史服从 Bin(r,q)。
预先选整数 k，使 Pr(Bin(r,ε)≤k)≤δ/(2H)，仅在 V≤k 时输出 Yes。</p>
<p>二项分布左尾随 q 增大而减小，因此即使 A 和是否运行验证由训练历史决定，条件误判率仍≤δ/(2H)。
验证集不能用于调整 A 后又重复验证。H 个时刻求和，提前误判 Yes 的概率≤δ/2。</p>
<h3>3．计数收尾：控制另一半漏报</h3>
<p>No 情形下，只要还没发现第 n+1 类，下一次观测发现新类别的条件概率至少为 ε。
用独立均匀随机数耦合发现过程与成功概率 ε 的 Bernoulli 过程，得到：</p>
<pre>Pr(D≤n | 固定 M 次训练观测) ≤ Pr(Bin(M,ε)≤n)
M~Poisson(m*) 独立于标签，因此 Pr(D≤n) ≤ Pr(Poisson(m*ε)≤n)。
选 m* 使右侧≤δ/2；到 m* 后，D&gt;n 输出 No，否则输出 Yes。</pre>
<p>应使用完整潜在训练过程的无条件界：到达收尾且误判的事件，是完整过程 D≤n 事件的子集。
不能把“已经到达收尾”当作新条件后继续套未经调整的尾界。故总漏报≤提前 δ/2＋收尾 δ/2＝δ。</p>
<h3>4．样本成本和数值正确性</h3>
<p>每次训练增量在该阶段开始前独立抽取。有限阶段下，期望训练观测数等于停止时累计泊松均值的期望。
验证样本另外计入成本；实现限制 Σrₜ≤m*/5，所以期望总观测数≤1.2m*。
泊松尾界可由 m*ε≥n+O(√(n log(1/δ))+log(1/δ)) 满足，得到上述复杂度。</p>
<p>Arb 区间计算认证参数条件、ρ、权重范围以及两种尾概率。对 Z 的各系数取保守二进制有理下界，对拒绝阈值取上界，
统一分母后用整数比较。批量浮点运算只筛选候选 No，每个 No 都需整数复核；漏掉候选最多推迟判断。
采样模拟器使用 NumPy/SciPy 双精度随机实现，模拟结果不是对随机数发生器的形式化验证；实数误差界与整数判定则独立给出。</p>
<h2 id="experiments">独立实验和消融</h2>
<p>六个验证策略候选在 n=2,000/20,000、ε=0.05/0.1/0.2 的 60 个承诺场景开发集上选择，每单元 {calibration['trials_per_cell']} 次，
只按等权平均相对样本成本排序。正式测试固定协议、种子和源码哈希后运行，期间不调参。
开发与测试的分布家族有重叠；这里检验的是固定网格上的效果，不宣称对所有未知分布都获得同样的加速。</p>
<p>计数基线也允许在相同的预定检查时刻看到 D&gt;n 时提前结束，最终漏报预算为 5%。
组合方法的最终计数预算使用 2.5%，为提前 Yes 留出另外 2.5%；这是公平比较所需的错误预算差别。</p>
<p>下面的“平均预算”是累计训练泊松均值加实际验证次数，其均值估计期望实际观测成本，已经包含验证开销。
压缩模拟没有重建每次实际 M，不能把这些数字解读成某次恰好消耗的标签数。</p>"""]
    comparison = []
    for method, label in [("count", "计数基线"), ("count_validation", "仅增加独立验证"),
                          ("polynomial_count", "仅增加多项式早拒绝"), ("hybrid", "完整组合")]:
        rr = [r for r in rows if r["method"] == method and r["truth"] != "Gap"]
        comparison.append([label, pct(max(r["error_rate"] for r in rr if r["truth"] == "Yes")),
                           pct(max(r["error_rate"] for r in rr if r["truth"] == "No")),
                           pct(np.mean([r["relative_budget"] for r in rr if r["truth"] == "Yes"])),
                           pct(np.mean([r["relative_budget"] for r in rr if r["truth"] == "No"]))])
    chunks += [table(["方法", "最大误报", "最大漏报", "Yes 平均相对预算", "No 平均相对预算"], comparison),
               f'<img src="data:image/png;base64,{png}" alt="按规模和真值拆分的样本效率">',
               "<h3>n=1000、ε=0.1 的全部承诺场景</h3>"]
    subset = [r for r in rows if r["suite"] == "synthetic" and r["method"] == "hybrid" and r["n"] == 1000
              and r["epsilon"] == .1 and r["truth"] != "Gap"]
    chunks.append(table(["场景", "真值", "组合平均预算", "计数平均预算", "相对预算", "错误率"],
                        [[r["scenario"], r["truth"], f"{r['mean_budget']:.1f}", f"{r['count_baseline_mean_budget']:.1f}",
                          pct(r["relative_budget"]), pct(r["error_rate"])] for r in subset]))
    chunks.append("<h3>五个领域的完整经验分布，ε=0.1、No 边界查询</h3>")
    real = [r for r in rows if r["suite"] == "real_empirical" and r["method"] == "hybrid" and
            r["epsilon"] == .1 and r["scenario"] == "native_boundary"]
    chunks.append(table(["数据", "阈值 n", "组合平均预算", "计数平均预算", "相对预算", "漏报率"],
                        [[r["dataset"], r["n"], f"{r['mean_budget']:.1f}", f"{r['count_baseline_mean_budget']:.1f}",
                          pct(r["relative_budget"]), pct(r["error_rate"])] for r in real]))
    chunks.append(f"<p>五套数据共 {sum(d['records'] for d in datasets):,} 条有效观测。真实频数定义了有限经验分布，测试再从其中 iid 模拟。")
    chunks.append("原始文档、订单、评分和森林样方没有被自动当作 iid；S 是这些参考数据中的类别数，不是自然界全部类别数。来源和清洗沿用 <a href='../data/large/sources.json'>固定数据清单</a>。</p>")
    chunks += ["<h3>破坏独立性时会怎样</h3>", table(["每块相同标签数", "iid 保证适用", "实测漏报", "平均实际读取样本"],
                [[r["block_size"], r["guarantee_applies"], pct(r["false_negative_rate"]), f"{r['mean_actual_samples']:.1f}"] for r in stress]),
               "<p>这里从零售经验分布抽一个类别后重复 1/10/100 次，保持单条标签的边际分布，故意制造相邻依赖。每种 200 次，单独记录，未混入 iid 结果。它检验采样前提，不能拿失败时的数值套用 iid 保证。</p>",
               "<h2 id='research'>调研结论：哪些路线可以直接采用</h2>"]
    chunks.append(table(["来源", "对本任务的结论"], [[r["title"], r["finding"]] for r in REFERENCES]))
    chunks.append("<p>文献原文："+" · ".join(f'<a href="{r["url"]}">{html.escape(r["title"])}</a>' for r in REFERENCES)+"</p>")
    chunks += ["""<p>上面的组合规则和三部分误差分配是本项目在这些基础工具上的推导，并非声称某一篇文献已经给出了同一个算法。
后续若要保留最优阶，需要进一步证明缩小参数后的统一漏报界，或为连续概率域的优化生成可验证证书；本次没有把有限测试或网格约束当成该证明。</p>
<h2 id="conditions">保证的必要前提</h2>
<p>保证针对同一个固定分布的独立观测；训练的泊松随机化与数据独立；验证使用新样本；n、ε、δ 预先固定；
类别标签没有合并或系统性误标。Gap 区域不要求固定输出，Yes 也不意味着排除所有极罕见的额外类别。
若数据有时间漂移、重复批次或渔具选择性，需先明确独立采样单位和所观察的分布，不能靠调小常数修复模型问题。</p>
<h2 id="run">复现与完整结果</h2><pre>cd /home/snowy/OSST
.venv/bin/python guaranteed.py plan --n 1000 --epsilon 0.1 --delta 0.05 \
  --config results_guaranteed/selected_config.json
.venv/bin/python guaranteed.py demo --n 1000 --epsilon 0.1 \
  --config results_guaranteed/selected_config.json
.venv/bin/python guaranteed_experiments.py --trials 1000 --development-trials 100
.venv/bin/python -m pytest -q</pre>
<p><a href="results.csv">全部单元结果</a> · <a href="protocol.json">冻结协议与源码哈希</a> ·
<a href="calibration.json">开发集排名</a> · <a href="calibration_cells.csv">全部开发单元</a> ·
<a href="plans.json">各配置的认证预算</a> · <a href="dependence_stress.csv">相关性压力测试</a> ·
<a href="tests.log">测试日志</a> · <a href="sample_efficiency.pdf">可导出的图</a></p>
<h3>筛选完整组合方法结果</h3><label>搜索数据或场景：<input id="search" placeholder="例如 enron、diffuse、1000000"></label>
<div class="scroll" id="allrows">""",
               table(["数据", "场景", "n", "ε", "真值", "错误率", "平均预算", "相对预算", "同时置信上界"],
                     [[r["dataset"], r["scenario"], r["n"], r["epsilon"], r["truth"],
                       "不计错" if r["error_rate"] is None else pct(r["error_rate"]), f"{r['mean_budget']:.1f}",
                       pct(r["relative_budget"]), "—" if r["error_simultaneous_upper95"] is None else pct(r["error_simultaneous_upper95"])]
                      for r in rows if r["method"] == "hybrid"]),
               "</div><script>document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#allrows tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></body></html>"]
    (out/"report.html").write_text("\n".join(chunks))
    (out/"research_sources.json").write_text(json.dumps(REFERENCES, ensure_ascii=False, indent=2)+"\n")
