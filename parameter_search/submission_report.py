"""Build the Chinese submission-readiness review from the read-only audit."""
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_submission_audit'


def main():
    stats=json.loads((OUT/'audit_summary.json').read_text())
    issues=[
      ('T01','P0','最优阶的 ε 范围','原论文 Theorem 1.1 未明确最优性所适用的 ε 区间。若把 Θ(n/(ε log n)) 理解为全 (0,1) 上的统一 minimax 下界，该解读有碰撞检验反例。','theory_scope.pdf，第 1 节；原 PDF 第 1 页','保留全 ε 的上界；将匹配下界单独写出准确范围、成功概率和 n 的条件。'),
      ('T02','P0','不能只替换原稿常数','原 Lemma 4.1 用 k_j<L/64 推出 ||f_j||∞≤4n^(1/16)。换成 C=70 后，k_0/L 趋近大于 5 的常数，旧前提不成立。新证明已采用另一套系数、范围和集中界，但必须真正并入论文。','原 PDF 第 7 页；results_parameter_search/proof.tex','保留旧证明作为旧参数版本，另陈述新常数定理；或整体替换旧 Lemma 4.1 及后续依赖。'),
      ('T03','P0','匹配下界的归约过于简写','WY19 是带最小概率条件的支持数估计下界，不是直接与本问题同模型的结论。必须写清困难先验、支持数间隔及产生 1/ε 的稀释归约。','原 PDF 第 1–2 页；WY19 Proposition 1；BGY26 Theorem 4.4','补一个明确引理及其归约，或准确引用已给出该模型和参数范围的定理。0<ε<1/32 是已查到的明确充分引用范围，不是声称最大适用范围。'),
      ('T04','P1','期望预算与最大样本量','70 是泊松均值系数，不是确定性的样本上限。原阶结论可容忍常数因子，新常数比较必须区分两种预算。','capped_certificate.json；theory_scope.pdf，第 2 节','已补上 M>ceil(2m) 时中止的耦合推论；八组误差目标均保留。最大预算系数写 140，期望系数写 70。'),
      ('T05','P0','提交稿尚不是自包含证明','新稿仍依赖早期工作目录中的期望、积分、系数、半直线和算术附录；相对 PDF 链接不等于正式提交的完整证明链。','results_parameter_search/proof.tex；results_proof_review/review.pdf','将必要引理与完整证明置于同一提交包，统一编号、符号、边界约定及最终常数表。'),
      ('T06','范围','70 的最优性不能扩大','整数最小性限于固定统计量、固定有限 L0>1、对 n 和 ε 统一的保证。实数最优值、最小 L0、固定 ε 的最优常数、最佳 CPU 时间均未确定。','results_parameter_search/proof.tex，必要界与 Scope','现有证明已正确限定；并稿时保留这些限定，尤其不能写“C0、L0 联合全局最优”。'),
      ('T07','范围','69 的渐近失效不是有限实验预测','必要下界的 ε_n 随 n 变化，CLT 是渐近论证。没有给出常用 n、固定 ε 下的具体失败阈值。','results_proof_review/review.pdf，第 4 节','保留 69 的零错误结果；不将其删去，也不声称当前实验已经观察到临界相变。'),
      ('E01','P0','当前预算无法显示二元检验的多项式优势','推荐参数所有单元的额外拒绝次数均为 0；同预算纯计数的二元及覆盖错误也均为 0。进一步证明在 n≤10^12 的整个范围内，纯计数失败已小于 1.863e-12。','reconstructed_events.csv；finite_counting_certificate.json','把当前实验目标定位为常数缩减、实现核验、支持下界强度与失效模式。不能靠追加相同预算试次证明二元样本优势。'),
      ('E02','P0','零错误的分辨率不足以验证小数点后的理论风险','每单元 2000 次；全 4424 单元校正后，零错误上界约 0.56791%，高于理论的 0.09% 和 0.0057%。异质单元合并不能直接给出最坏情形风险。','audit_summary.json，rare_event_power','理论界与经验区间分表。若要单独验证精细风险，预先固定少数实例和目标精度；不把数百万方法评估当成同一分布的重复。'),
      ('E03','P1','二元试验分母必须排除 Gap','(70,14) 的总覆盖试验是 1,994,000 次，其中二元承诺试验为 1,792,000 次；202,000 次属于 Gap。','audit_summary.json；paper_insert.tex 的表格','正文和表格同时列出两种分母。Gap 不计作二元成功，也不计作二元错误。'),
      ('E04','P0','自然数据没有进入多项式分支','10 个经验总体只有 4–240 个观测类别；76 个配置均为计数分支。原生自然数据的成功不能验证多项式效果。','data/c1025/datasets.json；audit_summary.json','明确命名 empirical-distribution resampling。扩展类别归入 synthetic。若论文不主张真实场景多项式优势，这项限制可诚实保留，不必强行寻找巨大自然数据。'),
      ('E05','P1','观测记录不等于未知总体真值或可用 iid 预算','10 个数据集共有 50,994 条原始记录；仅 13/76 个新配置的期望预算不超过原文件记录数。重采样次数不增加真实独立观测。','natural_budget_feasibility.csv','区分已知经验分布、未知自然总体、分组依赖与预算可行性；需要自然总体结论时另做采样设计和缺失质量分析。'),
      ('E06','P1','配对复用与“独立次数”','相同 C=70 的 L0 变体共享频数；4 个块各 500 次是 2000 个 MC 试次，不是算法做 4 次投票。','experiment_protocol.json；parameter_search/experiments.py','现有独立路径计数正确，写入论文时维持 5,116,000 条路径与 8,848,000 次方法评估的区别。'),
      ('E07','补齐','联合事件与多重比较','原报告明确按指标族分别给 95% 覆盖，这本身正确，但不能改写成所有指标共同 95%。本轮已从原始事件重建二元或覆盖失效的联合事件。','reconstructed_events.csv','论文可直接使用新的 joint_upper95_grid；另提供了二元与覆盖合并为一个指标族的修正界。不同声明族仍需在图注中注明。'),
      ('E08','P1','数值认证的样本增益不等于期望增益显著','50 个配置的样本平均下界提高。对 622 个配对主配置做 Hoeffding+Bonferroni 后，14 个配置可认证期望增益为正。','paired_gain_intervals.csv','分别报告数值范围、MC 不确定性和分布间泛化；其余配置不显著不代表无收益。当前补充分析属于事后审计，不能称为事前预注册结果。'),
      ('E09','P1','困难分布随 C 改变','各 C 的目标泊松均值构造和合法配置数量不同；各自最大错误率不能当作同一网格的风险曲线。C=70 与每个困难分布有同总体对照。','parameter_search/experiments.py:181；experiments.csv','保留各自压力测试的命名；若要比较 C 曲线，冻结共同分布集，再用新种子确认，全部失败结果照录。'),
      ('E10','P1','只与原大常数比较不足以支撑效率优势','1025 是本项目的历史常数，并非全部竞争算法。旧 (70,16) 结果中，1% 计数基线还少用约 2.44–8.39 倍样本。','results_c7016/summary.json，synthetic_cost_ratios','加入同风险目标的强计数基线、同预算消融，并与 FH/BGY 结果做模型和保证对照；没有忠实实现时保留理论比较，不伪造实测性能。'),
      ('E11','P0','不同算法版本的好结果不能混入原统计量','results_publication 的 31.2%/14.9% 标签流节省来自带验证和回退的 hybrid，证明复杂度为 O((n+log(1/δ))/ε)，并非当前 C=70 原统计量。','results_publication/manuscript.tex，第 3–4 节','这些结果放入独立的 practical extension 并给出自己的定理；主实验不要借用其加速结论。旧 27 次重复也只能作为历史分析，不能混进当前单次成本。'),
      ('E12','P1','现有时间字段不能直接做公平运行时排名','分组模拟跳过实际逐样本获取；相同采样时间记入多个方法；缓存与固定评估顺序影响编译和评分成本。','parameter_search/experiments.py:80–100；algorithm.py 的缓存','分别测冷启动、热缓存、逐样本计数、指纹评分、内存和数值重试，固定线程并交错方法顺序。已有 c7016 孤立计时可作基础，但尚非新网格完整基准。'),
      ('E13','P1','并非所有整数下界均已唯一求出','推荐参数有 37,071 次保存为整数区间。判定和覆盖事件仍可确定，不能写成每次完整整数输出都精确重建。','audit_summary.json；reconstructed_events.csv','区间端点保守分析已保留；若发布接口承诺唯一整数，则补做高精度细化并单列成本，不用中点代替。'),
      ('E14','范围','Arb 评分不认证随机采样器','分组指纹法在实数运算下等价；实际 SciPy/NumPy 的概率与 PRNG 用 binary64。PMF 总质量误差小不等于全指纹分布误差已全局控制。','c7016/populations.py:100；results_c7016/sampler_validation.json','当前已披露双精度，这是合理的科学计算范围。保留概率高精度抽查、矩检验与逐类别小规模交叉验证；不要把评分区间认证扩写为端到端精确随机数认证。'),
      ('E15','P1','同一权重递推的高精度复算不等于外部独立审查','Arb 与二进制评分交叉检查共享部分 reference_weights 逻辑。14,880 等计数是程序义务数，不是独立定理或专家审稿数量。','paper_reference.py:155；parameter_search/algorithm.py:140','以正系数闭式或另一种递推补充代表性对照，保持已有多实现算术审计；投稿前由作者及同领域研究者逐步核验关键新证明。'),
      ('E16','范围','协议有修订，但不是发现数据造假','已有冻结源、原始事件、精确真值、哈希和精度修订历史；本轮重建与保存结果一致。旧版边界修订也明确保留了 superseded 输出。','precision_amendment.json；results_publication/boundary_correction_audit.json','这些是可保留的证据优点。不能称外部预注册；文章需说明最终版本、修订原因和哪些结果用于探索。'),
      ('P01','P0','论文贡献主线要保持为原创理论','作者已说明原论文属于自己的成果。新增实验应服务于最优阶构造、pretty-good 下界和显式常数，而不是改写成他人论文的复现稿。','原 PDF Abstract/Theorem 1.1–1.3','主文保持原创定理；加入常数定理与可解释实验。避免以试验总次数代替贡献或把“实验验证定理”作为正确性来源。'),
      ('P02','P1','并行工作的贡献边界需具体','原稿已承认 BGY26 的并行最优支持检验。只说有限字母表与可数字母表不同，可能不足以独立支撑重大差异；需核对是否可经截断扩展。','原 PDF 第 5 页；BGY26 第 3 节和 4.1 节','比较直接计数接口、统计量、模型、pretty-good 数值保证、复杂度和显式常数，不淡化同期结果。此处是进一步比较任务，未证明两种方法等价。'),
      ('P03','待明确','投稿类型与原稿状态','当前可见公开预印本；这不等于已经确认从未正式发表或没有在审。用户尚未指定会议。','官方 CFP 链接见报告末尾','本审查按扩展现有预印本处理。若已有正式发表版本，须核对新稿的实质增量及 prior-publication 规则；若在审，先遵守该轮政策。'),
      ('P04','P1','AI 方法披露需反映新增工作的实际来源','原文已有 AI 使用说明。新阶段工具实际参与代码、推导、实验设计、数据分析和文字草稿，不能只沿用“头脑风暴/润色”的窄描述。','原 PDF Acknowledgment；STOC 2027 CFP','作者复核全部数学和结果后，按目标会议要求更新披露、责任与具体受影响章节。不是要求把 AI 列为作者。'),
    ]
    records=[dict(id=i,severity=s,title=t,finding=f,evidence=e,action=a) for i,s,t,f,e,a in issues]
    (OUT/'issues.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    inventory=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [
        r['suite'],r['cells'],f"{r['coverage_trials']:,}",f"{r['promised_binary_trials']:,}",
        f"{r['polynomial_trials']:,}",r['extra_rejects'],r['count_binary_errors'],r['count_coverage_errors']])+'</tr>' for r in stats['inventory'])
    issue_html=''.join(f'<article id="{r["id"]}"><h3>{r["id"]} · {r["severity"]} · {html.escape(r["title"])}</h3><p>{html.escape(r["finding"])}</p><p class="source">依据：{html.escape(r["evidence"])}</p><p><b>处理：</b>{html.escape(r["action"])}</p></article>' for r in records)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>原论文扩展投稿审查</title><style>
body{{font:16px/1.8 system-ui,sans-serif;color:#182332;max-width:1100px;margin:auto;padding:32px;background:#fff}}
h1,h2,h3{{line-height:1.4}} h2{{margin-top:2.2em;border-bottom:2px solid #e0e7ed;padding-bottom:.3em}}h3{{font-size:1.05em}}
a{{color:#006784}} .lead{{background:#eef6f8;padding:20px;border-left:4px solid #147d92}}.critical{{background:#fff4e9;padding:18px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow-x:auto}}
article{{border:1px solid #dce4e9;border-radius:7px;padding:14px 20px;margin:16px 0}}article p{{margin:8px 0}}.source{{font-size:14px;color:#536374}}
pre{{background:#f4f6f8;padding:16px;overflow:auto}}li{{margin:8px 0}}@media print{{body{{font-size:10pt;padding:0}}article{{break-inside:avoid}}}}
</style><h1>原论文扩展投稿审查</h1><p>2026-10-22 · 原理论、常数优化、原始实验、数值实现与论文整合</p>
<div class="lead"><b>结论：现有工作可以形成“原创理论 + 显式常数 + 有限规模实验”的扩展稿，但不能把现有报告直接拼进论文就视为已达到投稿完成状态。</b>
本轮没有发现原始试验计数与保存事件矛盾，也没有发现推翻新充分界或 70 整数最小性的错误。
新增的重要问题是原最优阶的参数范围、原证明不能只替换常数，以及当前预算下纯计数已经过于可靠，实验无法展示多项式的二元检验优势。</div>
<p>阅读入口：<a href="theory_scope.pdf">理论边界与补充推导 PDF</a> · <a href="paper_insert_preview.pdf">可并入论文的实验章节草稿</a> ·
<a href="paper_insert.tex">实验章节 LaTeX</a> · <a href="audit_summary.json">重建统计 JSON</a> · <a href="INTEGRATION.txt">整合与复现说明</a>。
这是对现有材料的研究审计，不是外部同行评审或录用保证。尚未确定目标会议与原稿正式发表状态，先按扩展现有预印本处理。</p>

<h2>一、优先修正的理论问题</h2>
<div class="critical"><b>最优阶需要写清 ε 的范围。</b> 若要求全 ε∈(0,1) 上都有统一的 Ω(n/(ε log n)) 下界，取 εₙ=1−n⁻² 就有反例：
抽取 t=⌈2√n⌉+1 个样本，出现碰撞则输出 Yes，否则 No。对支持≤n 的任意分布，无碰撞概率至多 exp[−t(t−1)/(2n)]≤e⁻²；
对 Δₙ≥1−n⁻² 的任意分布，∑pᵢ²≤n⁻³+n⁻⁴，故碰撞概率至多 C(t,2)(n⁻³+n⁻⁴)。两种错误都小于 1/3（n≥16）。
这只用 O(√n) 样本，严格小于该范围内的 n/log n。完整证明在 <a href="theory_scope.pdf">补充推导</a>。</div>
<p>修正办法是将全 ε 的充分上界和匹配下界分开。已核对的 BGY26 Theorem 4.4 明确给出充分大 n、0&lt;ε&lt;1/32 的对应下界；
可在准确引用其条件的基础上陈述这一范围的最优性。若要更广范围，应补足归约。新 Ccrit 反例使用 εₙ→0，所以仍支持这一小 ε 范围内 70 的整数最小性。</p>
<p><b>原证明的 kⱼ&lt;L/64 不能沿用到 C=70。</b> 需要把新的系数、范围和集中引理一起并入，不能只把算法框中的常数换掉。
另已补上泊松样本数的确定性上限转换：期望系数 70、最大预算 ceil(140n/(ε log(en)))，八组目标误差仍成立。</p>

<h2>二、现有实验实际说明了什么</h2>
<p>最新版共 4,424 个方法单元、8,848,000 次方法评估、5,116,000 条独立生成路径。对推荐的 (70,14)，覆盖试验为 1,994,000 次，
具有二元承诺的试验为 1,792,000 次；202,000 次 Gap 只评价覆盖。原始事件与 CSV、源代码、数据哈希均核对通过。</p>
<div class="scroll"><table><thead><tr><th>套件</th><th>单元</th><th>覆盖试验</th><th>二元承诺试验</th><th>多项式试验</th><th>多项式额外拒绝</th><th>同预算计数二元错误</th><th>同预算计数覆盖错误</th></tr></thead><tbody>{inventory}</tbody></table></div>
<p><b>这不是“还没找到足够难的自然数据”那么简单。</b> 本轮给出有限范围证书：所有 n≤10¹²、ε∈(0,1)，同预算纯计数失败概率已经小于
1.863×10⁻¹²。因此当前规模下继续增加同预算试次，不能有效展示多项式的二元优势。可保留的正面结果是支持下界更强：442,000 次试验获得比计数更大的认证下界。</p>
<p>降低 L0 的效果也应分层描述：622 个配对主配置中，50 个样本平均增益为正；同时 95% 的保守 Hoeffding 界能确认其中 14 个的期望增益为正。
这是对已有数据的补充分析，区间刻画模拟重复的随机性，不代表对未知分布集合的泛化保证。</p>

<h2>三、零错误不够验证精细误差值</h2>
<p>每格只有 2,000 次。全网格同时 95% 上界约为 0.56791%，而理论目标为 0.09% 或 0.0057%。不能把零错误解释为实测达到理论小数位。
若一格仍为零错误，要使单格 95% 上界低于 0.09%，至少需要 3,328 次；低于 0.0057%，至少 52,556 次。
若同时覆盖原来的 4,424 格，分别约需每格 12,651 和 199,829 次。这只是分辨率计算，不是建议盲目扩充全部网格。</p>
<p>原报告已正确注明“按指标族分别控制”。为匹配理论的同时正确输出，本轮另从原始布尔事件重建联合失效，提供
<a href="reconstructed_events.csv">联合风险及统一指标族的置信界</a>。不能把多个各自 95% 的声明自然合并成整篇论文共同 95%。</p>

<h2>四、自然数据、基线与计算成本</h2>
<p>自然数据总记录数 50,994，观测类别数 4–240，全部进入计数分支。只有 13/76 个新配置的期望预算不超过原文件记录数；
即使满足均值条件也不保证一次泊松抽样一定能从文件无放回完成。必须保留“经验分布 iid 重采样”的表述；分裂类别属于构造数据。
当前工作可以作为理论论文的经验形状验证，不应声称测出了未知生态总体的真实支持数。</p>
<p>与 1025 或 2²⁶ 的预算相比，70 的常数缩减是真实的理论改进，但不足以证明实践上优于强计数基线。
旧版同预算公式的比较已显示，1% 计数基线还可少用约 2.44–8.39 倍样本。现有 hybrid 的标签流节省属于另一个算法，不能挪作原统计量的性能。</p>
<p>当前时间字段是分组模拟与缓存评分；不能冒充实际采集 10¹² 乃至更大期望样本的耗时。标准运行时实验应把样本读取、指纹构造、冷编译、热评分、数值细化和内存分开。</p>

<h2>五、逐项问题、证据与处理</h2>
<p>P0：影响核心陈述或不应原样进入稿件。P1：应修正、补强或准确披露。范围：已有说明应保留，不等于发现算法错误。补齐：本轮已提供派生分析。
不要求把所有补充实验都做成主文结果；先决定论文要主张什么，再完成与该主张对应的证据。</p>{issue_html}

<h2>六、建议的论文整合结构</h2>
<ol><li><b>主文保留原创贡献：</b>最优阶支持检验、pretty-good 支持下界及学习理论含义。把匹配下界的 ε 范围写清。</li>
<li><b>增加显式常数小节：</b>参数化构造、(70,14)/(70,16) 的单次保证、期望/封顶样本数。将同构造整数最优性作为单独定理，避免“所有算法最优常数”的误读。</li>
<li><b>增加 2–4 页实验：</b>实验定义与来源；构造数据及边界；支持下界增益；较小 C 的失效；自然经验总体；实际可行性及负面结果。</li>
<li><b>附录合并证明和复现材料：</b>期望/系数/集中/无穷范围/必要界；所有数据来源、冻结设计、修订历史、完整结果、数值实现和运行环境。</li>
<li><b>单独放置工程扩展：</b>若保留验证/回退 hybrid，需要自己的算法定义、保证和基线，不能用来实证支撑原统计量的全部结论。</li></ol>
<p><a href="paper_insert_preview.pdf">已写好的英文实验节</a>只使用当前可追溯数据和本轮派生统计，包含负面结果；没有补写未执行的实验，也没有声称稿件已可录用。
原稿目前以 PDF 提供，实验 LaTeX 为独立插入块，原 PDF 和冻结结果均保持不变。</p>

<h2>七、补做工作的优先级与验收标准</h2>
<table><tr><th>顺序</th><th>工作</th><th>验收条件</th></tr>
<tr><td>1</td><td>修正最优阶范围、匹配下界与旧/新证明衔接</td><td>每项主定理有明确模型、量词、参数范围及闭合的证明依赖；不靠实验补数学漏洞。</td></tr>
<tr><td>2</td><td>将当前派生结果并稿</td><td>二元/覆盖分母正确；明确计数也成功；列出 50 个样本增益与 14 个期望增益；Gap、区间输出、经验总体说明准确。</td></tr>
<tr><td>3</td><td>有明确目标的实际成本基准</td><td>相同错误目标与输入；冷/热缓存分开、方法顺序交错、固定线程、记录内存与失败。算法版本逐项对应。</td></tr>
<tr><td>4</td><td>若主张实用二元优势，再开探索与确认</td><td>降低预算的参数属于未获统一保证的探索；独立开发/冻结测试；强计数与忠实文献基线；同时测误报漏报。允许结果是无优势。</td></tr>
<tr><td>5</td><td>必要时提高小风险测量分辨率</td><td>先选定少数问题和精度，再固定样本量/顺序有效区间；不得看到零错误后任意停止并套固定样本区间。</td></tr>
<tr><td>6</td><td>可移植复现与人工同行检查</td><td>锁定实际依赖；干净环境复现一套小实验与所有数学证书；作者能独立讲清 Bernstein 与 CLT 两步。</td></tr></table>
<p>步骤 3–5 是否必需取决于最终主张。若目标是理论贡献为主的稿件，诚实保留“有限规模未显示二元优势”通常比制造一条有利曲线更有价值。
本轮没有执行新的大规模模拟，补做方案不能当成已有结果。</p>

<h2>八、会议定位与核对来源</h2>
<p>“理论正确 + 补实验”不是 CCF A 录用充分条件。以本文内容判断，理论会议更贴近其核心贡献；这是适配性判断，不是录用预测。
<a href="https://acm-stoc.org/stoc2027/stoc2027-cfp.html">STOC 2027 官方 CFP</a>要求主数学结论可完整核验，并重视前 10–12 页的贡献和论证表达，未把大规模实验列作一般必要条件。
若选择机器学习会议，应把统计问题、相关方法和实验结论写得足够具体；<a href="https://neurips.cc/public/guides/PaperChecklist">NeurIPS checklist</a>要求主张、假设、复现和误差条等信息相互一致。</p>
<p>STOC 2027 的实质 AI 使用披露、公开投稿、视频及 prior-publication 政策与以往不同，应以本届 CFP 为准；本轮仅查阅规则，没有投稿或联系任何人。
目标会议待作者确定；正式发表或在审状态也须由作者确认。CCF 目录按正式 full/regular paper 区分会议成果，见
<a href="https://www.ccf.org.cn/Academic_Evaluation/By_category/">CCF 当前官方目录说明</a>。</p>
<ul><li><a href="https://arxiv.org/abs/2610.18723">原论文公开版本</a>：本地原 PDF 是本轮审查基准。</li>
<li><a href="https://arxiv.org/pdf/1504.01227">Wu–Yang 支持数估计论文</a>：检查其模型与 Proposition 1 的参数范围。</li>
<li><a href="https://arxiv.org/html/2609.37926v1#S4.SS1">Bao–Gao–Yao 并行工作，第 4.1 节</a>：明确的经典支持检验下界范围与附录归约。</li>
<li><a href="https://arxiv.org/pdf/2410.18915v4">Ferreira Pinto Jr.–Harms 完整版本</a>：同问题基线及保证，须区分忠实算法和调参变体。</li></ul>
</html>'''
    (OUT/'review_report.html').write_text(page)
    print(f'Report built: {len(records)} evidence-linked review items')


if __name__=='__main__':main()
