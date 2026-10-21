"""从 CSV/JSON 自动生成中文、自包含 HTML 实验报告，避免手写结果失配。"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from benchmark_data import DATA, ROOT
from run_large_benchmark import read_rows


def pct(x):
    return "—" if x is None else f"{100*x:.1f}%"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+"</tr></thead><tbody>"+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+"</tbody></table></div>"


def image_tag(out, stem, caption):
    blob = base64.b64encode((out / f"{stem}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{blob}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)} · <a href="{stem}.pdf">PDF</a></figcaption></figure>'


def generate_report(out: Path):
    summary = json.loads((out / "summary.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    datasets = json.loads((DATA / "dataset_summary.json").read_text())
    sources = json.loads((DATA / "sources.json").read_text())
    calibration = json.loads((out / "calibration.json").read_text())
    synthetic, real, efficiency, stress, richness = [read_rows(out/name) for name in
                                                   ["synthetic_large.csv", "real_large.csv", "sample_efficiency.csv", "dependence_stress.csv", "richness_curves.csv"]]
    trials = protocol["trials_per_test_cell"]
    selected = summary["config"]
    bounded_c4 = [r for r in synthetic if r["method"] == "bounded_multiscale" and r["factor"] == 4 and r["truth"] != "Gap"]
    max_fp = max(r["error_rate"] for r in bounded_c4 if r["truth"] == "Yes")
    max_fn = max(r["error_rate"] for r in bounded_c4 if r["truth"] == "No")
    max_upper = max(r["error_simultaneous_upper"] for r in bounded_c4)
    chunks = [f"""<p class="eyebrow">Optimal Support Size Testing · 工程修正版的可复现实验</p>
<h1>大规模支持集检验与多领域数据验证</h1>
<p class="finding"><b>算法身份：</b>本报告主要评测缩小参数并采用 Hoeffding 修正的工程版，不是论文第 3 节原算法。
严格原版已独立实现在 paper_reference.py，见 <a href="../results_audit/report.html">公式与函数调用核对</a>。
以下历史实验数值保留原样，不能用来宣称原版在同样的小预算下有论文保证。</p>
<p class="lead">五套公开数据共 <b>{sum(d['records'] for d in datasets):,}</b> 次有效观测，
合成阈值从 1,000 扩展到 1,000,000，最大构造支持集为 <b>{summary['max_synthetic_support']:,}</b> 类。
这次验证针对“支持集是否不超过 n，还是前 n 类之外仍有至少 ε 的质量”。</p>
<div class="finding"><b>主要结果：</b>在完整合成测试网格、C=4 的预算下，有限样本多尺度修正的最大实测误报为 {pct(max_fp)}、
最大实测漏报为 {pct(max_fn)}；相应单元的最大同时置信上界为 {pct(max_upper)}。
这是本次固定测试集的结果，不能代替对所有分布的最坏情形证明。真实频谱和自然分组相关性另行评估。</div>
<p>正式 iid 实验共 {summary['iid_cells_before_methods']:,} 个“总体 × 预算”单元，每个 {trials} 次，
共 {summary['iid_simulated_samples']:,} 个独立模拟样本，五种方法共享每个样本，产生 {summary['iid_method_decisions']:,} 次判定。
“一次模拟样本”是一批期望含 m 次观测的抽样，不是一条观测。开发集、正式测试和相关性压力测试使用不同随机子流。</p>
<nav><a href="#data">数据</a><a href="#design">实验设计</a><a href="#synthetic">规模实验</a><a href="#real">真实频谱</a>
<a href="#dependence">相关性</a><a href="#richness">类别数输出</a><a href="#explorer">完整结果查询</a><a href="#reproduce">复现</a></nav>
<h2 id="data">真实数据和参照总体</h2>
<p>这里的真值 S 指下载文件经过明示清洗后、频数大于零的类别数。将每类频数除以总次数，定义一个已知的经验分布，
再从这个分布独立有放回地泊松采样。这样可检验真实频谱下的算法行为；不能据此证明发现了外部世界所有鱼种、树种、词语或商品。
原始文档、用户评分、订单和空间样方也没有被直接假定为 iid。</p>"""]
    chunks.append(table(["数据", "领域", "有效观测次数", "正频数类别", "自然分组", "一条观测的含义"],
                        [[d["dataset"], sources[d["dataset"]]["domain"], f"{d['records']:,}", f"{d['support']:,}", f"{d['groups']:,}", d["sample_unit"]] for d in datasets]))
    for d in datasets:
        s = sources[d["dataset"]]
        chunks.append(f'<p><b>{d["dataset"]}</b>：{html.escape(d["filters"])}。{html.escape(d["limitations"])} '
                      f'<a href="{html.escape(s["page"])}">发布者说明</a>。'
                      f'引用：{html.escape(s["citation"])}；许可：{html.escape(s["license"])}。</p>')
    retail = next(d for d in datasets if d["dataset"] == "retail")
    chunks.append(f"""<p>零售清洗审计：原始 {retail['raw_rows']:,} 行，按固定优先级互斥剔除 {html.escape(str(retail['removed_rows']))}。
                  未把 Quantity 展开成互相独立的商品样本。BCI 全档案下载分别返回 HTTP 403/401，使用的是 vegan 提供的 1982 年 DBH≥10cm 子集；
                  没有通过合并年份扩大样本量。MovieLens 原始及派生数据只保存在本地，不进入分发包。</p>""")
    chunks.append(image_tag(out, "data_frequency_spectra", "完整经验分布的频率排序和累积覆盖率；S 与有效支持集大小并不相同"))
    chunks.append(f"""<h2 id="design">独立校准、真值与统计口径</h2>
<p>开发集使用 n=2,000 和 20,000、ε=0.1 和 0.2、四档预算及 10 个承诺场景，每单元 {calibration['trials_per_cell']} 次。
六个候选均保持相同的 5% 数学误报上界，按开发集等权平均漏报选择。
选定 B={selected['b_factor']} log(en)、最高阶数 {selected['max_degree']}、最大绝对权重 {selected['weight_cap']:g}。
正式测试阈值、分布参数、种子和所有真实数据均未用于这次选择。分布家族有重叠，测试规模和抽样独立；不称作所有家族都未见过的测试。</p>
<p>正式合成网格为 n∈{{10³,10⁴,10⁵,10⁶}}、ε∈{{0.05,0.1,0.2}}；
期望预算 m=ceil(Cn/(ε log(en)))，C∈{{0.5,1,2,4,8,16}}，另外加入保证漏报≤5%的计数预算。
11 个场景包括均匀、单一重头、极稀有尾部、分段幂律、分组对数正态、多尺度尾部，以及明确不计错的 Gap。
多尺度难例的 Δₙ 被构造成 1.01ε；不会只使用远离边界的容易样本。</p>
<p>真值 Yes 为 S≤n；No 为 Δₙ≥ε；其余 Gap 不计误差。
同一批 D,F₁,…,Fₖ 同时供五种方法使用。No 情形下的“未拒绝”记作漏报，以衡量检出能力；
实际低预算接口应将该输出解释为 Inconclusive，不能当作已经证实 Yes。Chao1 是点估计对照，将其直接阈值化不具有误报保证。</p>
<p>保守多尺度统计量使用实际多项式误差 ρ，减去 R√(n log(K/α)/2)，再除以 1+ρ，并与 D 取最大值。
尺度数 K 的联合界不需要尺度之间独立；α=0.05。各类别贡献独立需要泊松化 iid 模型。
它保留有限样本误报控制，缩小后的常数不继承论文的统一漏报或有效支持下界保证。数值运算也没有做形式化区间算术认证。</p>
<p>每个单元给出双侧 95% Wilson 区间；另外对全部 {summary['simultaneous_intervals_count']:,} 个正式承诺单元和方法，
使用 Bonferroni 校正的单侧 Clopper–Pearson 上界，整体覆盖至少 95%。这些区间只描述固定总体的重复抽样误差，不覆盖自然总体未知的类别数。
“最小通过预算”从预定网格中查找，要求该预算所有 10 个合成承诺场景的同时误差上界≤5%；缺失值表示未通过。</p>
<p>分组模拟不改变模型：若有 g 个等概率类别，则 (F₀,…,Fₖ,F&gt;ₖ) 服从以 Poisson(mp) 分箱概率为参数的多项分布。
它精确保留检验需要的统计量，避免生成数百万条标签或数亿个类别计数。高频部分的实际总样本数 M 未被重建，代码始终使用事先指定的 m。
此加速仅用于已知分布的模拟，不能减少现实取得观测的成本。</p>
<h2 id="synthetic">规模、难例和多尺度消融</h2>""")
    main_table = []
    for c in [.5, 1, 2, 4, 8, 16]:
        row = [f"C={c:g}"]
        for method in ["observed_count", "bounded_multiscale", "bounded_single_scale", "naive_chebyshev", "chao1_point"]:
            rr = [r for r in synthetic if r["factor"] == c and r["method"] == method]
            row.append(pct(max(r["error_rate"] for r in rr if r["truth"] == "Yes"))+" / "+pct(max(r["error_rate"] for r in rr if r["truth"] == "No")))
        main_table.append(row)
    chunks.append(table(["预算", "计数 FP / FN", "多尺度 FP / FN", "单尺度 FP / FN", "直接缩常数 FP / FN", "Chao1 FP / FN"], main_table))
    chunks.append('<p>上表为跨所有 n、ε、相应场景的最大实测误差，不用平均准确率掩盖难例。'
                  '单尺度对照只保留最小 a₀，并用其自身 K=1 的集中界；这是有利于单尺度的对照，不是人为套用多尺度惩罚。</p>')
    chunks.append(image_tag(out, "synthetic_worst_errors", "全部合成网格的最坏误差；直接缩常数和 Chao1 阈值化的误报不能忽略"))
    eff_table = []
    for n in [1000, 10000, 100000, 1000000]:
        rr = {r["method"]: r for r in efficiency if r["n"] == n and r["epsilon"] == .1}
        count_m, bounded_m = rr["observed_count"]["smallest_tested_passing_m"], rr["bounded_multiscale"]["smallest_tested_passing_m"]
        eff_table.append([f"{n:,}", f"{count_m:,}" if count_m else "未通过", f"{bounded_m:,}" if bounded_m else "未通过",
                          f"{count_m/bounded_m:.2f}×" if count_m and bounded_m else "—",
                          f"{rr['observed_count']['count_certified_m']:,}"])
    chunks.append(table(["n，ε=0.1", "计数最小通过 m", "多尺度最小通过 m", "网格预算比", "计数解析保证预算"], eff_table))
    chunks.append('<p>解析计数保证预算与“通过经验同时区间”的预算可以不同：解析保证对所有 No 分布成立，经验区间更保守且受有限重复次数影响。'
                  '这些比较支持有限规模的样本节省，不能从四个 n 值推出 Θ(n/(ε log n)) 的最坏情形最优性。最高阶数固定为 12，也不能无条件外推至 n→∞。</p>')
    chunks.append(image_tag(out, "sample_scaling", "ε=0.1 的通过预算与归一化预算；同时区间而非事后平均准确率决定通过"))
    chunks.append('<h2 id="real">在完整真实频谱上测试</h2><p>对每个经验分布和 ε∈{0.01,0.05,0.1}，'
                  'native_boundary 保留完整数据，取 n=effε(p)−1，构成接近边界的 No；'
                  'topn_null 是限制在前 n 类并重新归一化的人工 Yes 对照；native_null 使用完整数据且查询 n=S；native_gap 不计错。'
                  '下表固定展示全部五套数据的 ε=0.1、C=2 情形，并不按效果筛选数据集。</p>')
    rt = []
    for dataset in sources:
        rr = {r["method"]: r for r in real if r["dataset"] == dataset and r["scenario"] == "native_boundary" and r["epsilon"] == .1 and r["factor"] == 2}
        r = rr["bounded_multiscale"]
        null = next(x for x in real if x["dataset"] == dataset and x["scenario"] == "topn_null" and x["epsilon"] == .1 and x["factor"] == 2 and x["method"] == "bounded_multiscale")
        rt.append([dataset, f"{r['n']:,}", f"{r['m']:,}", f"{r['delta_n']:.5f}", pct(rr['observed_count']['error_rate']),
                   pct(r['error_rate']), pct(null['error_rate']), pct(r['error_simultaneous_upper'])])
    chunks.append(table(["数据", "n", "m", "实际 Δₙ", "计数漏报", "多尺度漏报", "匹配 Yes 误报", "多尺度漏报同时上界"], rt))
    chunks.append(image_tag(out, "real_boundary_errors", "每个数据集跨 ε 的最坏 No 未拒绝率；BCI 小支持集不一定从多项式中受益"))
    chunks.append('<p>零次观测到的误差不等于真实错误概率为零；查看表中区间和完整 CSV。词语数据的尾部已受发布者词频过滤影响，'
                  '不能把本结果推广成发现任意稀有词。BCI 与数字平台数据的表现不同，数据量大本身也不能消除稀有类别难题。</p>')
    chunks.append('<h2 id="dependence">自然分组相关性压力实验</h2><p>总样本数仍按 Poisson(m) 生成；'
                  '每 20 或 100 条观测共享一个文档、用户、订单或样方，按组内频率抽取标签。选择自然组的概率与组记录数成正比，'
                  '因此每条观测的边缘分布仍为原经验 p，只改变相关性。每个单元独立重复 '
                  f'{protocol["stress_trials"]} 次。这是受控的相关性模型，不是声称重建了原始时间顺序。</p>')
    st = []
    for dataset in sources:
        values = {r["block_size"]: r for r in stress if r["dataset"] == dataset and r["factor"] == 4 and r["truth"] == "No" and r["method"] == "bounded_multiscale"}
        st.append([dataset]+[pct(values[b]['error_rate']) for b in [1, 20, 100]])
    chunks.append(table(["数据，C=4，ε=0.1", "iid 漏报", "每 20 条同组", "每 100 条同组"], st))
    chunks.append(image_tag(out, "dependence_stress", "保证只适用于 iid；相同频谱在相关采样下可能表现不同"))
    chunks.append('<h2 id="richness">类别数输出应如何理解</h2><p>另按完整记录数的 0.1%、0.5%、1%、5%、10%、25%、50% 设置期望抽样预算，'
                  '比较观察到的类别数 D、Chao1 点估计和有误报控制的支持集下界。图中阴影是重复实验输出的 2.5%–97.5% 分位数，'
                  '不是未知 S 的置信区间。多尺度下界实验的 n 固定为不小于参考 S 的最小 2 的幂，记录在 CSV 中；'
                  '这是已知真值的实验设计，不把这个查询上限假装成算法估计出来的 S。</p>')
    rich_table = []
    for dataset in sources:
        rr = {r["method"]: r for r in richness if r["dataset"] == dataset and r["fraction"] == .05}
        r = rr["observed_count"]
        rich_table.append([dataset, f"{r['S']:,}", f"{r['m']:,}", f"{r['mean']:.1f}",
                           f"{rr['chao1_point']['mean']:.1f}", f"{rr['bounded_lower']['mean']:.1f}", pct(rr['chao1_point']['within_10pct'])])
    chunks.append(table(["数据，m≈5%参考记录", "参照 S", "m", "平均 D", "平均 Chao1", "平均保守下界", "Chao1 在±10%内"], rich_table))
    chunks.append(image_tag(out, "richness_curves", "下界、点估计、已见类别数的目标不同；不能把下界截断后的值用于 >n 检验"))
    chunks.append('<p>论文算法的目标是带概率质量间隔的检验与有效支持下界，不能保证精确恢复任意分布的全部类别。'
                  '这套实验能够检验决策误差、有限经验总体中的类别数表现和计算可行性；对真实湖泊未观测鱼种数量，仍需要独立采样设计和合适的生态观测资料。</p>')
    chunks.append('''<h2 id="explorer">查询完整正式实验结果</h2>
<p>每个方法均保留 Yes、No、Gap 结果。选择数据、n、ε、预算和方法即可查询，不只显示成功样本。原始机器可读文件见后文。</p>
<div class="controls"><label>数据 <select id="dataset"></select></label><label>n <select id="n"></select></label>
<label>ε <select id="eps"></select></label><label>预算 <select id="budget"></select></label>
<label>方法 <select id="method"></select></label></div><p id="matching"></p><div class="scroll" id="explorer-table"></div>''')
    chunks.append(f'''<h2 id="reproduce">复现、数据文件和检查</h2>
<pre>cd /home/snowy/OSST
.venv/bin/python run_large_benchmark.py --trials {trials} --calibration-trials {calibration['trials_per_cell']} --stress-trials {protocol['stress_trials']} --seed {protocol['test_seed']}</pre>
<p>首次运行自动下载固定文件；再次运行先验证 SHA256，再复用清洗缓存。新环境执行 <code>python -m pip install -r requirements.txt</code>。
原始数据保存在 <code>data/large/raw/</code>，类别频数与自然组稀疏矩阵在 <code>data/large/prepared/</code>。
便携包包含代码和结果，不包含跨领域原始或派生数据，复现时由脚本从发布者下载。</p>
<p><a href="protocol.json">评测前冻结的协议与代码哈希</a> · <a href="calibration.json">参数选择结果</a> ·
<a href="calibration_results.csv">全部开发结果</a> · <a href="synthetic_large.csv">完整合成结果</a> · <a href="real_large.csv">完整真实频谱结果</a> ·
<a href="sample_efficiency.csv">通过预算</a> · <a href="dependence_stress.csv">相关性压力结果</a> · <a href="richness_curves.csv">类别数曲线</a> ·
<a href="tests.log">测试日志</a> · <a href="summary.json">运行摘要</a></p>
<p>关键检查包括：分组抽样的均值与协方差和解析占位分布一致、矢量化统计量与原逐样本公式一致、
一亿类别不被合并丢失、所有构造分布的承诺标签正确、零售清洗守恒、五套数据的计数与分组矩阵一致、
原始文件及缓存哈希正确。现有公式和泊松化边界测试一并执行。</p>
<p class="muted">正式测试种子 {protocol['test_seed']}；协议版本 {protocol['version']}；结果完成于 {html.escape(summary['completed_at_utc'])}。
本报告从结果文件生成；任何不同配置的结论应以对应运行的 CSV 和协议为准。</p>''')
    # 仅嵌入展示需要的列，不把来源原始记录或用户信息放进 HTML。
    fields = ["dataset", "n", "epsilon", "budget_id", "method", "scenario", "truth", "S", "delta_n", "m",
              "error_rate", "error_ci95_low", "error_ci95_high", "error_simultaneous_upper"]
    compact = [[r[k] for k in fields] for r in synthetic+real]
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    script = '''<script>
const rows=DATA;
const ids=['dataset','n','eps','budget','method'];
const selects=ids.map(id=>document.getElementById(id));
function options(el, values, preferred){el.textContent=''; for(const value of values){const o=document.createElement('option');o.value=String(value);o.textContent=String(value);el.appendChild(o);} if(values.map(String).includes(String(preferred)))el.value=String(preferred);}
options(selects[0],[...new Set(rows.map(r=>r[0]))],'enron');
function update(from){
 let r=rows.filter(r=>r[0]===selects[0].value);
 if(from<1)options(selects[1],[...new Set(r.map(r=>r[1]))].sort((a,b)=>a-b),7108);
 r=r.filter(r=>String(r[1])===selects[1].value);
 if(from<2)options(selects[2],[...new Set(r.map(r=>r[2]))].sort((a,b)=>a-b),.1);
 r=r.filter(r=>String(r[2])===selects[2].value);
 if(from<3)options(selects[3],[...new Set(r.map(r=>r[3]))],'C=2');
 r=r.filter(r=>r[3]===selects[3].value);
 if(from<4)options(selects[4],['全部方法',...new Set(r.map(r=>r[4]))],'全部方法');
 if(selects[4].value!=='全部方法')r=r.filter(r=>r[4]===selects[4].value);
 const table=document.createElement('table'),head=document.createElement('tr');
 for(const name of ['场景','真值','S','Δₙ','m','方法','错误率','95% Wilson 区间','同时单侧上界']){const th=document.createElement('th');th.textContent=name;head.appendChild(th);}table.appendChild(head);
 const pc=x=>x===null?'—':(100*x).toFixed(2)+'%';
 for(const row of r){const tr=document.createElement('tr');const vals=[row[5],row[6],row[7],row[8].toFixed(5),row[9],row[4],pc(row[10]),pc(row[11])+' – '+pc(row[12]),pc(row[13])];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}table.appendChild(tr);}
 document.getElementById('explorer-table').replaceChildren(table);document.getElementById('matching').textContent='匹配 '+r.length+' 个实验单元；Gap 不定义错误率。';
}
selects.forEach((el,i)=>el.addEventListener('change',()=>update(i)));update(0);
</script>'''.replace("DATA", payload)
    css = '''body{font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f3f6f8;color:#172c3b;line-height:1.75;margin:0}main{max-width:1120px;margin:auto;background:white;padding:40px 5vw 80px}h1{font-size:2.1rem;line-height:1.3;letter-spacing:-.025em}h2{margin-top:48px;font-size:1.45rem;border-bottom:1px solid #d8e3e8;padding-bottom:10px}p{margin:16px 0}.lead{font-size:1.15rem}.eyebrow{color:#157e74;letter-spacing:.04em;font-weight:600}.finding{background:#e9f5f1;border-left:4px solid #157e74;padding:20px;margin:24px 0}nav{display:flex;flex-wrap:wrap;gap:14px;margin:25px 0}a{color:#166e99;text-decoration:underline;text-underline-offset:3px}table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:10px 12px;border-bottom:1px solid #dde5e9;text-align:left;vertical-align:top}th{background:#eef3f6;white-space:nowrap}.scroll{overflow:auto}figure{margin:24px 0}img{width:100%;height:auto;display:block}figcaption{font-size:.85rem;color:#566675;margin-top:7px}pre{overflow:auto;background:#122a3a;color:#eff5fa;padding:20px;font-size:.9rem;line-height:1.6;border-radius:6px}code{font-family:ui-monospace,monospace;font-size:.9em}.controls{display:flex;gap:12px;flex-wrap:wrap;background:#eef3f6;padding:15px}.controls label{font-size:.85rem}select{display:block;padding:8px;border:1px solid #b9cbd6;background:white;max-width:240px}.muted{color:#617383;font-size:.85rem}@media(max-width:700px){main{padding:24px 18px}h1{font-size:1.65rem}}@media print{body{background:white}main{padding:0}nav,.controls,#explorer-table,#matching{display:none}h2{break-after:avoid}figure{break-inside:avoid}}'''
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+\
               '<title>大规模支持集检验实验报告</title><style>'+css+'</style></head><body><main>'+''.join(chunks)+script+'</main></body></html>'
    path = out / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_large")
    args = parser.parse_args()
    print(generate_report(args.out.resolve()))
