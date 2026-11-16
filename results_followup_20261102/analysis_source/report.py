"""Standalone Chinese report, exportable figures and hashed evidence manifest."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from followup.protocol import ROOT,OUT,save,sha,csv_save


def read(path):
    return list(csv.DictReader(Path(path).open()))


def integer(r,k):return int(r[k])
def number(r,k):return float(r[k])


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def figure(out,name):
    plt.tight_layout()
    for ext in ['png','pdf']:plt.savefig(out/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close()
    return f'<figure><img src="figures/{name}.png"><figcaption><a href="figures/{name}.pdf">下载 PDF 图</a></figcaption></figure>'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args(argv);out=args.out.resolve()
    audit=json.loads((out/'audit.json').read_text());assert audit['passed']
    protocol=json.loads((out/'protocol.json').read_text())
    rows=read(out/'audited_results.csv');gains=read(out/'paired_gains.csv');frontier=read(out/'empirical_frontier.csv')
    mechanism=[r for r in rows if r['suite']=='mechanism'];budget=[r for r in rows if r['suite']=='budget']
    full=[r for r in mechanism if r['method']=='full14']
    risk=[]
    for suite,rr in [('mechanism',mechanism),('budget',budget)]:
        for method in sorted({r['method'] for r in rr}):
            chosen=[r for r in rr if r['method']==method];promised=[r for r in chosen if r['truth']!='Gap']
            risk.append(dict(suite=suite,method=method,cells=len(chosen),
                trials=sum(integer(r,'trials') for r in chosen),
                promised_trials=sum(integer(r,'trials') for r in promised),
                binary_errors=sum(integer(r,'errors') for r in promised),
                coverage_errors=sum(integer(r,'coverage_errors') for r in chosen),
                worst_binary_rate=max(integer(r,'errors')/integer(r,'trials') for r in promised),
                worst_binary_upper95=max(number(r,'binary_upper95_simultaneous') for r in promised),
                extra_correct_rejects=sum(integer(r,'extra_rejects') for r in chosen if r['truth']=='No'),
                extra_false_rejects=sum(integer(r,'extra_rejects') for r in chosen if r['truth']=='Yes'),
                extra_rejects=sum(integer(r,'extra_rejects') for r in chosen)))
    csv_save(out/'risk_summary.csv',risk)
    paired=[]
    for comparison in sorted({r['comparison'] for r in gains}):
        chosen=[r for r in gains if r['comparison']==comparison]
        paired.append(dict(comparison=comparison,cells=len(chosen),
            positive_sample_mean_cells=sum(number(r,'sample_mean_lo')>0 for r in chosen),
            positive_expected_gain_cells=sum(number(r,'expected_gain_lo95_simultaneous')>0 for r in chosen),
            max_normalized_mean_gain=max(number(r,'sample_mean_lo') for r in chosen)))
    plans=read(out/'budget_plans.csv');ratios=[number(r,'ratio_to_count_01') for r in plans if r['method']=='C70']
    matched_ratios=[]
    for r in plans:
        if r['method']=='C70':
            baseline=next(b for b in plans if b['n']==r['n'] and b['epsilon']==r['epsilon'] and b['method']=='count_risk_.0009')
            matched_ratios.append(number(r,'mean')/number(baseline,'mean'))
    summary=dict(**audit,mechanism_recommended_trials=sum(integer(r,'trials') for r in full),
        mechanism_recommended_binary_trials=sum(integer(r,'trials') for r in full if r['truth']!='Gap'),
        mechanism_recommended_errors=sum(int(r['errors'] or 0) for r in full),
        mechanism_recommended_coverage_errors=sum(integer(r,'coverage_errors') for r in full),
        mechanism_extra_rejects=sum(integer(r,'extra_rejects') for r in full),
        mechanism_certified_lower_improvements=sum(integer(r,'improvements') for r in full),
        risk_by_method=risk,paired=paired,C70_vs_count_01_ratio=[min(ratios),max(ratios)],
        C70_vs_count_same_0009_risk_ratio=[min(matched_ratios),max(matched_ratios)])
    summary['total_fresh_paths_including_guard']=audit['fresh_population_budget_paths']+(audit.get('post_primary_guard') or {}).get('independent_paths',0)
    save(out/'summary.json',summary)
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OSST 补充实验 · 2026-11-02</title>',
    '<style>body{font:17px/1.75 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172536;background:#fbfcfe}h1,h2{line-height:1.3}h2{margin-top:2.4em}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d6dee8;padding:9px;text-align:left}th{background:#eaf0f8}img{max-width:100%}a{color:#205cb3}.box{background:#edf3fa;border-left:4px solid #3272ad;padding:18px}code{background:#eaf0f8;padding:2px 5px}figure{margin:24px 0}small{color:#526376}</style>',
    '<h1>支持大小检验：机制、公平预算与完整数值输出</h1>',
    '<p>本报告对应独立目录 <code>results_followup_20261102</code>。设计参考此前研究，使用新种子；不是外部预注册。历史冻结实验保持不变。报告只陈述实际运行结果，不将缩预算探索解释为新的统一概率定理。</p>',
    '<div class="box">'+f"新增 {audit['fresh_population_budget_paths']:,} 条独立总体—预算模拟路径，{audit['method_trial_evaluations']:,} 次方法评估。"
    +f"机制主方法 (70,14) 二元错误 {summary['mechanism_recommended_errors']:,} 次，覆盖失效 {summary['mechanism_recommended_coverage_errors']:,} 次。"
    +f"相对同预算计数，额外拒绝 {summary['mechanism_extra_rejects']:,} 次；可认证的整数下界提高 {summary['mechanism_certified_lower_improvements']:,} 次。"
    +f"历史 {audit['refined_archived_outputs']:,} 个非唯一整数输出全部完成细化，原判定和覆盖事件没有改变。</div>",
    '<h2>1. 冻结设计和证据口径</h2>',
    table(['实验','设计','回答的问题'],[
        ['机制与消融',f"n={protocol['design']['mechanism']['ns']}；ε={protocol['design']['mechanism']['epsilons']}；每格 {protocol['design']['mechanism']['trials']} 次",'同一批样本上比较 full14/full16、单尺度、计数与去掉计数分支'],
        ['共同分布预算网格',f"C={protocol['design']['budget']['constants']}；每格 {protocol['design']['budget']['trials']} 次",'共同总体上检验小预算风险；与 Arb 认证的 1% 及 0.09% 计数预算比较'],
        ['历史整数细化','所有 37,071 个历史区间输出；独立新增文件','整数唯一性与高精度成本；不是新增随机试次'],
        ['完整接口与计时','逐类别泊松、真实 iid 标签、交错计时、正系数闭式核验','从采样到频数和唯一整数输出；区分编译、计数与评分成本']]),
    '<p>Yes 为 S≤n；No 为 Δₙ≥ε；Gap 只评估支持下界覆盖。方法复用样本时不重复计为独立路径。计数只给出保守下界，单尺度及去掉 D 的版本是消融，不继承完整统计量的双侧保证。不同预算采用独立子流；C=70 的两种 L₀ 共享同一频数。</p>',
    '<h2>2. 多尺度到底增加了什么</h2>',
    table(['方法','承诺二元试次','二元错误','覆盖错误','最坏二元同时 95% 上界'],[
        [r['method'],f"{r['promised_trials']:,}",r['binary_errors'],r['coverage_errors'],f"{100*r['worst_binary_upper95']:.4f}%"] for r in risk if r['suite']=='mechanism']),
    table(['配对比较','配置数','正样本均值配置','正期望增益可认证配置','最大平均增益 / n'],[
        [r['comparison'],r['cells'],r['positive_sample_mean_cells'],r['positive_expected_gain_cells'],f"{r['max_normalized_mean_gain']:.6f}"] for r in paired]),
    '<p>期望增益界使用嵌套最大值产生的 [0,1] 归一化配对增量，Hoeffding 加 Bonferroni 同时校正所有机制比较。先保留数值区间，再计算统计区间；没有把区间中点当作精确输出。未显著不等于没有收益。该置信声明族与风险声明族分别控制 95%。</p>']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for comparison in ['full14-count70','full14-single14','full14-full16']:
        chosen=[r for r in gains if r['comparison']==comparison]
        ns=sorted({integer(r,'n') for r in chosen});means=[]
        for n in ns:
            rr=[r for r in chosen if integer(r,'n')==n]
            means.append(np.mean([number(r,'sample_mean_lo') for r in rr]))
        axes[0].plot(ns,means,'o-',label=comparison)
    axes[0].set(xscale='log',xlabel='n',ylabel='Mean paired lower-bound gain / n',title='Equal weight per frozen scenario / epsilon');axes[0].legend(fontsize=8)
    winners=read(out/'scale_winners.csv');wc={}
    for r in winners:wc[integer(r,'winner_column')]=wc.get(integer(r,'winner_column'),0)+integer(r,'trials')
    xs=sorted(wc);axes[1].bar([('ambiguous' if j<0 else 'D' if j==0 else f'j={j-1}') for j in xs],[wc[j] for j in xs])
    axes[1].set(ylabel='Trials',title='Certified winning branch (overlap is not guessed)');axes[1].tick_params(axis='x',rotation=45)
    html.append(figure(out,'mechanism'))
    if audit.get('post_primary_guard'):
        guard=read(out/'count_guard/audited_results.csv')
        html+=['<h3>独立标注的追加诊断：为什么仍需保留 D</h3>',
        '<p>主网格的 no_count 消融没有覆盖失效。观察到这一结果后，另冻结 ε=1/(4n) 的边界诊断；这是事后提出的新机制实验，使用独立新种子与单独协议，不混入主网格置信声明。两个 n、两种总体各 2,000 次，共新增 8,000 条路径、24,000 次方法评估。</p>',
        '<p>采用均匀分布及最稀有类别质量为 2ε 的分布，均有 effε=S=n。完整算法和计数仍满足覆盖；去掉 D 时，保守修正使输出低于 n，四个单元均出现 100% 的覆盖失效。这是数值下界失效，二元 Yes 判断仍正确。巨大泊松均值通过分组频数模拟，不代表获取了同等数量的现场观测。</p>',
        table(['n','总体','去掉 D 后平均整数下界','真实要求 effε','覆盖失败','覆盖失败率单独同时 95% 下界'],[
            [r['n'],r['scenario'],r['mean_lower_lo'],r['effective'],r['coverage_errors']+'/'+r['trials'],
             f"{100*number(r,'coverage_lower95_simultaneous'):.3f}%"] for r in guard if r['method']=='no_count']),
        '<p><a href="count_guard/protocol.json">追加诊断协议</a> · <a href="count_guard/audited_results.csv">逐配置结果</a></p>']
    html+=['<h2>3. 同一风险目标与共同分布的预算比较</h2>',
      f'<p>在当前网格上，认证 C=70 的期望预算为 1% 强计数预算的 <b>{min(ratios):.3f}—{max(ratios):.3f} 倍</b>。计数预算通过泊松尾界的 Arb 认证；另提供 0.09% 计数预算，便于与 (70,14) 的已证风险上界直接比较。没有为了制造优势削弱计数基线。</p>',
      f'<p>将计数也设为同一 0.09% 风险上界时，C=70 的预算仍为计数预算的 <b>{min(matched_ratios):.3f}—{max(matched_ratios):.3f} 倍</b>。两种计数比较均以期望泊松样本量衡量，不是 CPU 加速比。</p>',
      '<p>理论风险保证与模拟置信上界分开解读：一个具有≤1%数学保证的基线，有限重复的置信上界仍可能高于1%，这不构成理论反例。缩预算多项式的额外拒绝可能包含误报；risk_summary.csv 分开记录 No 情况的额外正确拒绝和 Yes 情况的额外错误拒绝。</p>',
      table(['方法','最坏观测二元错误率','最坏二元同时 95% 上界','保证状态'],[
          [r['method'],f"{100*r['worst_binary_rate']:.3f}%",f"{100*r['worst_binary_upper95']:.3f}%",
           '统一理论保证' if r['method']=='C70' or r['method'].startswith('count_risk_') else '消融／探索；仅限网格证据'] for r in risk if r['suite']=='budget']),
      '<p>下表的“首个通过”要求同一 n、ε 下所有承诺分布的同时置信上界均≤1%。全部候选预算已纳入多重比较；没有假设多项式风险随 C 单调。它是有限共同分布集上的通过点，不是分布无关的样本复杂度定理。固定困难总体由参考 C=60 定义一次，之后各方法面对完全相同的精确概率。</p>',
      table(['n','ε','方法族','首个通过 C','期望预算','右删失'],[[r[k] for k in ['n','epsilon','family','first_passing_C','first_passing_mean','right_censored']] for r in frontier])]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for is_count,label in [(False,'Full statistic (exploratory below 70)'),(True,'Same-budget observed count')]:
        rr=[r for r in risk if r['suite']=='budget' and r['method'].startswith('C') and r['method'].endswith('_count')==is_count]
        rr.sort(key=lambda r:int(r['method'].removeprefix('C').removesuffix('_count')))
        xs=[int(r['method'].removeprefix('C').removesuffix('_count')) for r in rr]
        ax.plot(xs,[r['worst_binary_upper95'] for r in rr],'o-',label=label)
    ax.axhline(.01,color='black',ls='--',label='1% target');ax.set(xlabel='C',ylabel='Worst simultaneous 95% binary risk upper bound',yscale='log');ax.legend(fontsize=9)
    html.append(figure(out,'budget_risk'))
    html.append('<p>图中每个点对应一个实际测试的 C，连线仅辅助阅读，不表示未测预算上的插值保证；纵轴取共同网格中最坏的单元置信上界。</p>')
    html+=['<h2>4. 完整整数输出与独立数值核验</h2>',
      table(['指标','结果'],[
          ['历史区间输出细化',f"{audit['refined_archived_outputs']:,} / 37,071"],
          ['改变二元判定／覆盖事件','0 / 0'],
          ['细化评分总时间',f"{audit['refinement_seconds']:.3f} 秒（不含原文件加载）"],
          ['每次细化中位／95% 分位时间',f"{audit['refinement_seconds_median']*1000:.4f} / {audit['refinement_seconds_p95']*1000:.4f} 毫秒"],
          ['最大使用精度',f"{audit['refinement_max_bits']} bits"],
          ['独立闭式系数核验',audit['independent_coefficient_checks']],
          ['完整接口核验次数',audit['full_api_runs']]]),
      '<p>闭式核验采用 Tₖ(1+z) 的正系数展开，独立于生产代码的 Chebyshev 递推。完整接口显式要求预定泊松均值，分别记录实际 M；真实标签场景激活了多项式分支。所有整数必须唯一认证，未用中点替代。样本生成器仍是 NumPy/SciPy 双精度伪随机实现，Arb 评分认证不等于端到端精确采样认证。</p>']
    integration=json.loads((out/'integration.json').read_text())['runs']
    html.append(table(['输入路径','n','实际 M','D','输出','整数下界','评分秒'],[
        [r['kind'],r['n'],r['M'],r['observed'],r['decision'],r['lower'],f"{r['inference_seconds']:.5f}"] for r in integration]))
    html+=['<h2>5. 运行成本</h2>',
      '<p>冷编译和热评分分别测量；冷编译每配置单次，是开销示例。热评分每配置7轮、方法顺序每轮随机打散，固定 BLAS/OMP 单线程；内存追踪在计时之外执行。tracemalloc 峰值是可追踪分配，不是完整进程 RSS。模拟生成、真实标签计数和物理采样成本不混合。分组模拟的时间不能作为采集上亿观测的时间。</p>']
    html.append('<p>热评分使用同一 full14 频数矩阵。完整算法计时包含输入校验；count 一列只从已知 D 生成整数下界，不重新扫描低频矩阵。full16 的计数分支仍承担共享矩阵的接口校验成本。这是当前接口的局部开销实测，不代表各方法独立优化后的端到端运行时排名。</p>')
    timing=read(out/'runtime.csv');cold=read(out/'cold_compile.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in ['count','full14','full16','all_scale_diagnostics']:
        rr=[r for r in timing if r['method']==method and integer(r,'rep')>=0];ns=sorted({integer(r,'n') for r in rr})
        med=[statistics.median(number(r,'seconds')/integer(r,'trials') for r in rr if integer(r,'n')==n) for n in ns]
        axes[0].plot(ns,med,'o-',label=method)
    axes[0].set(xscale='log',yscale='log',xlabel='n',ylabel='Hot scoring seconds / fingerprint');axes[0].legend(fontsize=8)
    for L0 in [14,16]:
        rr=sorted([r for r in cold if integer(r,'L0')==L0],key=lambda r:integer(r,'n'))
        axes[1].plot([integer(r,'n') for r in rr],[number(r,'seconds') for r in rr],'o-',label=f'L0={L0}')
    axes[1].set(xscale='log',yscale='log',xlabel='n',ylabel='Cold parameters + weights seconds');axes[1].legend()
    html.append(figure(out,'runtime'))
    html+=['<h2>6. 能支持与不能支持的结论</h2>',
      '<p>这组实验补充最新算法的机制、共同网格功效、强计数对照和完整数值接口证据。原论文的渐近最优性、ε 适用范围与新常数充分性仍依赖数学证明。有限网格里 C&lt;70 通过或 C=69 不失败，不反驳全参数渐近必要界。去掉 D 的消融失效只解释组合结构，不是原算法失败。</p>',
      '<p>没有新增现场调查，因此不支持未知生态总体的支持真值；没有将类别复制当作新自然数据。旧 practical hybrid 的节省样本结论仍独立归属其算法。没有加入非忠实 FH/BGY 实现来制造实测排名。所有零错误同时报告置信上界，没有把多方法评估总量当作单一分布的重复数。</p>',
      '<h2>7. 复现与文件</h2><p><a href="REPRODUCE.txt">完整复现命令</a> · <a href="summary.json">机器可读摘要</a> · <a href="protocol.json">冻结协议与源码哈希</a> · <a href="audit.json">审计结果</a> · <a href="audited_results.csv">全部方法结果</a> · <a href="paired_gains.csv">配对增益区间</a> · <a href="empirical_frontier.csv">经验通过预算</a> · <a href="integer_refinement.csv">历史整数细化</a> · <a href="manifest.json">文件清单</a></p>',
      '<p>原始频数、全部方法事件、尺度区间和数值细化输出保存在 raw/；精确总体与每格哈希保存在 cells/；执行源码在 source_snapshot/。小规模 smoke 结果不计入正式试验。</p></html>']
    (out/'report.html').write_text('\n'.join(html))
    (out/'REPRODUCE.txt').write_text('''工作目录：/home/snowy/OSST

新目录完整复现（固定种子将复现同一组伪随机试验；计时不逐字相同）：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m followup.run --out results_followup_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m followup.guard.run --out results_followup_reproduced/count_guard
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m followup.audit --out results_followup_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m followup.report --out results_followup_reproduced

回归测试：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q

只核验已有数据：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m followup.audit

另有事后追加的 D 分支诊断：followup.guard.run。它有独立协议、独立种子与单独置信声明。
主网格结果先于这项诊断被观察；诊断不计入原先冻结的主试验。

完整接口：
from followup.core import test_fingerprint
from parameter_search.algorithm import parameters
from support_size import Fingerprint
plan = parameters('70', 14, 442414, '.1')
# fp 应来自预定 plan.m 下的 iid 泊松抽样；代码不能认证采样设计。
# result = test_fingerprint(fp, 442414, '.1', mean=plan.m)

验证文件包哈希：
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('results_followup_20261102')
for row in json.loads((root/'manifest.json').read_text())['files']:
    p=root/row['path']
    assert p.stat().st_size==row['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print('all artifact hashes pass')
PY

依赖版本在 protocol.json；原依赖锁 requirements-lock.txt 沿用。
历史整数细化依赖 results_parameter_search/raw 和 cells；其原哈希逐格核验。
协议冻结前用 smoke 做实现调试；正式数据使用固定新种子，不按结果调参。
定理、Monte Carlo 风险、数值认证及计算时间是不同证据层。
''')
    for name in ['audit.py','report.py']:
        target=out/'analysis_source'/name;target.parent.mkdir(exist_ok=True)
        target.write_bytes((ROOT/'followup'/name).read_bytes())
    files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
    save(out/'manifest.json',dict(files=files,total_bytes=sum(r['bytes'] for r in files)))
    print('REPORT',out/'report.html',flush=True)


if __name__=='__main__':main()
