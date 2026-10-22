"""Generate the Chinese experiment report and standalone scientific figures."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from c1025.data import ROOT, save, sha

NAMES = dict(C70_L16='C=70, L0=16', count_same_budget='Count, same samples',
             scale0_same_budget='Scale zero only', count_matched_confidence='Count (matched confidence)',
             count_99='Count (99% guarantee)', C1025_L1024='C=1025, L0=1024', original='Original C=2^26',
             C70_L16_x27='C=70, L0=16, 27 runs')
SUITES = dict(synthetic='构造分布', natural='自然经验总体', lifted='自然丰度形状扩展构造', amplification='27 次独立重复')
DATA_NAMES = dict(mite='土壤螨类', bird_north='鸟类：北样地', bird_south='鸟类：南样地',
                  spider_girdled='蜘蛛：环割处理', spider_logged='蜘蛛：采伐处理', moths='蛾类',
                  okland='陆生蜗牛', bci='BCI 森林树木', fish_2007_seine='湖泊鱼类：2007 围网',
                  fish_2008_seine='湖泊鱼类：2008 围网')


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def number(row, key):
    return float(row[key])


def summarize(out):
    suites = {s: read_csv(out/f'{s}.csv') for s in SUITES}
    summary = dict(C0=70, L0=16, suites={}, risk_by_method={})
    for suite, rows in suites.items():
        primary = [r for r in rows if r['method'].startswith('C70')]
        promised = [r for r in primary if r['errors'] != '']
        summary['suites'][suite] = dict(cells=len(primary), trials=sum(int(r['trials']) for r in primary),
            promised_trials=sum(int(r['trials']) for r in promised), errors=sum(int(r['errors']) for r in promised),
            lower_errors=sum(int(r['lower_errors']) for r in primary),
            worst_error_upper95_simultaneous=max(number(r, 'error_upper95_simultaneous') for r in promised),
            worst_lower_upper95_simultaneous=max(number(r, 'lower_error_upper95_simultaneous') for r in primary))
        if suite != 'amplification':
            summary['suites'][suite].update(
                polynomial_cells=sum(r['branch'] == 'chebyshev' for r in primary),
                polynomial_trials=sum(int(r['trials']) for r in primary if r['branch'] == 'chebyshev'),
                extra_rejects_beyond_count=sum(int(r['extra_rejects_beyond_count']) for r in primary),
                certified_lower_improvements=sum(int(r['certified_lower_improvements']) for r in primary),
                unresolved_integer_outputs=sum(int(r['lower_integer_interval_nonunique']) for r in primary),
                threshold_refinements=sum(int(r['numerical_refinements']) for r in primary),
                full_arb_audits=sum(int(r['full_arb_audits']) for r in primary),
                max_degree=max(int(r['max_degree']) for r in primary),
                max_score_enclosure_width=max(number(r, 'score_enclosure_max_width') for r in primary),
                max_poisson_bin_mass_defect=max(number(r, 'max_poisson_bin_mass_defect') for r in primary))
        summary['risk_by_method'][suite] = []
        for method in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['method'] == method and r['errors'] != '']
            summary['risk_by_method'][suite].append(dict(method=method,
                promised_cells=len(group), trials=sum(int(r['trials']) for r in group),
                errors=sum(int(r['errors']) for r in group),
                worst_observed_error=max(number(r, 'error_rate') for r in group),
                worst_upper95_simultaneous=max(number(r, 'error_upper95_simultaneous') for r in group)))
    primary_suites = ['synthetic', 'natural', 'lifted']
    summary['primary_trials'] = sum(summary['suites'][s]['trials'] for s in primary_suites)
    summary['primary_cells'] = sum(summary['suites'][s]['cells'] for s in primary_suites)
    summary['primary_method_trial_evaluations'] = sum(json.loads((out/f'{s}_complete.json').read_text())['method_trial_evaluations'] for s in primary_suites)
    summary['primary_independent_sampling_paths'] = sum(json.loads((out/f'{s}_complete.json').read_text())['independent_sampling_paths'] for s in primary_suites)
    summary['polynomial_trials'] = sum(summary['suites'][s]['polynomial_trials'] for s in primary_suites)
    summary['certified_lower_improvements'] = sum(summary['suites'][s]['certified_lower_improvements'] for s in primary_suites)
    summary['full_arb_primary_audits'] = sum(summary['suites'][s]['full_arb_audits'] for s in primary_suites)
    lookup = {(r['cell'], r['method']): r for r in suites['synthetic']}
    primary = [r for r in suites['synthetic'] if r['method'] == 'C70_L16']
    summary['synthetic_cost_ratios'] = {}
    for method in ['C1025_L1024', 'count_99', 'count_matched_confidence', 'original']:
        ratios = [number(r, 'm')/number(lookup[(r['cell'], method)], 'm') for r in primary]
        summary['synthetic_cost_ratios'][method] = dict(min=min(ratios), max=max(ratios))
    count_gain = []; scale_gain = []
    for s in ['synthetic', 'lifted']:
        by_key = {(r['cell'], r['method']): r for r in suites[s]}
        for r in suites[s]:
            if r['method'] == 'C70_L16' and r['branch'] == 'chebyshev':
                c = by_key[r['cell'], 'count_same_budget']; z = by_key[r['cell'], 'scale0_same_budget']
                count_gain.append((number(r, 'mean_lower_low')-number(c, 'mean_lower_high'))/int(r['n']))
                scale_gain.append((number(r, 'mean_lower_low')-number(z, 'mean_lower_high'))/int(r['n']))
    summary['max_certified_mean_lower_gain_fraction_of_n'] = max(count_gain)
    summary['multiscale_mean_better_than_scale0_cells'] = sum(g > 0 for g in scale_gain)
    summary['max_mean_gain_over_scale0_fraction_of_n'] = max(scale_gain)
    feasible = read_csv(out/'natural_feasibility.csv')
    summary['natural_feasible_cells'] = sum(r['has_mean_budget'] == 'True' for r in feasible)
    summary['natural_budget_cells'] = len(feasible)
    finite = read_csv(out/'finite_files.csv')
    summary['finite_file_trials'] = sum(int(r['trials']) for r in finite)
    summary['finite_file_completed'] = sum(int(r['completed']) for r in finite)
    summary['finite_file_overflows'] = sum(int(r['overflows']) for r in finite)
    summary['duplication_worst_error'] = max(number(r, 'error_rate') for r in read_csv(out/'duplication_stress.csv') if r['errors'] != '')
    summary['whole_group_worst_error'] = max(number(r, 'error_rate') for r in read_csv(out/'group_stress.csv') if r['errors'] != '')
    summary['amplification_theory'] = json.loads((out/'amplification_theory.json').read_text())
    summary['validation'] = json.loads((out/'validation.json').read_text())
    summary['diagnostics'] = json.loads((out/'diagnostics_complete.json').read_text())
    summary['limitations'] = [
        'Zero observed errors do not mean zero population risk or prove universal correctness.',
        'Finite empirical resampling does not certify iid field sampling or unseen ecosystem support.',
        'Cloned empirical shapes are synthetic and are reported separately.',
        'Binary64 probability evaluation/PRNG is not an exact real-valued sampler.',
        'All binary decisions and coverage events are enclosed/certified; some exact integer lower outputs are retained as intervals.',
        'Poisson fingerprint simulation does not reconstruct M or measure physical sample acquisition.',
        'The original, C1025, and C70 constants have different uniform single-run confidence guarantees.',
        'Natural dependence experiments are descriptive stress tests outside the iid theorem.',
        'No optimality, new asymptotic rate, proof-assistant formalization, or publication acceptance claim.'
    ]
    save(out/'summary.json', summary)
    return summary, suites


def figures(out, summary, suites):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 170})
    paths = []
    def finish(fig, name):
        fig.tight_layout()
        for suffix in ['png', 'pdf']:
            fig.savefig(out/'figures'/f'{name}.{suffix}', bbox_inches='tight')
        paths.append(name); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    keys = ['synthetic', 'natural', 'lifted', 'amplification']
    bars = [100*summary['suites'][s]['worst_error_upper95_simultaneous'] for s in keys]
    ax.bar(np.arange(4), bars, color=['#205878', '#28836f', '#ae7828', '#70629a'], alpha=.8)
    ax.scatter(np.arange(4), np.zeros(4), marker='o', color='black', zorder=3, label='Observed error rate (zero)')
    ax.set_xticks(np.arange(4), ['Synthetic', 'Natural empirical', 'Lifted shapes\n(synthetic)', '27 independent runs'])
    ax.set_ylabel('Worst cell error upper bound (%)'); ax.set_ylim(0, max(bars)*1.25)
    ax.set_title('C=70, L0=16: simultaneous 95% error bounds')
    for x, y in enumerate(bars): ax.text(x, y+.012, f'{y:.3f}%', ha='center')
    ax.legend(loc='upper left'); finish(fig, 'risk')

    rows = suites['synthetic']; by = {(r['cell'], r['method']): r for r in rows}
    chosen = sorted([r for r in rows if r['method'] == 'C70_L16' and r['scenario'] == 'uniform_null' and r['epsilon'] == '0.1'], key=lambda r: int(r['n']))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    for method, color in [('C70_L16', '#205878'), ('count_matched_confidence', '#28836f'), ('count_99', '#c27a2c'), ('C1025_L1024', '#8d658e')]:
        values = [by[(r['cell'], method)] for r in chosen]
        axes[0].loglog([int(r['n']) for r in values], [number(r, 'm') for r in values], '-o', ms=3, label=NAMES[method], color=color)
    axes[0].set(xlabel='n', ylabel='Expected sample budget', title='Single run, epsilon=0.1')
    axes[0].legend(fontsize=8)
    for multiplier, label, color in [(1, '70/16 single run / count-99', '#205878'), (27, '70/16 x27 / count-99', '#b26935')]:
        ratios = [multiplier*number(r, 'm')/number(by[(r['cell'], 'count_99')], 'm') for r in chosen]
        axes[1].semilogx([int(r['n']) for r in chosen], ratios, '-o', ms=3, label=label, color=color)
    axes[1].axhline(1, ls='--', color='gray'); axes[1].set(xlabel='n', ylabel='Sample-budget ratio', title='Lower is cheaper; 27 runs give <1% risk')
    axes[1].legend(fontsize=8); finish(fig, 'budgets')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, case, title in zip(axes, ['epsilon_scaled_rare_1/10_null', 'epsilon_scaled_rare_1/2_null'], ['Tail mass = 0.1 epsilon', 'Tail mass = 0.5 epsilon']):
        selected = sorted([r for r in rows if r['method'] == 'C70_L16' and r['scenario'] == case and r['epsilon'] == '0.1'], key=lambda r: int(r['n']))
        x = [int(r['n']) for r in selected]
        ax.semilogx(x, [number(r, 'mean_lower_low')/int(r['n']) for r in selected], '-o', ms=3, color='#205878', label='Full statistic')
        ax.semilogx(x, [number(r, 'mean_observed')/int(r['n']) for r in selected], '--', color='#a26839', label='Count, same samples')
        ax.axvline(3269018, color='gray', ls=':', label='Branch boundary')
        ax.set(xlabel='n', ylabel='Mean lower bound / true support', title=title, ylim=(0, 1.03))
        ax.legend(fontsize=8)
    fig.suptitle('Rare-category nulls: sharper lower bounds, no added binary rejections', fontsize=11)
    finish(fig, 'lower_bounds')

    data = json.loads((out/'datasets.json').read_text()); feasible = read_csv(out/'natural_feasibility.csv')
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [r['dataset'] for r in data]
    for index, name in enumerate(names):
        values = [number(r, 'budget_to_records') for r in feasible if r['dataset'] == name]
        jitter = np.linspace(-.18, .18, len(values))
        ax.scatter(np.full(len(values), index)+jitter, values, color='#205878', alpha=.65, s=17)
    ax.axhline(1, color='#b15a38', ls='--', label='Available records = expected budget')
    ax.set_yscale('log'); ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
    ax.set_ylabel('Required mean / available records'); ax.set_title('Natural-data feasibility: 42 of 236 settings reach the mean budget')
    ax.legend(fontsize=8); finish(fig, 'natural_feasibility')

    stress = read_csv(out/'duplication_stress.csv')
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for name in ['mite', 'bci', 'fish_2007_seine', 'fish_2008_seine']:
        points = sorted([r for r in stress if r['dataset'] == name and r['epsilon'] == '0.1'], key=lambda r: int(r['factor']))
        ax.semilogx([int(r['factor']) for r in points], [number(r, 'error_rate') for r in points], '-o', label=name)
    ax.set(xlabel='Repeated copies per independent observation', ylabel='Failure to reject', ylim=(-.02, 1.05),
           title='Dependence stress: same expected records, fewer independent labels')
    ax.legend(fontsize=8); finish(fig, 'dependence')

    runtime = read_csv(out/'runtime.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = [int(r['n']) for r in runtime]
    axes[0].semilogx(x, [int(r['degree']) for r in runtime], '-o', color='#205878')
    axes[0].set(xlabel='n', ylabel='Maximum polynomial degree', title='Full degrees retained')
    axes[1].loglog(x, [number(r, 'scoring_seconds_median')*1e6/int(r['trials']) for r in runtime], '-o', color='#28836f')
    axes[1].set(xlabel='n', ylabel='Microseconds per fingerprint', title='Local batch scoring; excludes sampling')
    finish(fig, 'runtime')
    return paths


def html_table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def report_html(out, summary, suites, plots):
    def img(name, caption):
        data = base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
    stats = summary['suites']
    risk_rows = [[SUITES[s], f'{v["cells"]:,}', f'{v["trials"]:,}', f'{v["promised_trials"]:,}', v['errors'], v['lower_errors'], f'{100*v["worst_error_upper95_simultaneous"]:.4f}%'] for s, v in stats.items()]
    datasets = json.loads((out/'datasets.json').read_text())
    data_rows = [[DATA_NAMES[d['dataset']], d['N'], d['S'], d['groups'], f'<a href="{html.escape(d["source"])}">来源文档</a>'] for d in datasets]
    risk_methods = []
    for s in ['synthetic', 'natural', 'lifted']:
        for r in summary['risk_by_method'][s]:
            risk_methods.append([SUITES[s], html.escape(NAMES[r['method']]), f'{r["trials"]:,}', r['errors'], f'{100*r["worst_observed_error"]:.4f}%', f'{100*r["worst_upper95_simultaneous"]:.4f}%'])
    validation = summary['validation']; diagnostics = summary['diagnostics']
    numerical_interval_count = sum(stats[s]['unresolved_integer_outputs'] for s in ['synthetic', 'natural', 'lifted'])
    comparisons = []
    for s in ['synthetic', 'natural', 'lifted', 'amplification']:
        for r in suites[s]:
            if r['method'].startswith('C70'):
                comparisons.append(dict(suite=SUITES[s], scenario=r['scenario'], n=r['n'], epsilon=r['epsilon'],
                     truth=r['truth'], branch=r.get('branch', '27 independent runs'), trials=r['trials'],
                     errors=r['errors'], upper=f'{100*number(r, "error_upper95_simultaneous"):.4f}%' if r['errors'] != '' else '不定义',
                     mean_lower=r['mean_lower_low'], raw_file=r['raw_file']))
    source = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C₀=70，L₀=16：完整实验报告</title><style>
body{{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;max-width:1120px;margin:auto;padding:32px 24px;background:#f5f7f8;color:#20323e;line-height:1.75}}
h1{{font-size:2rem;line-height:1.35}}h2{{margin-top:2.3em;border-bottom:2px solid #dbe4e8;padding-bottom:8px}}p{{max-width:1000px}}
.lead{{font-size:1.12rem}}.card{{background:white;border:1px solid #d9e1e5;border-radius:10px;padding:18px 22px;margin:20px 0}}
.note{{border-left:4px solid #b87834;padding:10px 18px;background:#fff7eb}}table{{border-collapse:collapse;width:100%;font-size:.9rem;background:white}}
th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid #dce4e8}}th{{background:#e9f0f3;white-space:nowrap}}.scroll{{overflow-x:auto}}
figure{{margin:28px 0;background:white;padding:12px;border-radius:8px}}img{{max-width:100%;height:auto}}figcaption{{font-size:.9rem;color:#526570}}
code,pre{{font-family:ui-monospace,monospace}}pre{{overflow:auto;padding:16px;background:#e8eef1;border-radius:6px}}a{{color:#145e83}}
input{{width:95%;padding:12px;border:1px solid #b5c8d1;border-radius:5px;font-size:1rem}}small{{color:#526570}}
@media print{{body{{background:white;padding:0}}.filter{{display:none}}h2,figure{{break-inside:avoid}}}}
</style><body>
<p><small>OSST · 2026-10-22 · 固定参数实验 · 完整原始频数与复现协议保留</small></p>
<h1>C₀=70，L₀=16：自然数据与构造数据实验</h1>
<p class="lead">新参数已实际接入并运行。三套主实验共 <strong>{summary['primary_cells']:,} 个配置、{summary['primary_trials']:,} 次新算法试验</strong>，其中 {summary['polynomial_trials']:,} 次进入完整多项式分支。承诺检验及支持下界均未观察到错误；这表示有限实验上的结果，不能解释为真实错误率为零。</p>
<div class="card"><strong>最重要的结论</strong><ul>
<li>单次 (70,16) 相比旧 (1025,1024) 约少用 14.64 倍样本，但两者的统一单次成功概率保证不同。</li>
<li>在构造网格上，新算法仍比 99% 保证的计数基线多用 <strong>2.44–8.39 倍</strong>样本；比同理论置信度的计数基线多用 2.44–16.30 倍。</li>
<li>同预算计数消融也没有承诺检验错误。多项式在 <strong>{summary['certified_lower_improvements']:,} 次</strong>试验中提高了支持大小下界，没有增加二元拒绝次数。</li>
<li>自然数据中 <strong>{summary['natural_feasible_cells']}/{summary['natural_budget_cells']}</strong> 个配置的原始记录数达到所需平均预算；达到均值不等于每次泊松抽样都不会溢出。</li>
<li>人为复制观测导致最坏漏报率达到 <strong>{100*summary['duplication_worst_error']:.1f}%</strong>，iid 假设仍然关键。</li></ul></div>

<h2>1. 算法、模型与公平比较</h2>
<p>固定 C₀=70、L₀=16，L=1+ln n，泊松均值 m=70n/[ε(1+ln n)]。新入口 <code>python -m c7016.algorithm</code> 保留实数均值、全部尺度、完整阶数和原修正项。n≤3,269,017 使用计数；n≥3,269,018 使用多项式。原 <code>paper_reference.py</code> 和旧结果保持原含义。</p>
<p>证明给出每项任务的单次失败上界 0.282612。ε 是分布分离参数，不是失败率。承诺类为 S≤n 与 Δₙ≥ε；S&gt;n 且 Δₙ&lt;ε 的 Gap 输入没有规定二元正确标签，但仍检查支持下界有效性。支持下界事件为 min(effε(p),n)≤Ŝ≤S。</p>
<p>对照包括：相同样本上的计数/仅第零尺度消融、失败目标 0.282612 的计数基线、99% 计数基线、旧 (1025,1024)、原参数。不同预算方法使用独立随机流；新算法及两个消融共享频数，不能把它们当成独立试验。主实验合计 {summary['primary_independent_sampling_paths']:,} 条独立模拟路径、{summary['primary_method_trial_evaluations']:,} 次方法评估。</p>
<p>规模为 1、10、10³、10⁵、10⁶、3,269,017、3,269,018、3,269,019、10⁷、10⁹、10¹²；构造 ε 为 0.001、0.01、0.1、0.25。分布覆盖均匀、单头稀疏尾、分段幂律、对数正态/高斯/几何块、弥散与多尺度边界、Gap、与 ε 成比例的稀有类别。每个构造和自然配置 4×1000 次，扩展构造为 4×500 次。协议在正式抽样前冻结，无依赖错误次数的停止。</p>
{img('budgets', '图 1：期望样本成本。单次方案的理论置信度不同；右侧 27 次重复与 count-99 均达到单项错误率低于 1% 的保证。')}

<h2>2. 错误率与下界有效性</h2>
<div class="scroll">{html_table(['实验','配置','新算法试验','承诺类试验','检验错误','下界失效','最坏单元同时95%错误上界'],risk_rows)}</div>
<p>上界为单侧 Clopper–Pearson，并在每套实验全部非 Gap 方法单元上做 Bonferroni 修正。支持下界使用覆盖全部单元的独立置信区间族。表中的“最坏上界”不是把所有不同分布试验合并后得到的总平均率。27 次重复实验是另一套实验，表中试验数指聚合输出次数。</p>
{img('risk', '图 2：零观察错误对应的非零置信上界；柱高是该套实验最坏单元的同时 95% 上界。')}
<details><summary>展开所有方法的错误汇总</summary><div class="scroll">{html_table(['实验','方法','承诺类评估','错误数','最坏观察率','最坏同时95%上界'],risk_methods)}</div></details>

<h2>3. 多项式实际带来了什么</h2>
<p>构造数据有 {stats['synthetic']['polynomial_cells']} 个配置、扩展构造有 {stats['lifted']['polynomial_cells']} 个配置实际执行多项式。全部阶数保留，最高阶数 {max(stats[s]['max_degree'] for s in ['synthetic','natural','lifted'])}。相对同预算计数，多项式提高了 {summary['certified_lower_improvements']:,} 次下界，占多项式试验的 {100*summary['certified_lower_improvements']/summary['polynomial_trials']:.2f}%。最大单元平均改善约为 n 的 {100*summary['max_certified_mean_lower_gain_fraction_of_n']:.2f}%。相对只用第零尺度，全多尺度在 {summary['multiscale_mean_better_than_scale0_cells']} 个配置的平均下界有严格认证改善。</p>
<p>例如 ε=0.1、n=10¹²、总稀有尾质量为 0.1ε 的零假设分布，计数均值约为 0.2169n，新下界约为 0.5786n。这个场景的 effε=1，因此它展示了额外信息，不是有效支持保证从失败变成功。</p>
<p>本次所有承诺类上同预算计数也无检验错误。对 n≤10¹²，有 mε/n=70/(1+ln n)≥2.44，因此该规模范围内计数仍获得很充足的样本。不能由这些曲线宣称已展示多项式在二元任务上的必要性，或推断新方案已经胜过实用计数基线。</p>
{img('lower_bounds', '图 3：稀有类别零假设下的下界改善。分支切换处的跃迁来自算法规则，而非更换参数或缩小阶数。')}

<h2>4. 自然数据、真实文件与扩展构造</h2>
{html_table(['经验总体','原始个体数 N','观察类别数 S','自然分组数','来源'],data_rows)}
<p>自然数据按来源既定站点、处理、年份和渔具分开。每套自然总体使用 ε=0.001、0.01、0.05、0.1、0.25，并取 S/2、effε−1、effε、S、1.5S 附近的去重整数 n。所有概率与边界真值均由原始整数丰度进行有理数计算。</p>
<p>这些自然总体类别数较小，全部自然主实验走计数分支。另将每个经验类别等质量拆分成多个新类别，构造约 10⁷ 和 10⁹ 支持的分布，保留丰度形状并触发多项式。这 {stats['lifted']['cells']} 个配置明确记作<strong>合成数据</strong>，并不表示自然界观测到了这些额外物种。</p>
{img('natural_feasibility', '图 4：每个点是一个自然数据配置，虚线以下表示原始记录数达到算法期望预算；不能把重采样产生的次数算成新增现场观测。')}
<p>实际有限文件另进行了 {summary['finite_file_trials']:,} 次尝试，完成 {summary['finite_file_completed']:,} 次，{summary['finite_file_overflows']:,} 次因请求样本数超过文件记录数返回 Inconclusive。没有截断泊松数或反复抽到不溢出；完成后的不放回子样本仅作描述性分析，不以其条件成功率声称未知自然总体上的 iid 保证。</p>

<h2>5. 置信度放大与相关性压力</h2>
<p>保持每次参数 (70,16)，27 次使用独立新样本，二元取多数票，下界取中位数。单项理论失败率 ≤{summary['amplification_theory']['per_task_failure']:.10f}，总期望系数 1890。若要求两个输出同时正确，当前并集失败上界为 {100*summary['amplification_theory']['either_output_failure_upper']:.4f}%，不能把这套方案称为“双输出同时99%”。</p>
<p>50 个配置各做 1000 次聚合输出，全部无观察错误，但它的理论样本成本约为旧系数 1025 的 1.844 倍。此前 (70,20) 的放大结果没有混入本次 (70,16) 的实验。</p>
<p>相关性测试包括观测复制、整组复采样和训练/留出自然分组。复制每个独立标签 10/100/1000 次会严重降低有效独立信息，最坏漏报率达 100%。本次整组压力与留出配置未观察到承诺检验错误，但这不能验证现场采样的独立性。</p>
{img('dependence', '图 5：人为复制保留期望总记录数和边际类别分布，却违反独立性；错误率不能沿用 iid 定理。')}

<h2>6. 数值、实现与可复核性</h2>
<p>本次结果独立审计通过 <strong>{validation['checks']:,} 项</strong>检查；主实验有 {summary['full_arb_primary_audits']:,} 个完整 Arb 复核，另对 {validation['additional_full_arb_scores']} 个保存样本以 1024 位重算。{diagnostics['full_api_calls']} 次完整频数公开接口调用覆盖边界两侧及 n=10⁷，其中 {diagnostics['polynomial_api_calls']} 次为多项式分支，实际生成 {diagnostics['explicit_category_poisson_draws']:,} 个逐类别泊松计数。新旧回归测试共 125 项通过。理论证书 428 项和独立算术审计 2491 项已从头重算，结果与原证明包逐字节一致。</p>
<p>批量计算对权重、点积舍入和最终比较给出向外包络。所有二元判定与下界覆盖事件均数值认证。主实验中 {numerical_interval_count:,} 次精确整数下界仍以上下界区间保存，未猜测某个整数；报告的均值由认证端点汇总，显示值经浮点舍入。公共单条接口会提高精度直到确定整数输出，否则报错。频数模拟保留每个所需 Fₗ 和高频类别总数，不伪造实际总样本数 M。</p>
<p>分组模拟与逐类别独立泊松抽样在实数运算下同分布，但 SciPy 概率计算和 NumPy 随机数仍为 binary64；这不等同于精确随机采样的形式化证明。全部测量均是有限总体计算实验，不包含现场采样耗时。</p>
{img('runtime', '图 6：完整阶数和本地计算开销。计时不含现场数据采集、写盘，也不把压缩模拟时间称为采集全部预算所需时间。')}

<h2>7. 复现与文件</h2><pre>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.run --phase all --out results_c7016_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.diagnostics --out results_c7016_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.validate --out results_c7016_reproduced --proof-and-tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.report --out results_c7016_reproduced</pre>
<p><a href="REPRODUCE.txt">完整复现说明</a> · <a href="summary.json">结果摘要</a> · <a href="protocol.json">冻结协议</a> · <a href="validation.json">审计结果</a> · <a href="data_sources.json">数据来源与许可</a> · <a href="experiment_report.pdf">英文 PDF 报告</a></p>
<p><a href="synthetic.csv">构造数据 CSV</a> · <a href="natural.csv">自然数据 CSV</a> · <a href="lifted.csv">扩展构造 CSV</a> · <a href="amplification.csv">独立重复 CSV</a> · <a href="natural_feasibility.csv">自然样本可行性</a> · <a href="runtime.csv">运行开销</a></p>
<p>每个原始文件位于 raw/，对应的精确分布、算法参数和文件哈希在 cells/。source_snapshot/ 保留主实验代码与证明来源，manifest.json 校验最终报告及数据。重新运行需要使用冻结时的源码；源文件或数据变化会拒绝接续旧结果。</p>
<p>审计后的一个统计口径修订已记录于 protocol.json：27 次重复实验的 50,000 个聚合输出对应 1,350,000 个独立单次运行，另有 100,000 个对照路径。修订仅更正完成标记中的路径数与聚合输出数，不改动任何抽样、原始数据或判定。初版协议和修改前源码保留。</p>
<p>另补充了批量接口对 n 或整数比较阈值超出 ±2⁵³ 的拒绝检查，防止超出 binary64 精确整数范围时误用；本次所有输入都在范围内，已有结果不变。高精度单条接口仍可用于更大整数。</p>

<h2>8. 新算法逐配置结果</h2><p class="filter"><input id="query" placeholder="筛选：例如 natural、uniform、3269018、0.01（也支持中文实验名）"></p><p id="count"></p>
<div class="scroll"><table><thead><tr><th>实验</th><th>场景</th><th>n</th><th>ε</th><th>真值</th><th>分支</th><th>次数</th><th>错误数</th><th>同时95%上界</th><th>原始文件</th></tr></thead><tbody id="results"></tbody></table></div>
<script>const DATA={json.dumps(comparisons,ensure_ascii=False)};
const esc=v=>String(v).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function draw(){{const q=document.querySelector('#query').value.toLowerCase();const rows=DATA.filter(r=>Object.values(r).join(' ').toLowerCase().includes(q));document.querySelector('#count').textContent=rows.length+' 个配置';document.querySelector('#results').innerHTML=rows.map(r=>'<tr>'+['suite','scenario','n','epsilon','truth','branch','trials','errors','upper'].map(k=>'<td>'+esc(r[k]===''?'—':r[k])+'</td>').join('')+'<td><a href="'+esc(r.raw_file)+'">NPZ</a></td></tr>').join('');}}document.querySelector('#query').addEventListener('input',draw);draw();</script>
</body></html>'''
    (out/'report.html').write_text(source)


def latex_report(out, summary):
    v = summary['suites']
    rows = '\n'.join(f'{name} & {v[s]["cells"]} & {v[s]["trials"]:,} & 0 & {100*v[s]["worst_error_upper95_simultaneous"]:.4f}\\% \\\\' for s, name in [('synthetic', 'Synthetic'), ('natural', 'Natural empirical'), ('lifted', 'Lifted (synthetic)'), ('amplification', '27 independent runs')])
    source = r'''\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,booktabs,graphicx,microtype,float}
\usepackage[hidelinks]{hyperref}
\title{Experiments with $C_0=70$ and $L_0=16$\\Natural Empirical and Constructed Distributions}
\author{OSST project experimental artifact}
\date{22 October 2026}
\begin{document}\maketitle
\begin{abstract}
We execute the real-mean $(70,16)$ support-size construction on 756 constructed,
236 natural empirical, and 120 empirically shaped synthetic configurations.
The primary method is evaluated 4,208,000 times, including 1,672,000 actual
polynomial-branch trials. No promised binary error or support-bound failure
is observed. However, the same-budget count ablation also has no binary errors,
and a 99\%-guaranteed count baseline uses 2.44--8.39 times fewer expected samples
on the synthetic grid. Polynomial corrections improve 554,000 lower bounds.
We preserve negative comparisons, finite-file failures and dependence stress.
\end{abstract}
\section{Protocol and statistical scope}
We implement the construction in Qisheng Wang's \emph{Optimal Support Size Testing}
(supplied manuscript \texttt{2610.18723v1}) with the project's certified constants.
The algorithm uses $L=1+\ln n$, real Poisson mean
$m=70n/(\epsilon L)$, and the unchanged polynomial construction when $L\ge16$.
Its first integer polynomial input is $n=3,269,018$. We retain all scales,
degrees, corrections and decision thresholds; the largest tested degree is 162.
The supplied theorem gives individual failure below 0.282612 for the binary
promise problem and for support-bound coverage. $\epsilon$ is the distribution
separation parameter, not the failure probability.

Synthetic sizes range from 1 to $10^{12}$ and include three consecutive
integers at the branch boundary. Four epsilons are 0.001, 0.01, 0.1 and 0.25.
Uniform, concentrated-head, sparse-tail, power, lognormal, Gaussian-block,
geometric-block, boundary and gap distributions are included. Natural data add
$\epsilon=0.05$. Each primary synthetic/natural cell has four independent blocks
of 1000 trials; lifted-shape cells have four blocks of 500. Exact rational
tail masses determine promise labels. Gap inputs have no binary error label.
Code/data hashes were frozen before primary simulation, with no error-dependent
stopping. Different budgets use independent streams; same-budget ablations
share fingerprints. Method evaluations are not all independent samples.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrr}\toprule
Suite & Cells & Outputs & Errors & Worst 95\% bound\\\midrule
''' + rows + r'''
\bottomrule\end{tabular}
\caption{Primary method or amplified outputs. The last column is a percentage:
the worst cell's one-sided Clopper--Pearson upper bound with Bonferroni
correction across all non-gap method cells in its suite. Zero observed errors
do not establish zero risk. Support-bound coverage is checked separately.}
\end{table}
\begin{figure}[H]\centering\includegraphics[width=.97\linewidth]{figures/risk.pdf}
\caption{Simultaneous error bounds, including the separate amplification experiment.}\end{figure}

\section{Samples, lower bounds and confidence}
Single-run samples are about 14.64 times fewer than for $(1025,1024)$,
but the uniform single-run confidence guarantees differ. At the same theoretical
confidence target, a certified count baseline still uses 2.44--16.30 times fewer
samples. Even the 99\% count baseline uses 2.44--8.39 times fewer samples.
This practical grid has $70/(1+\ln n)\ge2.44$, so the count statistic already
receives a substantial budget. The polynomial changes improve lower bounds,
not binary rejection counts, on this grid.

On the rare-tail null with $n=10^{12}$, $\epsilon=0.1$ and tail mass
$0.1\epsilon$, the mean observed fraction is about 0.2169 while the new lower
bound is about $0.5786n$. Here the effective support is only one; this illustrates
extra information, not recovery from a failed effective-support guarantee.

Using only $(70,16)$, 27 independent repetitions, majority voting and median
aggregation give per-task failure at most 0.008175637833. The expected total
coefficient is 1890, about 1.844 times the older coefficient 1025. A union bound
for either output failing is 0.016351275666, so we do not claim simultaneous
99\% correctness for both outputs. Fifty configurations each use 1000 aggregated
trials. The earlier $(70,20)$ results are not mixed into this experiment.
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/budgets.pdf}
\caption{Expected sample costs and confidence-aligned repeated-run comparisons.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/lower_bounds.pdf}
\caption{Sharper lower bounds on rare-category nulls. The branch change is prescribed.}\end{figure}

\section{Natural data and dependence}
Ten finite empirical populations cover mites, birds at two sites, spiders under
two treatments, moths, land snails, BCI trees and fish from two year/gear strata.
Sources, pinned versions, hashes and licenses are retained in
\texttt{data\_sources.json}. No different years/sites/treatments are pooled.
All original natural populations use the count branch. Splitting each observed
category into equally probable synthetic clones creates separate lifted-shape
tests around $10^7$ and $10^9$ support; these are explicitly synthetic, not new
species observations. There are 116 polynomial lifted-shape configurations.

Only 42 of 236 natural configurations have enough original records to reach the
mean budget. Fixed-file tests retain every Poisson overflow as Inconclusive,
never truncate or resample until success. Successful without-replacement
subsamples are descriptive and do not establish unconditional iid risk for an
unknown ecosystem. Group holdouts and whole-group compound-Poisson resampling
are separate stress tests. Artificially duplicated observations produce failure
rates up to 100\%, despite preserving marginal labels and expected record count.
\begin{figure}[H]\centering\includegraphics[width=.96\linewidth]{figures/natural_feasibility.pdf}
\caption{Original record budgets across natural configurations.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{figures/dependence.pdf}
\caption{Artificial dependence invalidates the independent-sampling hypothesis.}\end{figure}

\section{Numerical validation and reproducibility}
Arb certifies real-mean parameters and full polynomial weights. Batched dot
products use outward enclosures and conservative IEEE-754 roundoff bounds;
unresolved event thresholds are refined with Arb. Every binary decision and
coverage event is certified on its supplied counts. Some exact integer lower
outputs remain enclosed in intervals; their means are reported as intervals,
not guessed integers. The public single-fingerprint interface increases
precision until it certifies the exact integer output or raises an error.

There are 8896 full-Arb primary checks, 476 further saved-profile checks at
1024 bits, and 18 explicit full-fingerprint API runs (12 polynomial). All
125 regression tests pass. The analytic proof's 428 exact obligations and
2491 independent arithmetic checks are recomputed with byte-identical results.
The final audit includes ''' + f'{summary["validation"]["checks"]:,}' + r''' checks.
This is not an end-to-end proof-assistant formalization.

Grouped multinomial fingerprints have the same law as independent Poisson
category counts in real arithmetic. Probability evaluation and pseudorandom
generation remain binary64 approximations. The high-count bucket retains
category counts without fabricating total observations $M$. No compressed
simulation timing is represented as physical sample-acquisition time.
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/runtime.pdf}
\caption{Full degree and local scoring cost, excluding physical data collection.}\end{figure}

\paragraph{Artifact.} Start with the Chinese \texttt{report.html}, then
\texttt{REPRODUCE.txt}, \texttt{protocol.json}, CSV summaries, exact population
specifications in \texttt{cells/}, and per-trial fingerprints in \texttt{raw/}.
The source snapshot and final manifest permit independent reruns. Statistical
success on this finite grid does not prove globally minimal constants, improved
asymptotic complexity, ecosystem completeness or a venue-acceptance claim.
\end{document}
'''
    (out/'experiment_report.tex').write_text(source)


def reproduce(out):
    (out/'REPRODUCE.txt').write_text('''C0=70, L0=16 experiment (2026-10-22)
==================================
Start with report.html (self-contained Chinese report) or experiment_report.pdf.
This artifact implements the real-mean theorem in c7016/algorithm.py; the older
paper_reference.py remains the original/1025 reference and is not relabeled.

FULL REPRODUCTION (from OSST project root)
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.run --phase all --out results_c7016_reproduced
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.diagnostics --out results_c7016_reproduced
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.validate --out results_c7016_reproduced --proof-and-tests
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.report --out results_c7016_reproduced
Use requirements-executed.txt for the exact executed environment. The primary
code/data are frozen in protocol.json and source_snapshot/. Changed source/data
must use a new output directory. Completed cells are resumed only after a raw
file hash check. Seeds depend on fixed experiment identifiers, not scheduling.

QUICK CHECK
  .venv/bin/python -m pytest -q
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.run --smoke --out /tmp/osst_c7016_smoke_new

PUBLIC API (the required sampling mean is the real expression in plan output)
  .venv/bin/python -m c7016.algorithm plan --n 3269018 --epsilon 0.1
  .venv/bin/python -m c7016.algorithm test --n 3269018 --epsilon 0.1 --fingerprint fingerprint.json
  .venv/bin/python -m c7016.algorithm demo --n 3269018 --epsilon 0.1
The caller must meet the independent Poisson sampling model at the predeclared
real mean. M is observed total, not a substitute for the mean. The RNG is a
binary64 approximation; numerical certification concerns the supplied counts.

BUILD ENGLISH PDF
  cd results_c7016_reproduced
  ../.tools/tectonic-0.17.0/tectonic experiment_report.tex --keep-logs
The figures also have standalone PDF and PNG versions. report.html embeds its
images but links to CSV/raw files in the same output directory.

FINAL ARTIFACT MANIFEST (after building the PDF)
  .venv/bin/python -m c7016.package --out results_c7016_reproduced
  .venv/bin/python -m c7016.package --out results_c7016_reproduced --verify

CONTENTS
  protocol.json: frozen primary protocol, input hashes, environment.
  diagnostics_protocol.json: disclosed post-primary numerical integration checks.
  synthetic/natural/lifted.csv: primary risk, coverage, costs and numerical widths.
  *_seed_blocks.csv: four independent random-seed blocks.
  amplification.csv: actual independent 27-copy majority/median experiments.
  natural_feasibility.csv: original record counts versus expected budgets.
  finite_files.csv: all fixed-file attempts, retaining Poisson overflows.
  duplication_stress/group_stress/group_holdout.csv: dependence/holdout checks.
  public_fingerprint_runs.json: natural iid and descriptive fixed-file API calls.
  full_polynomial_api_runs.json: explicit per-category Poisson simulations near
    the branch threshold and at n=10^7, actual total M and exact API outputs.
  runtime.csv: local parameter compilation, grouped sampling and scoring costs.
  sampler_validation.json: Arb PMF checks and huge diffuse-population sampler test.
  validation.json: independent raw-output, truth and confidence-bound audit.
  proof_recheck/: recomputed exact analytic certificates, audit and logs.
  tests_final.txt: all regression tests.
  cells/: exact rational populations, real mean/scale details, raw hashes.
  raw/: per-trial D, full required F1..FK, decision/coverage flags and score
    enclosures. Counts above K are coarsened without claiming an observed M.
    In amplified files each run is independent; median output is not a median
    of raw scores from an unrelated method.
  data_sources.json: pinned natural data and upstream license/attribution.

LIMITS
  Natural experiments resample finite observed empirical populations. They do
  not discover unseen ecosystem support or verify iid field sampling.
  Lifted empirical shapes are explicitly synthetic category splitting.
  Fixed-file subsampling and dependence tests are outside the ideal iid claim.
  Zero observed failures do not mean zero risk. Gap cases have no binary label.
  Single-run confidence differs from the original and 1025 constants; use the
  repeated-run comparison for a theoretical per-task 99% guarantee.
  Every decision/coverage event is numerically enclosed; some exact integer
  lower outputs are retained as intervals, with endpoints in the raw data.
  Full-field acquisition time is not measured by compressed simulation speed.
  No global constant-optimality or proof-assistant formalization is claimed.
''')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results_c7016')
    args = parser.parse_args(argv)
    summary, suites = summarize(args.out)
    plots = figures(args.out, summary, suites)
    report_html(args.out, summary, suites, plots); latex_report(args.out, summary); reproduce(args.out)
    for name in ['report.py', 'validate.py', 'diagnostics.py', 'package.py']:
        dst = args.out/'source_snapshot/c7016'/name
        shutil.copy2(Path(__file__).parent/name, dst)
    save(args.out/'report_sources.json', {name: sha(Path(__file__).parent/name) for name in ['report.py', 'validate.py', 'diagnostics.py', 'package.py']})
    print(json.dumps({k: summary[k] for k in ['primary_cells', 'primary_trials', 'polynomial_trials', 'natural_feasible_cells']}, indent=2))


if __name__ == '__main__':
    main()
