"""Generate concise Chinese HTML/TXT analysis and portable plots from saved data."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results_real_l01_20261104'
NAMES = {'midfish':'MidFish 湖泊鱼类', 'ltrm':'LTRM 河流鱼类',
         'nrmn':'NRMN 礁区鱼类', 'epa_nrsa':'EPA 河流鱼类', 'bbs':'BBS 鸟类'}
METHODS = {'full':'完整多尺度 C70L1', 'single':'第0尺度 + D',
           'count':'相同预算 D（等同 C70L14）', 'no_count':'去掉 D 的消融',
           'count_budget':'独立较小预算 D（风险≤1/3）'}


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    results = json.loads((OUT/'results.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    pops = json.loads((OUT/'prepared/populations.json').read_text())
    audit_path = OUT/'results_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    extra_path = OUT/'budget_01/summary.json'
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else None
    extra_results = json.loads((OUT/'budget_01/results.json').read_text()) if extra else []
    gain_audit = audit.get('gain_interpretation', {}).get('counts', {})
    proof_path = ROOT/'results_proof_l01/summary.json'
    full_proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
    proof_complete = full_proof is not None and full_proof.get('status') == 'PASS'
    if proof_complete:
        for suffix in ['html','txt']:
            current=OUT/f'report.{suffix}'
            saved=OUT/f'report_before_full_proof.{suffix}'
            if current.exists() and not saved.exists():
                saved.write_bytes(current.read_bytes())
    grouped = defaultdict(list)
    for row in results:
        grouped[row['dataset']].append(row)
    data_rows, accuracy_rows = [], []
    for dataset, rows in grouped.items():
        ps = [p for p in pops if p['dataset']==dataset]
        Ns = [sum(p['counts']) for p in ps]
        Ss = [len(p['counts']) for p in ps]
        trials = sum(r['trials'] for r in rows)
        gain = sum(r['full_better_than_count'] for r in rows)
        worst = max(r['stats']['full']['joint']/r['trials'] for r in rows)
        data_rows.append([NAMES.get(dataset,dataset),len(ps),f'{min(Ns):,}–{max(Ns):,}',
                          f'{min(Ss)}–{max(Ss)}',len(rows),f'{trials:,}'])
        accuracy_rows.append([NAMES.get(dataset,dataset),
                              sum(r['stats']['full']['joint'] for r in rows),
                              f'{worst:.2%}',f'{gain/trials:.2%}',
                              f'{np.mean([r["mean_gain_over_count"] for r in rows]):.3f}',
                              f'{np.median([r["m_ratio_to_count_budget"] for r in rows]):.2f}×'])
    method_rows = []
    for mode, label in METHODS.items():
        method_rows.append([label]+[sum(r['stats'][mode][key] for r in results)
                                   for key in ['binary','over','short','joint']]+[
                                       f'{max(r["stats"][mode]["joint_upper_simultaneous_95"] for r in results):.2%}'])
    if extra:
        method_rows.append(['独立较小预算 D（风险≤1%，补充）']+[
            sum(r['stats'][key] for r in extra_results) for key in ['binary','over','short','joint']]+[
                f'{extra["maximum_cell_upper_simultaneous_95"]:.2%}'])
    worst_gain = sorted(results, key=lambda r:r['mean_gain_over_count'], reverse=True)[:5]
    gain_rows = [[r['id'], NAMES.get(r['dataset'],r['dataset']),r['S'],r['n'],r['epsilon'],
                 f'{r["mean_gain_over_count"]:.3f}',f'{r["full_better_than_count"]/r["trials"]:.1%}']
                for r in worst_gain]
    flat = []
    for r in results:
        for mode in METHODS:
            flat.append(dict(cell=r['id'],dataset=r['dataset'],population=r['population_id'],
                             method=mode,n=r['n'],epsilon=r['epsilon'],S=r['S'],eff=r['eff'],
                             truth=r['truth'],trials=r['trials'],m=r['m'] if mode!='count_budget' else r['baseline_m'],
                             **{k:v for k,v in r['stats'][mode].items() if isinstance(v,(float,int))}))
    fields = sorted(set().union(*(x.keys() for x in flat)))
    with (OUT/'cell_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    fig, axes = plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
    datasets=list(grouped)
    axes[0,0].bar(datasets,[100*sum(r['full_better_than_count'] for r in grouped[d])/sum(r['trials'] for r in grouped[d]) for d in datasets],color='#326b9a')
    axes[0,0].set(ylabel='Trials with a larger integer lower bound (%)',title='Full multiscale versus observed count, same budget')
    axes[0,1].hist([r['m_ratio_to_count_budget'] for r in results],bins=24,color='#a06538')
    axes[0,1].axvline(1,color='black',linestyle='--')
    axes[0,1].set(xlabel='C70L1 mean / certified count baseline mean (risk 1/3)',ylabel='Cells',title='Expected sampling cost')
    for d in datasets:
        rs=grouped[d]
        axes[1,0].scatter([r['n'] for r in rs],[r['mean_gain_over_count'] for r in rs],s=12,alpha=.5,label=d)
    axes[1,0].set(xscale='log',xlabel='Threshold n',ylabel='Mean integer output increase',title='Paired output increase on fixed empirical distributions')
    axes[1,0].legend(fontsize=8)
    modes=list(METHODS)
    labels=['full','scale0+D','D same m','no D','D risk 1/3']
    bounds=[100*max(r['stats'][mode]['joint_upper_simultaneous_95'] for r in results) for mode in modes]
    if extra:
        labels.append('D risk .01');bounds.append(100*extra['maximum_cell_upper_simultaneous_95'])
    axes[1,1].bar(labels,bounds,color='#507e61')
    axes[1,1].axhline(100/3,color='black',linestyle='--',label='1/3 target')
    axes[1,1].set(ylabel='Worst simultaneous 95% failure upper bound (%)',title='Within each fixed method, across all tested cells')
    axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    small_path=OUT/'small_n/summary.json'
    small=json.loads(small_path.read_text()) if small_path.exists() else None
    small_text=(f'另有 {small["cells"]} 格小 n 合成诊断、{small["paths"]:,} 次模拟，完整算法零失败，95%同时上界为 {small["method_summary"]["full"]["maximum_cell_upper_simultaneous_95"]:.3%}。纯计数亦零失败、下界收益为零；该诊断未计入真实数据统计。'
                if small else '小 n 独立实现审计已完成；合成概率诊断另行存放，不计入真实数据统计。')
    mean_gain = summary['mean_gain']
    comparison=f'''本次完整算法在 {summary['paths']:,} 条独立泊松重采样路径中出现 {summary['full_failures']:,} 次联合失败。
最差单格失败率 {summary['max_full_failure_rate']:.2%}，单格单侧95%上界最大为 {summary['max_full_upper_95']:.2%}；
对全部 {summary['cells']} 格做 Bonferroni 校正后，完整算法最坏95%同时上界为 {summary['max_full_upper_simultaneous_95']:.2%}。
相同样本预算下，有 {summary['gain_paths']:,} 条路径（{summary['gain_paths']/summary['paths']:.2%}）的整数下界高于纯计数，平均增加 {mean_gain:.3f} 个类别；此描述性增量包含错误抬高输出的路径。
完整多尺度输出高于“第0尺度+D”的路径有 {summary['multiscale_gain_paths']:,} 条。
算法泊松样本均值是独立认证计数基线的 {summary['m_ratio_range'][0]:.2f}–{summary['m_ratio_range'][1]:.2f} 倍。'''
    if gain_audit:
        comparison+=f'\n独立审计确认，其中 {gain_audit["higher_output_full_joint_success"]:,} 次是合法且更紧的下界，{gain_audit["higher_output_full_joint_failure"]} 次是错误抬高；这些提高没有修复纯计数的失败，纯计数在相同大预算下原本全部成功。'
    extra_text=''
    if extra:
        extra_text=f'''为检查精度与预算的权衡，观察主结果后又预设了一组1%理论风险计数对照，保持640格和每格2000次，另用独立随机种子。
该基线共 {extra['total_failures']} 次联合失败，最坏场景失败率 {extra['worst_failure_rate']:.2%}，95%同时上界 {extra['maximum_cell_upper_simultaneous_95']:.2%}；C70L1预算仍是它的 {extra['c70_m_over_baseline_m_range'][0]:.2f}–{extra['c70_m_over_baseline_m_range'][1]:.2f} 倍。
这说明即使要求计数基线提供≤1%的理论风险，本批小支持分布中C70L1仍没有样本预算优势。该补充使用独立抽样，未宣称逐路径显著性；较低预算下的下界紧度可能不同。'''
    theory='''独立512位 Arb 区间运算确认：69.0062713812 < C* < 69.0062713813；C=70 时 H=0.4929199202809152 < 1/2。
原附录 A 的期望/系数引理可移除 b≤1 限制，因其证明仅使用 0<a<b 及其余尺度条件。
另外用992个连续闭区间，独立严格认证了C≥69、1≤L≤32的下偏差上界小于1.656775×10⁻²⁷。本次失败全部来自上偏差，与下偏差控制相符。
已有 C70L14 证书可覆盖当前相同统计量的 L≥14（n≥442414）部分，已有联合失败界约0.000879401164。
你粘贴的新有限证书仅覆盖 C∈[69.00627,69.00628]，不包含70；增加C同时改变次数取整和权重，不能直接用“采样更多”推出成功率单调。
本机未找到新PDF/TEX/ZIP，故小n上偏差证书、式(8)完整证明、临界点可取及必要性CLT的完整证明尚未完成逐行复核。式(8)的480次数值诊断未发现反例，但不能替代证明。
当前未补齐的 C70L1 全分布概率保证集中在1≤n≤442413的上尾控制。数值实验和系数一致性检查均不能代替这部分证明。'''
    if proof_complete:
        theory='''更新：本次已自主完成C₀=70、L₀=1的充分性证明，原先小n统一上尾控制的缺口已经补齐。
对任意正整数n、任意0<ε<1、任意有限或可数支持分布，在iid泊松模型下，原算法的联合失败概率严格小于0.331，因此成功概率严格大于0.669>2/3。
n=1…73的2701个(n,S)组合通过共同零计数/单例事件、已解析证明的同均值归约及大概率类别锚点控制；最坏上尾<0.330679396。n=74桥接<0.301518，n=75…403<0.291304，连续7≤L≤14小于0.183157，L≥14重放既有联合界<0.0009。
小L下偏差<1.657×10⁻²⁷，合并后仍有严格余量。连续区间、次数跳变、无限半线及所有小支持大小均有覆盖检查，验证器另做独立重放和缺项负控。
这是一套C=70的计算机辅助充分性证明；没有把优化器或随机模拟当作证明，也没有宣称形式化证明系统验证。精确最优常数C★及其可取性不属于本次结论。
旧theory_review.json/txt记录此前仅审查用户摘要时的状态；其中“小n未补齐”的判断现由results_proof_l01的新证明更新。'''
    methods='''从5个来源各预选8个群体，共40个固定经验分布；每来源按有效总计数取最大的6组及排序中部2组，要求S≥20。
清理规则在观察算法结果之前固定：剔除未定种、杂交/非目标类群及无效计数，保留来源和筛选审计。
ε取0.01、0.1、0.5、0.9；n依次取max(1,effε−1)、floor((effε+S)/2)、S、2S，并去除重复阈值；effε=1时第一个阈值不属于有效支持以下。
每格2000次，分4个预先固定种子块。按各物种经验概率独立生成Poi(m p_i)，总量M自然服从Poi(m)。
采样使用NumPy的binary64泊松随机数近似；Arb认证针对参数和统计量计算，不将随机数生成器称为严格区间采样器。
算法使用C0=70、L0=1、m=70n/[ε(1+ln n)]，完整保留多尺度、ceil(k)、校正和ceil输出/截断。
尾质量、有效支持和Yes/No/Gap标签用有理数精确计算。Gap不计二元检验错误，但仍检查估计区间。
联合失败事件是二元判断错误或S_hat不在[min(effε,n),min(S,n)]；报告该更严格事件。
浮点区间包围评分，跨整数/阈值时用高精度Arb复核。独立审计覆盖500组参数、665个尺度、13583个权重、3325个多项式点及133个指纹。
完整算法、单尺度、纯计数、去D消融共享同一指纹；较小预算计数另行抽样，预算经Arb认证Poi(mε)左尾≤1/3。
所有本次n满足L<14，所以相同预算下旧C70L14恰好等于纯计数。每种方法的置信上界分别覆盖全部格，未把五种方法合并为一个同时置信家族。'''
    limits=f'''这些实验衡量固定真实数据经验分布上的算法表现。真实生态数据存在调查努力差异、群体相关和探测偏差；现场总体中的未观测物种没有已知真值。
因此，实验真值是清理后经验分布的类别数及尾质量。重采样次数不等于新增野外样本数，跨年度/地点观测也不等于唯一动物个体。
本轮类别数S为26–603、阈值n为1–1206；约979万源计数没有转化为大n渐近区间。全部真实数据场景均位于L<14，恰好属于本轮新增、尚缺统一上尾证书的区间。
本次有 {summary['m_exceeds_source_N_cells']} 格的期望抽样量超过源群体已有总计数；有放回模拟仍有效，但不能解释为已有独立野外样本足够。
最大组加中等组的选样适合覆盖规模差异，不构成对全球生态系统的随机代表性样本。二名法清理仅为可复现过滤，不能替代完整分类学复核。BBS未额外过滤天气质量；LTRM白名单会排除新增但未列入旧文档的有效物种代码，详见清理审计。
eBird教程只有一个目标鸟种，出现记录库和测序reads的计量单位不同，未纳入本次物种个体计数主比较。
固定C70L1在本次小支持规模上具有经验准确性，但没有显示相对独立认证计数法的样本预算优势。样本量、统计精度和CPU开销应分开陈述。
这些结果可支撑应用验证及参数可运行性；CCF-A级发表还取决于理论新意、完整证明和明确优于强基线的适用区间。'''
    actions='''下一步优先取得新的完整证明和数值证书，把有限范围认证改为覆盖C=70（或覆盖[69.00627,70]并显式处理次数跳变）。
在理论闭环后，补充更大类别数、重尾/稀有类比例变化、独立调查或时间留出验证，并在相同目标风险下比较样本预算与计算时间。
若主要目标是现实数据上的低成本应用，保留本轮C70L1作为基准，再单独研究自适应/更低预算方案；新方案需单独界定保证。'''
    if proof_complete:
        limits=limits.replace('恰好属于本轮新增、尚缺统一上尾证书的区间','位于本次新证明覆盖的有限范围')
        actions='''C70L1的理论充分性和现有经验分布实验已形成闭环。后续应重点展示相同风险下的样本预算—下界紧度曲线、更大类别数及稀有尾部、独立调查或时间留出验证。
现有理论不改变计数基线的预算优势。若提出更低预算或自适应新方案，需要独立定义算法与保证；当前结果也不能单独证明CCF-A级论文的创新性和竞争力。'''
    failure_text='78次失败分为61次零假设下的错误拒绝与17次估计下界超过真实支持；没有低于有效支持的失败。全部78条路径经1024位起始Arb精度重算确认。最坏场景是LTRM群体ltrm_9300397bb9，S=n=36、ε=0.1，22/2000次错误拒绝。相同预算的纯计数及第0尺度+D均零失败；完整算法的额外尺度提高了下界，同时引入少量上偏差。'
    timing=f'主实验逐格累计耗时 {summary["elapsed_seconds"]:.2f} 秒，其中泊松采样 {sum(r["sampling_seconds"] for r in results):.2f} 秒、区间评分 {sum(r["scoring_seconds"] for r in results):.2f} 秒；不含原始数据清洗，也不是野外采样时间。采用本机批量指纹计算，不能据此推断逐样本在线延迟。'
    text='C0=70、L0=1：真实数据验证简报\n\n'+comparison+'\n'+failure_text+'\n\n1%风险计数对照\n'+extra_text+'\n\n理论核验\n'+theory+'\n\n实验方法\n'+methods+'\n'+timing+'\n\n解释与限制\n'+limits+'\n'+small_text+'\n\n建议\n'+actions+'\n'
    (OUT/'report.txt').write_text(text)
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C70L1真实数据实验</title><style>
body{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui,sans-serif;color:#243345;background:#fafbfc}h1{font-size:30px}h2{font-size:23px;margin-top:34px}p{white-space:pre-line}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid #dfe4e9}th{background:#eaf0f5}img{max-width:100%}.lead{background:#edf4fa;padding:20px;border-left:5px solid #326b9a}a{color:#165e91}code{background:#e8edf0;padding:2px 5px}@media print{body{margin:0;max-width:none}table{font-size:11px}h2{break-after:avoid}}</style>
<h1>C₀=70、L₀=1：真实数据验证简报</h1><div>2026-11-04 · OSST · 固定协议、保存指纹、可复现</div>'''
    content+='<p class="lead">'+html.escape(comparison)+'</p>'
    content+='<h2>使用的数据与结果</h2>'+table(['数据来源','群体','每群体计数N','类别S','参数格','重采样路径'],data_rows)
    content+='<p>原始归档、清理结果及每一群体的来源路径/校验值保存在 <a href="prepared/populations.json">populations.json</a> 与 <a href="prepared/">prepared/</a>。</p>'
    content+=table(['数据来源','联合失败次数','最坏单格失败率','输出高于D的路径','平均增加类别数','预算倍数中位数'],accuracy_rows)
    content+='<h2>方法比较</h2>'+table(['方法','二元错误','超过真实S','低于有效支持','联合失败','最坏95%同时上界'],method_rows)
    content+='<p>'+html.escape(failure_text)+'</p>'
    content+='<p>'+html.escape(extra_text)+'</p><p><a href="budget_01/summary.json">1%风险基线摘要</a> · <a href="budget_01/results.json">补充实验完整结果</a></p>'
    content+='<p>以上错误次数按同一参数格的重复路径统计；各方法重复使用相同路径不增加独立证据量。去D消融用于检查观测类别保护项。二元与估计事件可能重叠，因此联合失败不等于各列之和。</p>'
    content+='<img src="comparison.png" alt="配对下界收益、预算倍数、规模与失败概率上界"><p><a href="comparison.pdf">可导出PDF图</a> · <a href="cell_results.csv">完整参数格CSV</a> · <a href="results.json">全部统计JSON</a></p>'
    content+='<p>整数输出平均增量最大的5格（预设实验完成后作描述性排序）：</p>'+table(['格','来源','S','n','ε','平均增加类别','输出抬高比例'],gain_rows)
    content+='<h2>理论核验边界</h2><p>'+html.escape(theory)+'</p><p><a href="theory_review.json">理论核验明细</a> · <a href="implementation_checks.json">独立实现审计</a> · <a href="results_audit.json">实验结果独立复核</a></p>'
    if proof_complete:
        content+='<p><a href="../results_proof_l01/reliability_report.html">最新理论与可靠性复审</a> · <a href="../results_proof_l01/C70_L1_complete_proof.pdf">完整C70L1证明PDF</a> · <a href="../results_proof_l01/summary.json">新总证书</a></p>'
    content+='<h2>实验协议与适用范围</h2><p>'+html.escape(methods)+'</p><p>'+html.escape(timing)+'</p><p>'+html.escape(limits)+'</p><p>'+html.escape(small_text)+'</p>'
    content+='<h2>后续重点</h2><p>'+html.escape(actions)+'</p>'
    content+='<h2>复现</h2><pre>cd /home/snowy/OSST\nOPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.run\n.venv/bin/python real_l01/independent_checks.py --module real_l01.algorithm\n.venv/bin/python real_l01/theory_review.py --diagnostic-cases 120\nOPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.small_n\nOPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.budget_check\n.venv/bin/python -m real_l01.audit_results\n.venv/bin/python -m real_l01.report</pre><p><a href="protocol.json">预设协议</a> · <a href="source_hashes.json">执行源码校验</a> · <a href="source_snapshot/">执行源码快照</a> · <a href="raw/">逐格原始指纹与输出</a> · <a href="report.txt">纯文本简报</a></p></html>'
    (OUT/'report.html').write_text(content)
    print(text[:1800])


if __name__=='__main__':
    main()
