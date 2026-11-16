"""Build Chinese review, scientific plots, tables and a portable PDF."""
import html
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from real_stream_95.run import ROOT, OUT, dump, sha

NAMES={'midfish':'MidFish 淡水鱼','ltrm':'LTRM 河流鱼','nrmn':'NRMN 珊瑚礁鱼',
       'epa_nrsa':'EPA NRSA 河流鱼','bbs':'BBS 鸟类','mite':'螨类','bci':'BCI 热带树木',
       'moths':'蛾类','okland':'蜗牛','bird_north':'北部鸟类','bird_south':'南部鸟类',
       'spider_girdled':'环割地蜘蛛','spider_logged':'伐木地蜘蛛',
       'fish_2007_seine':'Inch Lake 2007 鱼','fish_2008_seine':'Inch Lake 2008 鱼'}
METHODS={'observed':'Observed species','chao1_bc':'Chao1 (bias corrected)',
         'C70_L14_report_maxD':'OSST L14 + observed',
         'C70_L1_report_maxD':'OSST L1 + observed',
         'C70_L1_median23_report_maxD':'OSST L1 median23 + observed',
         'count95':'Count test (95%)'}


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    OUT=parser.parse_args().out
    protocol=json.loads((OUT/'protocol.json').read_text());pops=protocol['populations']
    base=pd.read_csv(OUT/'results.csv');wrapped=pd.read_csv(OUT/'reported_lower.csv')
    execution=json.loads((OUT/'execution.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    hist=pd.read_csv(OUT/'historical_richness_diagnostic.csv')
    records=pd.DataFrame([{k:r[k] for k in ['id','dataset','N','S']} for r in pops])
    inventory=records.groupby('dataset',sort=False).agg(groups=('id','size'),total_N=('N','sum'),
        min_N=('N','min'),max_N=('N','max'),min_S=('S','min'),max_S=('S','max')).reset_index()
    inventory['名称']=inventory.dataset.map(NAMES)
    inventory.to_csv(OUT/'dataset_inventory.csv',index=False)
    allmethods=pd.concat([base,wrapped],ignore_index=True)
    aggregate=allmethods.groupby(['sampling','fraction','method']).agg(
        populations=('population','size'),available=('completed',lambda x:int((x>0).sum())),
        mape_available=('mape','mean'),within5_all=('within5_rate','mean'),
        coverage_available=('lower_interval_coverage','mean'),
        total_trials=('trials','sum'),outputs=('completed','sum')).reset_index()
    aggregate.to_csv(OUT/'aggregate_metrics.csv',index=False)
    # A common-population comparison prevents availability from changing denominators.
    common=[]
    for mode in ['without_replacement','iid_with_replacement']:
        for frac in protocol['fractions']:
            target=wrapped[(wrapped.sampling==mode)&(wrapped.fraction==frac)&
                (wrapped.method=='C70_L14_report_maxD')&(wrapped.completed>0)]
            ids=set(target.population)
            x=allmethods[(allmethods.sampling==mode)&(allmethods.fraction==frac)&
                allmethods.population.isin(ids)&allmethods.method.isin(METHODS)]
            for method,g in x.groupby('method'):
                common.append(dict(sampling=mode,fraction=frac,method=method,populations=len(g),
                     available=int((g.completed>0).sum()),mape_available=g.mape.mean(),
                     within5_rate_all=g.within5_rate.mean()))
    pd.DataFrame(common).to_csv(OUT/'common_population_metrics.csv',index=False)
    common_half=pd.DataFrame(common).query('sampling == "without_replacement" and fraction == 0.5').set_index('method')
    chosen=['midfish_7f62e037bd','ltrm_9300397bb9','nrmn_af72da7f36',
            'epa_nrsa_af3ce9cc24','bbs_3cfd793b60','mite','moths','bci','spider_logged']
    selected=allmethods[(allmethods.sampling=='without_replacement')&(allmethods.fraction==.5)&
                        allmethods.population.isin(chosen)]
    examples=[]
    for pid in chosen:
        g=selected[selected.population==pid].set_index('method');r=g.iloc[0]
        osst=g.loc['C70_L14_report_maxD'];raw=g.loc['C70_L14']
        examples.append(dict(数据=NAMES[r.dataset],population=pid,记录数=int(r.N),真实记录物种数=int(r.S),
             读取记录数=int(r.budget),直接计数均值=round(g.loc['observed','mean'],2),
             论文截断输出=round(raw['mean'],2),保留已发现物种后的下界=round(osst['mean'],2),
             下界均值相对误差百分比=round(100*osst.mape,2),
             下界在正负5百分比内的比例=round(100*osst.within5_rate,2),
             Chao1均值=round(g.loc['chao1_bc','mean'],2),
             下界输出2_5分位=osst.q025,下界输出97_5分位=osst.q975))
    examples=pd.DataFrame(examples);examples.to_csv(OUT/'examples_half_sample.csv',index=False)
    iid_examples=wrapped[(wrapped.sampling=='iid_with_replacement')&(wrapped.fraction==.5)&
                         (wrapped.method=='C70_L14_report_maxD')&wrapped.population.isin(chosen)]
    iid_examples.to_csv(OUT/'examples_iid95_half_sample.csv',index=False)
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    colorset=plt.get_cmap('tab10')
    for i,(method,label) in enumerate(METHODS.items()):
        g=aggregate[(aggregate.sampling=='without_replacement')&(aggregate.method==method)]
        axes[0].plot(100*g.fraction,100*g.mape_available,'o-',label=label,color=colorset(i))
        axes[1].plot(100*g.fraction,100*g.within5_all,'o-',label=label,color=colorset(i))
    axes[0].set(xlabel='Available records (% of file)',ylabel='Mean absolute relative error (%)',
                title='Available outputs only; population counts vary')
    axes[1].set(xlabel='Available records (% of file)',ylabel='Within 5% of file richness (%)',
                title='All 50 populations; unavailable = unsuccessful')
    axes[1].axhline(95,color='gray',ls=':',lw=1)
    for a in axes:a.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='upper right');fig.tight_layout()
    fig.savefig(OUT/'figures/budget_vs_richness.png',dpi=180)
    fig.savefig(OUT/'figures/budget_vs_richness.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for method,label,marker in [('observed','Observed','o'),('chao1_bc','Chao1','x'),
                               ('C70_L14_report_maxD','OSST L14 + observed','+')]:
        g=allmethods[(allmethods.sampling=='without_replacement')&(allmethods.fraction==.5)&(allmethods.method==method)]
        axes[0].scatter(g.S,g['mean'],s=22,label=label,marker=marker)
    axes[0].plot([0,620],[0,620],color='gray',ls=':');axes[0].set(xlabel='File reference richness',ylabel='Mean output',title='Half-file sampling, no replacement');axes[0].legend()
    g=base[(base.sampling=='iid_with_replacement')&(base.fraction==.5)&base.method.isin(['C70_L14','C70_L1_median23','count95'])]
    for i,(method,x) in enumerate(g.groupby('method')):
        axes[1].bar(i-.18,100*x.joint_success_rate.mean(),width=.36,color='steelblue',label='Valid bound / decision (available)' if i==0 else None)
        axes[1].bar(i+.18,100*x.within5_rate.mean(),width=.36,color='darkorange',label='Within 5% of S (all)' if i==0 else None)
    axes[1].set_xticks(range(3),sorted(g.method.unique()),rotation=10)
    axes[1].set(ylabel='Rate (%)',title='Correct lower bound is not accurate total richness',ylim=(0,115));axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'figures/targets_and_predictions.png',dpi=180)
    fig.savefig(OUT/'figures/targets_and_predictions.pdf');plt.close(fig)
    def metric(method,fraction,field):
        return float(aggregate[(aggregate.sampling=='without_replacement')&(aggregate.fraction==fraction)&(aggregate.method==method)].iloc[0][field])
    history=hist[(hist.regime=='twice_support')&(hist.method=='count')].groupby('epsilon')[['mape','within5_rate']].mean()
    summary=dict(**execution,audit_checks=audit['checks'],selected_source_records=int(records.N.sum()),
        historical_large_main_paths=1280000,historical_large_main_joint_failures=78,
        historical_worst_failure_rate=.011,historical_simultaneous95_failure_upper=.022710341756733414,
        half_sample={method:{field:metric(method,.5,field) for field in ['available','mape_available','within5_all']}
                     for method in METHODS},
        latest_theory='C70L1: joint failure <0.331 under iid Poisson. 23 independent copies + cap: failure <0.046627.',
        important='This task is an analysis, not a fresh end-to-end review of every historical analytic proof. Current proof artifact hashes checked; prior audit statuses reported separately.',
        no_publication_or_push_performed=True)
    dump(OUT/'summary.json',summary)
    sections=[]
    def section(title,body):sections.append((title,body))
    section('结论：复现已有扎实基础，但尚未证明能以95%概率准确预测总物种数',
        '原算法、改进常数算法、真实数据经验分布、理论证书和原始试次均已有对应材料。'
        '目前证据支持支持规模承诺检验和有效支持下界；不能把它们改写成“物种总数预测准确率95%”。'
        '本轮真正逐条生成、读取已有观测标签，并以完整文件的已记录物种数作为参照。未知自然总体的真实总物种数仍然未知。')
    section('1. 当前进度与版本',
        '原版 C0=2^26、L0=1024 已复现；先后完成 C0=1025、(70,16)、(70,14) 的实验与审计。'
        '最新本地 (70,1) 在所有 n>=1、0<epsilon<1 的 iid 泊松模型下有计算机辅助充分性证明，记录的联合失败上界为0.331。'
        '本轮核对了该证明清单中的源码和证书哈希，未重新宣称完成一遍全部解析证明审查。'
        '旧 (70,14) 的更强保证和新 (70,1) 的较弱单次保证应分开解释；此前README没有完整呈现11月4日这批结果。'
        '论文稿、PDF和数据已存在，仍缺乏在真实小物种数数据上优于强计数基线的样本效率证据，不能据此判断已经达到顶会录用要求。')
    section('2. 已经验证哪些真实数据',
        '历史10个经验总体来自螨类、南北鸟类、两种处理的蜘蛛、蛾、蜗牛、BCI树木、两年Inch Lake鱼，合计50,994条计数。'
        '11月新增MidFish、LTRM、NRMN、EPA NRSA、BBS五来源，各8个群体，共40个总体、9,785,885条计数，物种数26至603。'
        '这些40组按预定的六个最大记录数组和两个中部组选择，并非随机代表所有生态系统。'
        '汇总丰度被展开为单个观测标签；记录是捕获、鸟类观测、鱼类目击或树木计数，不一概代表独立且从未重复的动物个体。'
        '已下载但未纳入验证的数据不能计入实验：eBird当前仅有单种教程，AmazonFish/Ictio偏出现记录，CPUE与eDNA读数不能直接当作动物个体数。')
    sections[-1]=(sections[-1][0],sections[-1][1]+'此外旧实验还使用了NIPS/Enron词频、MovieLens评分、Online Retail商品类别；这些验证离散类别方法的跨领域表现，不计作生态物种验证。')
    section('3. 历史大型真实数据结果',
        '40总体 x 16配置 x 2000次 = 1,280,000条泊松路径，78次联合失败，平均0.00609%。'
        '最差配置1.10%，640配置的同时95%置信失败上界为2.271%，即在这个有限网格上有统计证据支持每格成功率至少97.729%。'
        '这不是所有分布的95%保证，也不是总物种数误差在5%内。'
        '259,098条路径有合法且更紧的下界，另17次错误抬高，61次错误拒绝；同预算计数原本全部成功。'
        '独立1%风险计数对照有35次失败，最差0.45%；OSST均值预算仍是其8.09至10.59倍。'
        '这些数据足以暴露真实失败和收益，但没有显示整体精度或采样成本优势。')
    section('4. 本轮逐条样本实验怎么做',
        f'统一50个总体，共{int(records.N.sum()):,}条源计数。每个总体、每种抽样模型固定128个独立随机种子，合计12,800条完整标签流；'
        '有放回随机读取记录索引形成iid经验分布样本，不放回随机排列全部记录形成有限文件样本。'
        '每条流检查1%、5%、10%、25%、50%、100%的可用记录预算，共3,600个方法配置、460,800次配对评估。'
        '方法和预算共享同一条流，不能把这些评估次数当成独立野外样本数。'
        '程序实际处理约25.18亿个重排/重采样标签，未增加真实调查记录。'
        '每个算法的n由预算和epsilon=0.1确定，完全不使用参照物种数S或有效支持eff。'
        '所有泊松长度先生成；超过预算时不输出，统计时计为不成功，不重抽直到成功。'
        '独立审计重建所有表中指标，每总体每模型重放一条完整流，并进行高精度评分核对。')
    section('5. 95%究竟指什么',
        '第一，理论正确概率：原始输出满足 min(eff_epsilon,n)<=S_hat<=min(S,n)，承诺检验正确。'
        'eff_epsilon是覆盖至少1-epsilon概率质量所需的最少物种数，epsilon不是算法犯错概率。'
        '第二，有限试验的95%置信区间：这是对某个成功率的估计区间。本轮给出逐格Clopper-Pearson区间及另列的Bonferroni同时区间。'
        '第三，总数误差不超过5%：|输出-S|<=0.05S。这是另外一个目标，原论文并不保证。'
        '表中的2.5%和97.5%分位数是重复运行输出的中间95%范围，不是自然总体物种数的95%置信区间。'
        '对 (70,1) 使用23次独立重复、二元多数表决和整数下界中位数，精确二项式失败界为0.04562647；'
        '再加0.001泊松预算截断风险，小于0.05。仅iid模型有此推论，不放回抽样不自动继承该定理。'
        '该放大保证下界有效，不保证接近S；在固定总预算下，分成23份会降低每份样本量。')
    section('6. 预测值与参照总数相差多少',
        '原始输出被n截断。即使已经看到35种，n=28时仍可能输出28；将它直接解释为总物种数是不合适的。'
        '因此新增一个明确标记为事后报告层的结果：max(已经看到的种数, 原始下界)。'
        '它不丢掉已知物种，仍是下界，不是无偏总数估计，也不是原论文输出。'
        f'使用一半记录、不放回抽取时，直接计数在50个总体上的平均绝对相对误差为{metric("observed",.5,"mape_available")*100:.2f}%，'
        f'误差落在±5%内的比例为{metric("observed",.5,"within5_all")*100:.2f}%。'
        f'Chao1相应为{metric("chao1_bc",.5,"mape_available")*100:.2f}%和{metric("chao1_bc",.5,"within5_all")*100:.2f}%，仍远非95%的总体预测成功率。'
        f'保留已发现物种后的C70L14可在{int(metric("C70_L14_report_maxD",.5,"available"))}/50个总体启动，'
        f'其可运行总体平均误差为{metric("C70_L14_report_maxD",.5,"mape_available")*100:.2f}%；'
        f'把不可运行也计入分母，±5%成功比例为{metric("C70_L14_report_maxD",.5,"within5_all")*100:.2f}%。'
        '不同方法可运行总体不同，不能直接用可运行平均数宣称优劣；common_population_metrics.csv另外给出共同总体的对照。'
        f'在共同可运行的42个总体、相同半文件预算下，直接计数的平均误差为{100*common_half.loc["observed","mape_available"]:.2f}%，'
        f'C70L1报告下界为{100*common_half.loc["C70_L1_report_maxD","mape_available"]:.2f}%，'
        f'Chao1为{100*common_half.loc["chao1_bc","mape_available"]:.2f}%；'
        f'三者的±5%成功比例依次为{100*common_half.loc["observed","within5_rate_all"]:.2f}%、'
        f'{100*common_half.loc["C70_L1_report_maxD","within5_rate_all"]:.2f}%、'
        f'{100*common_half.loc["chao1_bc","within5_rate_all"]:.2f}%。'
        '这些方法的实际读取数仍可能不同，OSST泊松长度及预算剩余均已记录，不把预算上限等同于实际成本。'
        '在符合理论抽样条件的iid半文件预算实验中，C70L14加预算截断后现有联合保证至少99.81%，但参照82种的MidFish报告下界均值仅77.85种，'
        '603种的NRMN均值550.28种，319种的EPA均值298.76种，225种的BCI均值204.66种。'
        '四者下界覆盖观测上均为100%，总数±5%成功比例却分别只有57.81%、0%、8.59%、0%，具体值见examples_iid95_half_sample.csv。'
        '完整文件读完后直接计数必然等于文件参照S，这是盘点而不是对未知野外物种的预测。Chao1全文件时仍可能外推，故本有限文件目标下会高估。')
    section('7. 排除n截断后，是否就能95%准确预测总数',
        '重新分析历史n=2S的诊断实验：这个n使用了参照真值，只能解释机制，不能冒充可部署预测。'
        f'即便不受n截断影响，同预算计数在epsilon=0.1时平均相对误差仍为{history.loc[.1,"mape"]*100:.2f}%，±5%比例为{history.loc[.1,"within5_rate"]*100:.2f}%；'
        f'epsilon=0.01时分别为{history.loc[.01,"mape"]*100:.2f}%和{history.loc[.01,"within5_rate"]*100:.2f}%。'
        '说明稀有物种漏检确实存在，不能只归因于阈值选得小。降低epsilon通常增加预算、改善结果，但没有将概率质量误差转换成种数相对误差。'
        '许多极稀有物种可以共用很小概率质量，有限样本无法普遍区分它们与不存在的物种；没有最小丰度等附加假设时，不应承诺总数相对误差保证。')
    section('8. 结果可靠性与尚待完成的工作',
        f'本轮{audit["checks"]:,}项主审计通过，保存每条路径的物种计数、泊松长度、输出、固定种子、代码快照及协议。'
        '数值评分使用向外舍入包围及必要时Arb复核，PRNG和泊松随机数仍是数值实现；没有把它们称为理想实数采样器。'
        '真实物种标签有来源限定、未定种过滤、同物异名合并；仍可能存在检测偏差、跨地点混合和记录相关。'
        '随机打乱只改变排列，不能证明原野外采样是iid；有放回验证仅针对文件经验分布。'
        '本轮所有epsilon固定为0.1；更多epsilon的直接流验证尚未跑，历史泊松网格覆盖0.01、0.1、0.5、0.9。'
        '当前完整程序的目标是规模检验/支持下界。若论文要声称总物种数预测，应另外定义估计器、做不使用S的参数选择、给出物种级误差与区间覆盖，并加入Chao/ACE等丰富度基线、时间/地点留出和调查事件整组验证。'
        '若继续原论文目标，则重点应放在同一风险和总预算下的额外正确拒绝、下界提升及大支持/稀有尾部优势；不能只增加重采样次数。')
    section('9. 复现与资料入口',
        '在项目根目录运行：OPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_stream_95.run --out 新目录；'
        '然后对同一输出目录依次运行 real_stream_95.derive、real_stream_95.audit、real_stream_95.report，各加 --out 新目录；'
        '不加 --out 时默认使用本次结果目录。'
        '完整原始表为results.csv，未截断报告层为reported_lower.csv，原文件顺序鱼类案例为fish_source_order.csv。'
        '本次只新增分析与实验目录，没有提交或推送GitHub，也没有覆盖既有冻结结果。')
    text='OSST复现进度、真实数据与95%目标专项分析\n\n'+'\n\n'.join(t+'\n'+b for t,b in sections)
    (OUT/'report.txt').write_text(text+'\n',encoding='utf-8')
    refs=[('LTRM原始采样说明','https://umesc.usgs.gov/data_library/fisheries/fish_page.html'),
          ('EPA NRSA调查定义','https://www.epa.gov/national-aquatic-resource-surveys/what-national-rivers-and-streams-assessment'),
          ('BBS调查说明','https://www.usgs.gov/centers/eesc/science/north-american-breeding-bird-survey'),
          ('Chao1公式：vegan官方文档','https://vegandevs.github.io/vegan/reference/specpool.html')]
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OSST真实数据分析</title>',
        '<style>body{max-width:1100px;margin:35px auto;padding:0 22px;font:16px/1.85 sans-serif;color:#17212b}h1,h2{line-height:1.35}table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd4db;white-space:nowrap}th{background:#edf3f8}img{max-width:100%}p{margin:1em 0}.note{padding:14px;background:#fff2d9}</style>',
        '<h1>OSST复现进度、真实数据与95%目标</h1><p>2026-11-16 · 本地实际运行与保存结果核对</p>']
    for idx,(title,body) in enumerate(sections):
        parts.append(f'<h2>{html.escape(title)}</h2><p>{html.escape(body)}</p>')
        if idx==2:parts.append(inventory.to_html(index=False))
        if idx==6:
            parts.append(examples.to_html(index=False,na_rep='预算不足',float_format=lambda v:f'{v:.2f}'))
            parts.append('<p class="note">示例都是读取50%源记录的128次不放回重排。数字是平均输出；分位范围描述重复结果，不是物种总数置信区间。</p>')
            parts.extend(f'<img src="figures/{name}.png" alt="{name}">' for name in ['budget_vs_richness','targets_and_predictions'])
    parts.append('<h2>资料来源</h2><ul>'+''.join(f'<li><a href="{u}">{t}</a></li>' for t,u in refs)+'</ul>')
    parts.append('<p><a href="results.csv">全部原始方法结果CSV</a> · <a href="reported_lower.csv">保留已发现种数的结果CSV</a> · <a href="report.pdf">PDF报告</a></p></html>')
    (OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CN',fontName='STSong-Light',fontSize=10,leading=16,wordWrap='CJK'))
    styles.add(ParagraphStyle(name='CNH',fontName='STSong-Light',fontSize=14,leading=20,spaceBefore=12,spaceAfter=8,wordWrap='CJK',keepWithNext=True))
    story=[Paragraph('OSST复现进度、真实数据与95%目标专项分析',styles['CNH']),Paragraph('2026-11-16',styles['CN'])]
    for idx,(title,body) in enumerate(sections):
        story.extend([Paragraph(title,styles['CNH']),Paragraph(html.escape(body),styles['CN']),Spacer(1,8)])
        if idx==6:
            table=[['数据','真实S','直接计数','OSST报告下界','Chao1']]
            for _,r in examples.iterrows():
                table.append([r['数据'],str(r['真实记录物种数']),f'{r["直接计数均值"]:.2f}',
                              '预算不足' if pd.isna(r['保留已发现物种后的下界']) else f'{r["保留已发现物种后的下界"]:.2f}',f'{r["Chao1均值"]:.2f}'])
            t=Table(table,colWidths=[130,55,75,100,75],repeatRows=1)
            t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f8')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.append(t)
            for name in ['budget_vs_richness','targets_and_predictions']:
                story.append(Image(str(OUT/f'figures/{name}.png'),width=480,height=183))
    story.append(Paragraph('资料来源',styles['CNH']))
    for title,url in refs:story.append(Paragraph(f'{title}: <link href="{url}">{url}</link>',styles['CN']))
    SimpleDocTemplate(str(OUT/'report.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45).build(story)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
