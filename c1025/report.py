"""Audit saved trials, produce scientific figures and a LaTeX experimental paper."""
from __future__ import annotations
import argparse,csv,html,json,math,shutil,subprocess,sys,zipfile
from fractions import Fraction
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from c1025.data import ROOT,DATA,load,save,sha
from c1025.core import cp_upper,truth,budget

LABELS={'C1025':'C = 1025','C67108864':'Original C = 2^26','C1024':'C = 1024','count_chernoff_99':'99% count baseline'}
NAMES={'bci':'Forest trees','mite':'Soil mites','bird_north':'Birds (North)','bird_south':'Birds (South)',
       'spider_girdled':'Spiders (Girdled)','spider_logged':'Spiders (Logged)','moths':'Moths','okland':'Land snails',
       'fish_2007_seine':'Fish (2007 seine)','fish_2008_seine':'Fish (2008 seine)'}


def read(out,name):return pd.read_csv(out/f'{name}.csv',dtype={'m':str})


def enrich_provenance(out):
    from benchmark_data import SOURCES
    records=json.loads((out/'data_sources.json').read_text())
    for item in records:
        if item.get('name')=='bci':
            item['pinned_raw_source']=SOURCES['bci']
            assert sha(ROOT/'data/large/raw/BCI.rda')==SOURCES['bci']['sha256']
    save(out/'data_sources.json',records)


def validate(out,frames):
    checks=0;files={};natural_checks=[]
    protocol=json.loads((out/'protocol.json').read_text())
    for rel,digest in protocol['source_hashes'].items():
        assert sha(ROOT/rel)==digest,rel;checks+=1
    for name,digest in protocol['data_hashes'].items():
        assert sha(DATA/'prepared'/name)==digest,name;checks+=1
    for suite,frame in frames.items():
        if suite not in ['synthetic','natural','curves']:continue
        blocks=pd.read_csv(out/f'{suite}_seed_blocks.csv')
        for raw_path,group in frame.groupby('raw_file',sort=False):
            path=out/raw_path;files[raw_path]=sha(path)
            with np.load(path) as raw:
                model=SimpleNamespace(multiplicities=raw['multiplicities'],probabilities=raw['probabilities'],
                                      support=int(raw['multiplicities'].sum()))
                first=group.iloc[0];label,tail,eff=truth(model,int(first.n),float(first.epsilon))
                order=sorted(range(len(raw['means'])),key=lambda j:int(raw['means'][j]))
                assert np.all(np.diff(raw['observed'][order],axis=0)>=0)
                checks+=1
                for r in group.itertuples():
                    D=raw['observed'][r.raw_index];reject=D>r.n;lower=np.clip(D,1,r.n)
                    assert r.truth==label and r.effective_support==eff and r.S==model.support
                    assert Fraction(r.delta_n_exact)==tail
                    assert np.all((D>=0)&(D<=r.S))
                    assert int(r.m)==budget(r.n,str(r.epsilon),r.method)
                    assert len(D)==r.trials and str(raw['means'][r.raw_index])==r.m
                    assert str(raw['methods'][r.raw_index])==r.method
                    assert int(reject.sum())==r.rejects
                    errors=None if r.truth=='Gap' else int(np.sum(reject if r.truth=='Yes' else ~reject))
                    if errors is not None:
                        assert errors==int(r.errors)
                        assert abs(r.error_rate-errors/r.trials)<1e-12
                        assert abs(cp_upper(errors,r.trials,.05/frame.errors.notna().sum())-r.error_upper95_simultaneous)<1e-12
                    else:assert pd.isna(r.errors) and pd.isna(r.error_rate)
                    assert int(np.sum((lower>r.S)|(lower<min(r.n,r.effective_support))))==r.lower_errors
                    assert abs(cp_upper(r.lower_errors,r.trials,.05/len(frame))-r.lower_error_upper95_simultaneous)<1e-12
                    assert abs(r.reject_rate-r.rejects/r.trials)<1e-12
                    assert math.isclose(r.mean_lower,float(lower.mean()),rel_tol=1e-14,abs_tol=1e-10)
                    assert math.isclose(r.mean_observed,float(D.mean()),rel_tol=1e-14,abs_tol=1e-10)
                    b=blocks[(blocks.cell==r.cell)&(blocks.method==r.method)]
                    assert len(b)==4 and int(b.trials.sum())==r.trials and int(b.rejects.sum())==r.rejects
                    if errors is not None:assert int(b.errors.sum())==errors
                    assert int(b.lower_errors.sum())==r.lower_errors
                    checks+=18
                    if suite=='natural':
                        counts=sorted(map(int,load(r.dataset)),reverse=True);N=sum(counts);eps=Fraction(str(r.epsilon));count_tail=Fraction(sum(counts[r.n:]),N)
                        count_label='Yes' if len(counts)<=r.n else 'No' if count_tail>=eps else 'Gap'
                        discard=0;mass=0
                        for c in counts[::-1]:
                            if Fraction(mass+c,N)<=eps:mass+=c;discard+=1
                            else:break
                        count_eff=max(1,len(counts)-discard)
                        assert count_label==r.truth and count_eff==r.effective_support
                        natural_checks.append(dict(dataset=r.dataset,n=r.n,epsilon=r.epsilon,exact_count_tail=str(count_tail),effective_support=count_eff,method=r.method))
                        checks+=2
    audit=json.loads((out/'algebra_audit/results.json').read_text());assert audit['passed'];checks+=1
    for filename in ['fractional_output.json','exact_endpoint_certificates.json']:
        assert json.loads((out/'algebra_audit'/filename).read_text())['passed'];checks+=1
    for file in ['reference_api_runs.json','natural_public_api.json']:
        runs=json.loads((out/file).read_text())
        for r in runs:
            assert r['numeric_decision_certified'] and r['decision']==('No' if r['observed']>r['n'] else 'Yes')
            assert r['support_lower_bound']==min(r['n'],max(1,r['observed']));checks+=2
    # Validate all declared experiment phases completed, not just tables existing.
    for phase in ['synthetic','natural','curves','dependence','reference','audit','stream']:
        assert json.loads((out/f'execution_{phase}.json').read_text())['status']=='complete';checks+=1
    pd.DataFrame(natural_checks).to_csv(out/'natural_exact_truth_audit.csv',index=False)
    save(out/'result_validation.json',dict(passed=True,checks=checks,raw_files=len(files),sha256=files,
         scope='Recomputed every saved primary decision, error, lower bound, seed aggregate and simultaneous confidence bound; natural targets independently recomputed from exact integer counts.'))
    return audit


def analyze(out,frames,audit):
    rows=[]
    for suite in ['synthetic','natural']:
        for method,group in frames[suite].groupby('method'):
            promised=group[group.truth!='Gap']
            rows.append(dict(suite=suite,method=method,cells=len(group),promised_cells=len(promised),
                promised_trials=int(promised.trials.sum()),errors=int(promised.errors.sum()),
                worst_error=float(promised.error_rate.max()),worst_simultaneous_upper=float(promised.error_upper95_simultaneous.max()),
                worst_lower_failure_upper=float(group.lower_error_upper95_simultaneous.max())))
    pd.DataFrame(rows).to_csv(out/'risk_summary.csv',index=False)
    cost=frames['synthetic'].pivot_table(index=['n','epsilon','scenario'],columns='method',values='m',aggfunc='first')
    for c in cost.columns:cost[c]=cost[c].map(int)
    cost['ratio_new_to_count']=cost.C1025/cost.count_chernoff_99
    cost['ratio_original_to_new']=cost.C67108864/cost.C1025
    cost.reset_index().to_csv(out/'cost_summary.csv',index=False)
    curves=frames['curves'];curve_rows=[]
    for (n,eps),group in curves.groupby(['n','epsilon']):
        passed=[]
        for method,cells in group[group.truth!='Gap'].groupby('method'):
            if cells.error_upper95_simultaneous.max()<.01:passed.append(float(method[1:]))
        curve_rows.append(dict(n=int(n),epsilon=float(eps),smallest_passing_tested_C=min(passed) if passed else None,
                              scope='Only the frozen selected scenarios and tested constants; not an optimal universal coefficient'))
    pd.DataFrame(curve_rows).to_csv(out/'tested_constant_thresholds.csv',index=False)
    feasibility=frames['natural_feasibility'];public=len(json.loads((out/'reference_api_runs.json').read_text()))+len(json.loads((out/'natural_public_api.json').read_text()))
    s=dict(primary_method_cells=sum(len(frames[x]) for x in ['synthetic','natural','curves']),
       primary_method_trial_evaluations=sum(int(frames[x].trials.sum()) for x in ['synthetic','natural','curves']),
       primary_independent_poisson_paths=sum(int(frames[x].drop_duplicates('cell').trials.sum()) for x in ['synthetic','natural','curves']),
       synthetic_base_cells=int(frames['synthetic'].cell.nunique()),natural_base_cells=int(frames['natural'].cell.nunique()),
       natural_datasets=len(json.loads((out/'datasets.json').read_text())),public_api_runs=public,
       audit_assertions=audit['total_assertions'],risk_summary=rows,
       new_to_count_ratio_min=float(cost.ratio_new_to_count.min()),new_to_count_ratio_max=float(cost.ratio_new_to_count.max()),
       original_to_new_ratio_min=float(cost.ratio_original_to_new.min()),original_to_new_ratio_max=float(cost.ratio_original_to_new.max()),
       natural_budget_sufficient_cells=int(feasibility.has_mean_budget.sum()),natural_budget_cells=len(feasibility),
       duplication_worst_error=float(frames['dependence'].error_rate.max()),group_stress_worst_error=float(frames['natural_group_stress'].error_rate.max()),
       holdout_gap_cells=int((frames['group_holdout'].truth=='Gap').sum()),
       holdout_worst_error=float(frames['group_holdout'].error_rate.max()),
       finite_file_attempts=len(json.loads((out/'finite_file_runs.json').read_text())),
       finite_file_overflows=sum(r['status']=='overflow' for r in json.loads((out/'finite_file_runs.json').read_text())),
       limitations=['Working theorem assumption, not proof of universal correctness from simulation','Practical sizes never exercise the polynomial branch',
       'C1024 also succeeds on this finite practical grid; integer minimality is not empirically established',
       'Finite empirical distributions do not reveal undiscovered ecosystem species','Most original files do not meet C1025 budget',
       'Compressed occupancy computation is not physical collection of the nominal sample budget','No new field survey, formal proof-assistant certificate, or venue acceptance guarantee'])
    save(out/'summary.json',s);return s,cost


def figures(out,frames,summary,cost):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    descriptions=[]
    def export(fig,name,caption,zh):
        fig.savefig(dest/f'{name}.pdf');fig.savefig(dest/f'{name}.png',dpi=170);plt.close(fig)
        descriptions.append(dict(name=name,caption=caption,description_zh=zh))
    methods=['C67108864','C1025','C1024','count_chernoff_99'];risk=pd.DataFrame(summary['risk_summary'])
    fig,axes=plt.subplots(1,2,figsize=(10,3.2),sharey=True)
    for ax,suite in zip(axes,['synthetic','natural']):
        values=risk[risk.suite==suite].set_index('method').loc[methods]
        ax.bar(np.arange(4),100*values.worst_simultaneous_upper,color=['#999999','#0072b2','#56b4e9','#d55e00'])
        ax.axhline(1,color='black',ls='--',label='1% target');ax.set_xticks(range(4),[LABELS[m].replace(' ','\n',1) for m in methods],fontsize=10)
        ax.set_title('Synthetic' if suite=='synthetic' else 'Empirical resampling');ax.set_ylim(0,1.1)
    axes[0].set_ylabel('Worst simultaneous 95%\nerror upper bound (%)');fig.tight_layout()
    export(fig,'risk','Worst cellwise error upper confidence bounds, Bonferroni-adjusted within each suite over all promised method cells. Zero observed errors are not zero risk.','柱子是测试网格中最差单元的错误率置信上界，越低越好；虚线是 1% 目标。')
    fig,axes=plt.subplots(1,2,figsize=(10,3.2));d=cost.reset_index();d=d[(d.epsilon==.01)&(d.scenario=='uniform_null')].sort_values('n')
    for method in ['C67108864','C1025','count_chernoff_99']:axes[0].loglog(d.n,d[method].map(float),'o-',label=LABELS[method])
    axes[0].set(xlabel='n',ylabel='Prescribed Poisson mean',title='epsilon = 0.01');axes[0].legend(fontsize=10)
    axes[1].semilogx(d.n,d.ratio_new_to_count,'o-',color='#0072b2');axes[1].axhline(1,color='black',ls='--');axes[1].set(xlabel='n',ylabel='C1025 budget / count-baseline budget',title='Finite-scale conservatism');fig.tight_layout()
    export(fig,'budgets','Prescribed acquisition budgets, not simulator run times. The original coefficient decreases by about 65,472x, but a 99% Chernoff count baseline remains cheaper over all tested sizes.','左图比较理论样本预算；右图显示新常数仍比简单计数基线多用多少倍样本，不能把预算倍数说成运行加速。')
    fig,axes=plt.subplots(2,2,figsize=(10,6.1),sharex=True,sharey=True)
    for ax,n in zip(axes.flat,[10,1000,1000000,1000000000000]):
        d=frames['curves'];d=d[(d.n==n)&(d.epsilon==.01)]
        for scenario in ['uniform_boundary','diffuse_boundary','multiscale_boundary']:
            g=d[d.scenario==scenario].copy();g['C']=g.method.str[1:].astype(float);g=g.sort_values('C')
            ax.semilogx(g.C,100*g.error_rate,'o-',ms=3,label=scenario.replace('_boundary',''))
        ax.axvline(1025,color='black',ls=':',lw=1);ax.axhline(1,color='gray',ls='--',lw=1);ax.set_title(f'n = {n:,}');ax.set_ylim(-2,102)
    axes[0,0].legend(fontsize=10);axes[1,0].set_xlabel('Sampling coefficient C');axes[1,1].set_xlabel('Sampling coefficient C');axes[0,0].set_ylabel('False-negative rate (%)');axes[1,0].set_ylabel('False-negative rate (%)');fig.tight_layout()
    export(fig,'constant_sweep','Coefficient sensitivity for three alternatives at epsilon=0.01. Each point has 4,000 trials; the CSV includes simultaneous confidence bounds. Dashed horizontal line: 1% target. Vertical dotted line: C=1025.','展示常数过小时会失败，以及有限规模下远小于 1025 的常数也可能成功；不将网格阈值称为全局最优。')
    fig,ax=plt.subplots(figsize=(8.5,4.2))
    for row in json.loads((out/'datasets.json').read_text()):
        c=np.sort(load(row['dataset']))[::-1];ax.loglog(np.arange(1,len(c)+1),c/c.sum(),label=NAMES[row['dataset']])
    ax.set(xlabel='Species abundance rank',ylabel='Empirical probability',title='Published natural abundance distributions');ax.legend(ncol=2,fontsize=10);fig.tight_layout()
    export(fig,'natural_shapes','Rank-abundance distributions for ten published empirical populations spanning seven biological groups. These are observed finite populations, not complete ecosystem inventories.','真实数据涵盖七类生物、十个总体；曲线描述丰度不均匀程度，不是预测准确率。')
    f=frames['natural_feasibility'];minimum=f.groupby('dataset').budget_to_records.min().sort_values();typical=f[(f.n==f.S)&(f.epsilon==.1)].set_index('dataset').budget_to_records.reindex(minimum.index)
    fig,ax=plt.subplots(figsize=(9,4.7));x=np.arange(len(minimum));ax.barh(x-.18,minimum,.35,label='Best tested n, epsilon');ax.barh(x+.18,typical,.35,label='n = observed support, epsilon = 0.1')
    ax.axvline(1,color='black',ls='--');ax.set_xscale('log');ax.set_yticks(x,[NAMES[k] for k in minimum.index]);ax.set_xlabel('Required mean budget / original record count');ax.legend(fontsize=10);fig.tight_layout()
    export(fig,'natural_feasibility','Budget feasibility: ratios above one mean that the original file has fewer observations than the prescribed mean. Even ratios below one do not establish iid field sampling or guarantee a Poisson draw fits in the file.','大于 1 表示原始记录数连理论平均预算都达不到；这里只能重采样验证经验分布，不能冒充新增野外观测。')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,key,col in [(axes[0],'dependence','duplicate_factor'),(axes[1],'natural_group_stress','block_size')]:
        d=frames[key];d=d[d.epsilon==.1]
        for name,g in d.groupby('dataset'):ax.semilogx(g[col],100*g.error_rate,'o-',label=NAMES[name])
        ax.axhline(1,color='gray',ls='--');ax.set_xlabel('Repeated labels per parent' if col=='duplicate_factor' else 'Labels per selected natural group')
    axes[0].set_ylabel('False-negative rate (%)');axes[0].set_title('Artificial identical-label duplication');axes[1].set_title('Artificial sampling within real groups');axes[0].legend(fontsize=9);fig.tight_layout()
    export(fig,'dependence','Stress tests preserve the marginal category distribution but violate independence. Group selection is abundance-weighted. Error increases measure artificial perturbations, not an estimate of actual field correlation.','即使每条观测的物种概率不变，重复或成组相关也可能破坏保证；这些是压力测试，不是测得的自然界相关强度。')
    fig,ax=plt.subplots(figsize=(8.5,3.5));d=frames['stream'];d=d[d.epsilon==.01].copy();x=np.arange(len(d));base=d.C1025_fixed_expected_samples.astype(float)
    ax.bar(x-.25,np.ones(len(d)),.25,label='C1025 fixed budget');ax.bar(x,d.count_fixed_expected_samples/base,.25,label='99% count fixed budget');ax.bar(x+.25,d.count_immediate_mean_samples/base,.25,label='99% count, immediate rejection')
    ax.set_yscale('log');ax.set_xticks(x,[f'{r.n}\n{r.scenario.replace("_boundary", " alt").replace("_null", " null")}' for r in d.itertuples()],fontsize=9);ax.set_ylabel('Acquisition cost / C1025 mean');ax.legend(fontsize=10);fig.tight_layout()
    export(fig,'stream','Actual label-stream simulation at epsilon=0.01 with 300 trials per cell. The count baseline stops immediately upon observing n+1 species; a prescribed Poisson cap controls acceptance. C1025 bars are prescribed means, not an additional simulated giant stream.','从实际逐条标签流检查早停成本，避免只比较批处理检查点而夸大采样效率。')
    fig,axes=plt.subplots(1,2,figsize=(10,3.2))
    for c,g in frames['runtime'].groupby('C0'):
        g=g.sort_values('n');label=f'C = {c}'
        axes[0].loglog(g.n,g.count_generation_median_seconds*1000,'o-',label=label)
        axes[1].loglog(g.n,g.reference_decision_median_seconds*1000,'o-',label=label)
    axes[0].set(xlabel='Number of explicit categories',ylabel='Median milliseconds',title='Generate category counts');axes[1].set(xlabel='Number of explicit categories',ylabel='Median milliseconds',title='Full public reference decision');axes[0].legend(fontsize=10);fig.tight_layout()
    export(fig,'runtime','Ten timed repetitions after one warm-up in each isolated process. Independent Poisson category-count generation and certified decision computation are separated. These timings exclude physical data acquisition.','区分模拟生成数据的耗时和完整接口判定耗时；这些不是采集理论样本量的现实耗时。')
    save(out/'figures.json',descriptions);return descriptions


def tex(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('^',r'\textasciicircum{}')]:s=str(s).replace(a,b)
    return s


def write_paper(out,frames,summary,figures,audit):
    s=summary;risk=pd.DataFrame(s['risk_summary']).set_index(['suite','method']);syn=risk.loc[('synthetic','C1025')];nat=risk.loc[('natural','C1025')]
    body=[]
    def p(x):body.append(x+'\n')
    def section(x):p('\\section{'+x+'}')
    def fig(name):
        item=next(f for f in figures if f['name']==name)
        p(r'\begin{figure}[tbp]\centering\includegraphics[width=.96\linewidth]{figures/'+name+r'.pdf}\caption{'+tex(item['caption'])+r'}\end{figure}')
    p(r'\begin{abstract}')
    p(fr'We evaluate the explicit coefficient $C_0=1025$ for Wang\textquotesingle s support-size tester, conditional on the accompanying proof and with $L_0=1024$ unchanged. The frozen main design comprises {s["primary_independent_poisson_paths"]:,} independent simulated Poisson-process paths and {s["primary_method_trial_evaluations"]:,} paired method evaluations. Synthetic sizes range from $1$ to $10^{{12}}$ across five separation parameters and up to fifteen distribution scenarios. Ten published empirical populations cover trees, mites, birds, spiders, moths, snails, and fish. No errors are observed for $C_0=1025$ in the promised primary cells; worst simultaneous 95\% error upper bounds are {100*syn.worst_simultaneous_upper:.3f}\% on synthetic and {100*nat.worst_simultaneous_upper:.3f}\% on empirical populations. However, all practical sizes use the observed-count branch, a simple certified count baseline requires fewer samples, and only {s["natural_budget_sufficient_cells"]} of {s["natural_budget_cells"]} ecological configurations have as many original records as the prescribed mean. Correlation stress tests expose violations of the iid requirement. Full-degree interval checks and public-API executions support implementation fidelity without substituting for a universal proof or a field validation.')
    p(r'\end{abstract}')
    section('Question and claim boundaries')
    p(r'We test the proposed constant reduction from $2^{26}$ to $1025$ in the supplied manuscript~\cite{wang}. For integer $n\ge1$, separation $0<\epsilon<1$, and $L=1+\log n$, the prescribed Poisson mean is $m=\lceil C_0n/(\epsilon L)\rceil$. The separation parameter $\epsilon$ is not the target error probability, which is $0.01$. Yes instances satisfy $S\le n$; No instances satisfy $\Delta_n(p)\ge\epsilon$. Gap instances satisfy neither promise and receive no classification-error label. We also check $\min\{\mathrm{eff}_\epsilon(p),n\}\le\widehat S\le S$ for every distribution, including Gap instances.')
    p(r'The coefficient reduction is an analytic refinement of the same construction, not a new asymptotic sample-complexity rate. Empirical evidence addresses finite tested distributions. It cannot establish uniform correctness over all distributions or integer minimality. In particular, $C_0=1024$ also succeeds on the tested practical grid; its theoretical obstruction lies near $n=e^{1023}\approx10^{444.28}$. The accompanying proof is included in the appendix for a precise statement of scope.')
    section('Implementation and comparators')
    p(r'The public reference interface now accepts an explicit sampling coefficient and records it in every output, while retaining the original coefficient as its default. Parameters, branch decisions, integer rounding and outputs are computed with Arb intervals and adaptive precision. The command \texttt{python paper\_reference.py demo --n 1000 --epsilon 0.1 --c0 1025} runs the new coefficient. All tested practical $n$ satisfy $L<1024$, so the literal algorithm is $D>n$ for rejection and $\widehat S=\min\{\max\{D,1\},n\}$. We do not force a polynomial branch at small $n$ or attribute tuned practical variants to this theorem.')
    p(r'Comparators are the original $C_0=2^{26}$ construction, $C_0=1024$ as a coefficient control, and a count baseline with the same 1\% target. For the latter, let $a=\log100$ and choose $m_{\rm count}=\lceil(n+a+\sqrt{a^2+2na})/\epsilon\rceil$. Discovery-time domination and a Poisson Chernoff bound certify its type-II error below 1\%; its type-I error is zero. This is a conservative analytic count baseline, not a data-tuned oracle. Constant sweeps below 1025 are sensitivity experiments without a claimed universal theorem.')
    section('Experimental protocol and statistical analysis')
    p(r'The primary grid uses $n\in\{1,2,10,10^2,10^3,10^4,10^6,10^9,10^{12}\}$ and $\epsilon\in\{0.001,0.01,0.05,0.1,0.25\}$. Scenarios include uniform and smaller-support nulls, dominant heads with extremely rare tails, piecewise power-law, lognormal, discrete Gaussian-shaped and geometric-shaped nulls, near-boundary uniform alternatives, diffuse and ultradiffuse tails, multiscale alternatives, clear alternatives, and two Gap controls. Blocked shapes are explicitly constructed discrete distributions, not claimed exact continuous Gaussian or lognormal laws. Invalid null variants at $n=1$ are omitted prospectively. Promise labels and effective-support integers are checked with rational arithmetic; empirical-data labels are additionally checked from integer counts.')
    p(r'Each primary distribution/parameter cell has four independent blocks of 2,500 trials. Methods share nested Poisson-process occupancies within a trial for paired comparisons. For a group of $g$ equal-probability categories, unseen categories are discovered over an increment $\Delta m$ by binomial thinning with probability $1-e^{-p\Delta m}$. This is the exact joint occupancy law in real arithmetic; numerical random generation uses binary64. It does not materialize the nominal sample stream or recover its realized length $M$. We save every method\textquotesingle s observed count, not just averages. The coefficient sweep uses 4,000 trials per cell and fourteen coefficients from 0.25 to 1025.')
    p(r'For $k$ failures in $T$ trials, we report the one-sided Clopper--Pearson upper bound $\operatorname{Beta}^{-1}(1-\alpha;k+1,T-k)$, with value one when $k=T$. Bonferroni adjustment sets $\alpha=0.05/K$ across all promised method cells within each suite. Lower-bound failures use a separate family over all cells, including Gap controls. These are suite-specific simultaneous statements, not a single global 95\% statement across every analysis. No trial count or parameter was changed in response to observed errors. A saved protocol amendment fixes duplicate output-log keys only; it does not change completed simulations. Supplemental API, runtime and group checks were separately recorded after primary summary inspection.')
    section('Natural abundance data and validation targets')
    p(r'We use abundance counts rather than biomass, percentage cover, or incidence frequencies. BCI trees and soil mites come from vegan~\cite{vegan}; birds and two spider treatments from iNEXT~\cite{inext}; moths and land snails from sads~\cite{sads}; seine-caught fish are stratified by year in InchLake1~\cite{fish}. Source commits, SHA256 digests, licensing notes and cleaning rules are saved in \texttt{data\_sources.json}. BCI refers to the 1982 large-tree subset, not a modern full-forest inventory. Historical abundance vectors have no recoverable event order. No species are invented to enlarge a real dataset.')
    p(r'\begin{table}[tbp]\centering\small\begin{tabular}{lrrr}\toprule Empirical population & Records & Species & Natural groups\\\midrule')
    for d in json.loads((out/'datasets.json').read_text()):p(fr'{tex(NAMES[d["dataset"]])} & {d["N"]:,} & {d["S"]} & {d["groups"] or "--"}\\')
    p(r'\bottomrule\end{tabular}\caption{Published finite populations. Groups are spatial plots, soil cores or capture events; separate bird sites and spider treatments are not pooled.}\end{table}')
    p(r'The confirmatory numerical target is each cleaned empirical distribution $\widehat p_i=c_i/\sum_j c_j$. Independent Poisson resampling provides known finite truth, but cannot discover species absent from the source file. For each $\epsilon$, thresholds are the unique positive values among $\lfloor S/2\rfloor$, $\mathrm{eff}_\epsilon-1$, $\mathrm{eff}_\epsilon$, $S$, and $\lceil1.5S\rceil$. Truth is used to construct a benchmark and score results; the decision interface receives only the observed fingerprint and predeclared $n,\epsilon,m$. This is not a deployment procedure for choosing $n$ from an unknown ecosystem.')
    fig('natural_shapes')
    section('Risk and acquisition-budget results')
    p(r'\begin{table}[tbp]\centering\small\begin{tabular}{llrrr}\toprule Suite & Method & Errors / trials & Worst error & Simultaneous upper\\\midrule')
    for r in summary['risk_summary']:
        p(fr'{r["suite"].capitalize()} & {tex(LABELS[r["method"]])} & {r["errors"]:,}/{r["promised_trials"]:,} & {100*r["worst_error"]:.3f}\% & {100*r["worst_simultaneous_upper"]:.3f}\%\\')
    p(r"\bottomrule\end{tabular}\caption{Errors on promised instances. ``Worst error'' is the largest cellwise observed rate, not a pooled rate. Confidence families include all methods in the respective suite.}\end{table}")
    p(fr'The main experiment comprises {s["synthetic_base_cells"]} synthetic cells and {s["natural_base_cells"]} empirical cells, each evaluated by four methods. The C1025 joint lower-bound failure upper limits are {100*syn.worst_lower_failure_upper:.3f}\% and {100*nat.worst_lower_failure_upper:.3f}\%, respectively, below the theorem\textquotesingle s 2\% joint target. The count baseline also meets the 1\% error target on the tested grid. Thus the experiments support implementation correctness and conservative guarantees, not practical sample-efficiency dominance.')
    p(f'Across the synthetic grid, the C1025 budget is {s["new_to_count_ratio_min"]:.1f}--{s["new_to_count_ratio_max"]:.1f} times the count-baseline budget. Reducing the original coefficient changes its prescribed mean by approximately 65,472 times; this does not imply an equal wall-clock speedup or equal savings relative to a stronger finite-scale baseline. The coefficient sweep finds much smaller sufficient tested constants on its restricted scenarios. Such selected grid values are not universal optima.')
    fig('risk');fig('budgets');fig('constant_sweep')
    section('What can the original ecological records validate?')
    p(fr'Only {s["natural_budget_sufficient_cells"]}/{s["natural_budget_cells"]} ecological configurations have $N\ge m_{{1025}}$. We also report $\Pr\{{\operatorname{{Poi}}(m)>N\}}$, because $N\ge m$ alone does not guarantee a feasible Poisson draw. The only mean-feasible configurations are from the mite dataset. We run 200 finite-file Poisson subsampling attempts for each and retain every overflow rather than redrawing. These without-replacement file analyses are descriptive: the underlying spatial core collection has not been established to be an iid sample from the desired external population. For insufficient files, resampling with replacement does not create additional independent ecological evidence.')
    p(fr'In the mite configurations, $(n,\epsilon,m,N)=(6,0.25,8812,9800)$ and $(7,0.25,9743,9800)$. The first has no overflows in 200 attempts; the second has 52 in 200, consistent with its nominal overflow probability of about 0.280. Both thresholds already lie below the 35 observed species, so these runs do not establish previously unknown ecological richness or certified acceptance.')
    p(r'Observing more than $n$ valid species directly refutes $S\le n$ without an iid assumption. Conversely, seeing at most $n$ species in an inadequate or biased survey does not certify acceptance. In particular, neither a high empirical resampling success rate nor an algorithmic Yes proves a complete species inventory. Sampling effort, detection bias, spatial clustering and the chosen biological sampling unit must be specified for a field application.')
    fig('natural_feasibility')
    section('Dependence, held-out groups and stopping rules')
    p(fr'We test two controlled departures from independence while retaining the same marginal species distribution: identical-label duplication and multiple labels drawn within real abundance-weighted groups. Their worst observed false-negative rates are {100*s["duplication_worst_error"]:.3f}\% and {100*s["group_stress_worst_error"]:.3f}\%, respectively. These deliberate stresses are not estimates of natural field dependence. Selecting a group independently for every individual with probability proportional to its abundance recovers the pooled empirical iid target; batching individuals within the same selected group does not.')
    p(f'Five 70/30 group splits per grouped dataset select $n$ using training groups only. Held-out finite populations are independently resampled and scored using their own truth. No errors are observed on promised held-out cells; {s["holdout_gap_cells"]} cells are Gap instances and are not silently assigned a failure label. This measures robustness to finite empirical heterogeneity, not unknown species outside both splits.')
    p(r'A separate actual-label experiment uses $n\in\{10,100,1000\}$, $\epsilon\in\{0.01,0.1\}$ and 300 repetitions. A count comparator stops immediately at the $(n+1)$st distinct label, otherwise reaching its independent Poisson cap. Reporting this stopping rule prevents a batch-checkpoint comparison from overstating the value of the fixed-budget construction.')
    fig('dependence');fig('stream')
    section('Full-program and numerical verification')
    p(f'We executed {s["public_api_runs"]:,} complete public-reference calls on actual independent Poisson category counts, including {len(json.loads((out/"natural_public_api.json").read_text()))} calls generated from natural empirical probabilities. The resulting decisions and integer lower bounds agree with the count-branch specification. An independent finite occupancy generating-function convolution checks unsaturated small cases. All {s["audit_assertions"]:,} independent algebraic assertions pass, including exact coefficients, branch boundaries, large-degree weights, final decisions, and intentional error controls. Large-parameter fixtures exercise the complete prescribed polynomial degrees and all scales with adaptive interval precision; these are deterministic formula checks, not simulations of $10^{{444}}$ physical observations.')
    p(r'An additional noninteger-output fixture requires precision escalation from 1608 to 3216 bits; its 443-digit rounded lower bound matches a separately evaluated 4096-bit polynomial formula. This checks rounding after the polynomial, rather than only an easy clipped endpoint. The independent interval formulas share the FLINT arithmetic stack, so this is not an independent proof-assistant certificate.')
    p(r'Runtime measurements isolate count generation from certified public-reference decisions. Every $(n,C_0)$ case runs in a separate process with one warm-up and ten timed repetitions; process peak RSS is retained. Compressed simulation time is reported as computation only. It must not be interpreted as the cost of obtaining the nominal number of observations in nature.')
    fig('runtime')
    section('Reproducibility and limitations')
    p(r'The artifact contains prospective design files, all primary trial occupancies, per-seed results, exact target audits, source snapshots, environment versions, source-data provenance, figures, code and LaTeX. Run \texttt{python -m c1025.run --phase all}, then \texttt{python -m c1025.supplement}, \texttt{python -m c1025.numeric\_extra}, and \texttt{python -m c1025.report}. Existing completed stages are preserved. The new raw-output audit recomputes every primary decision, confidence bound and lower-bound failure from saved trials. A local frozen protocol is not an external preregistration.')
    p(r'The tested biological collections are diverse but small, historical, and nonrandomly selected; their reference supports omit undiscovered species. No new field survey or external full-population inventory was obtained. The practical grid cannot distinguish 1024 from 1025 at the astronomical worst-case boundary. The proof is not certified by a proof-assistant kernel or independent peer review. The evidence supports a precise constant-refinement evaluation with explicit negative results; publication novelty and fit to a specific conference remain separate questions. No conference template or acceptance claim is implied.')
    return '\n'.join(body)


def generate_tex(out,body):
    proof=(ROOT/'artifacts/c0_1025_paper/c0_1025_note.tex').read_text()
    macros=proof[proof.index('\\newtheorem'):proof.index('\\title')]
    pre=r'''\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,amsthm,mathtools,microtype}
\usepackage{graphicx,booktabs,enumitem,textcomp}
\usepackage[hidelinks]{hyperref}
\setlist[enumerate]{leftmargin=*,itemsep=2pt,topsep=4pt}
'''+macros+r'''\begin{document}
\begin{center}{\Large\bfseries Evaluating an Explicit Constant for Support-Size Testing\par}\vspace{.4em}
{\normalsize Reproducible experiments for $C_0=1025$, with proof appendix}\end{center}
'''
    appendix=proof[proof.index('\\section{Construction'):proof.index('\\begin{thebibliography}')]
    refs=r'''\clearpage
\begin{thebibliography}{6}
\bibitem{wang} Qisheng Wang. \emph{Optimal Support Size Testing}. Supplied manuscript, arXiv:2610.18723v1, 2026.
\bibitem{dlmf} NIST Digital Library of Mathematical Functions. Chapter 18, Sections 18.5 and 18.16. \url{https://dlmf.nist.gov/18.5}.
\bibitem{vegan} vegan developers. BCI and Oribatid mite data documentation; original sources Condit et al. (2002) and Borcard et al. (1992, 1994). \url{https://vegandevs.github.io/vegan/reference/BCI.html}; \url{https://vegandevs.github.io/vegan/reference/mite.html}.
\bibitem{inext} T. C. Hsieh, K. H. Ma, and Anne Chao. iNEXT package data documentation: bird and spider abundance data, citing Chao et al. (2015), Ellison et al. (2010), and Sackett et al. (2011). \url{https://annechao.r-universe.dev/iNEXT/doc/manual.html}.
\bibitem{sads} Paulo I. Prado, Murilo Dantas Miranda, and Andre Chalom. sads package data documentation: Rothamsted moths and Okland land snails. \url{https://pilaboratory.r-universe.dev/sads/doc/manual.html}.
\bibitem{fish} Derek H. Ogle. InchLake1, FSAdata package. \url{https://fishr-core-team.github.io/FSAdata/reference/InchLake1.html}.
\end{thebibliography}
\end{document}
'''
    (out/'experiment_body.tex').write_text(body)
    (out/'proof_appendix.tex').write_text(appendix)
    (out/'experiment_paper.tex').write_text(pre+r'\input{experiment_body.tex}'+'\n'+r'\clearpage\appendix\input{proof_appendix.tex}'+'\n'+refs)
    (out/'experiments_only.tex').write_text(pre+r'\input{experiment_body.tex}'+'\n'+refs)


def chinese(out,summary,figures):
    s=summary
    def link(file,label):return '<a href="'+file+'">'+label+'</a>'
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>C0=1025 实验报告</title><style>body{font:17px/1.8 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#172638}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}img{max-width:100%}.card{background:#eef5fa;padding:20px;border-radius:8px}details{margin:25px 0}a{color:#0072b2}code{font-size:14px}</style><body>']
    parts+=['<h1>C₀ = 1025：完整实验与自然数据可行性</h1>',
      '<p>'+link('experiment_paper.pdf','实验论文 PDF（含完整证明附录）')+' · '+link('experiments_only.pdf','仅实验章节 PDF')+' · '+link('experiment_paper.tex','LaTeX 主文件')+' · '+link('summary.json','机器可读结果')+'</p>',
      f'<div class="card">在 {s["synthetic_base_cells"]} 个合成配置和 {s["natural_base_cells"]} 个自然经验分布配置上，C₀=1025 的承诺检验未观察到错误。主设计共 {s["primary_independent_poisson_paths"]:,} 条独立模拟路径、{s["primary_method_trial_evaluations"]:,} 次配对方法评估；二者不混为独立试验次数。所有现实规模都进入计数分支。</div>',
      '<h2>首先看成功率</h2><p>Yes 表示支持数不超过 n；No 表示 n 个最高概率类别外的质量至少为 ε。Gap 不定义二元错误率。ε 是分离距离，不是错误概率。表中的上界是按整套测试网格校正后的单侧 95% 置信上界；零观察错误不等于真实错误率为零。</p>',
      '<table><tr><th>数据</th><th>方法</th><th>错误/承诺试验</th><th>最差错误率上界</th></tr>']
    for r in s['risk_summary']:
        parts.append(f'<tr><td>{r["suite"]}</td><td>{html.escape(LABELS[r["method"]])}</td><td>{r["errors"]:,}/{r["promised_trials"]:,}</td><td>{100*r["worst_simultaneous_upper"]:.3f}%</td></tr>')
    parts+=['</table><h2>必须同时报告的限制</h2>',
      f'<p>新常数比原来的理论预算小约 6.55 万倍，但仍比本实验的 99% 计数基线多用 {s["new_to_count_ratio_min"]:.1f}–{s["new_to_count_ratio_max"]:.1f} 倍样本。因此，实验支持常数改进和实现正确性，不能宣称现实采样效率超过这个基线。C=1024 在现实网格上同样通过，最小整数结论不能靠这组实验区分。</p>',
      f'<p>十个真实总体来自七类生物，均为公开的有限丰度记录。{s["natural_budget_cells"]} 个配置中只有 {s["natural_budget_sufficient_cells"]} 个原始记录数达到 C₀=1025 的平均预算，且都来自土壤螨数据。其他配置的结果是从经验分布生成新 iid 样本的算法基准，不能说已经验证了未知自然总体。已有数据中看见超过 n 个物种可以直接拒绝；看见不超过 n 个物种且样本不足则不能认证接受。</p>',
      f'<p>保持边际物种概率但人为引入相关性后，最坏错误率分别达到 {100*s["duplication_worst_error"]:.3f}%（重复观测）和 {100*s["group_stress_worst_error"]:.3f}%（真实分组内成块采样）。这说明 iid 假设实质重要，并非仅在报告里加一句免责声明。</p>',
      '<h2>每张图在说明什么</h2>']
    for f in figures:
        parts.append('<details'+(' open' if f['name'] in ['risk','natural_feasibility'] else '')+'><summary>'+html.escape(f['description_zh'])+'</summary><img src="figures/'+f['name']+'.png"><p>'+html.escape(f['caption'])+'</p></details>')
    parts += [f'<h2>核验与复现</h2><p>完整公开接口执行 {s["public_api_runs"]:,} 次；巨大参数、多项式系数和完整输出的独立代数检查通过 {s["audit_assertions"]:,} 项。原始逐次 D 值、四个种子块、所有失败结果和 Gap 单元均保留。自然数据标签与有效支持整数另用原始整数计数复核。数值核验不能替代对所有分布的数学证明。</p>',
      '<p>'+link('result_validation.json','原始结果一致性审计')+' · '+link('protocol.json','冻结实验设计及日志修正说明')+' · '+link('data_sources.json','数据来源与许可')+' · '+link('natural_feasibility.csv','真实数据预算逐项表')+' · '+link('REPRODUCE.txt','复现说明')+'</p>',
      '<p>当前材料是可审阅、可复现的实验论文稿；未指定具体会议模板，也不能由实验规模或排版推断达到某会议的录用标准。</p></body></html>']
    (out/'report.html').write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results_c1025');args=parser.parse_args();out=args.out.resolve()
    keys=['synthetic','natural','curves','natural_feasibility','dependence','group_holdout','natural_group_stress','stream','runtime']
    frames={k:read(out,k) for k in keys};audit=validate(out,frames);summary,cost=analyze(out,frames,audit)
    enrich_provenance(out)
    figs=figures(out,frames,summary,cost);body=write_paper(out,frames,summary,figs,audit);generate_tex(out,body);chinese(out,summary,figs)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['risk_summary','limitations']},indent=2))

if __name__=='__main__':main()
