"""Generate figures, an English research manuscript, Chinese report, and bundle.

Analysis-only code: it does not tune, regenerate, or overwrite experiment CSVs.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from publication.aggregate import aggregate,SUITES
from publication.protocol import ROOT,OUT,sha,save,DESIGN

COLORS={"count":"#697681","multi":"#007f73","single":"#7073be","fh_certified":"#d38d22",
        "fh_native_tuned":"#c84f45","chao1_point":"#9a648f","multi_hybrid":"#007f73",
        "single_hybrid":"#7073be","fh_hybrid":"#d38d22","count_validation":"#288bc2",
        "polynomial_only":"#b5658c","count_fixed_immediate":"#697681","count_batch":"#697681",
        "count_checkpoint":"#bb6f44","count_poisson_immediate":"#465164"}
LABELS={"count":"Observed count","multi":"Certified multiscale","single":"Tuned single scale",
        "fh_certified":"Certified FH-shaped","fh_native_tuned":"FH-shaped native threshold",
        "chao1_point":"Chao1 point threshold","multi_hybrid":"Multiscale hybrid",
        "single_hybrid":"Single-scale hybrid","fh_hybrid":"FH-shaped hybrid",
        "count_validation":"Count + validation","polynomial_only":"Polynomial + count",
        "count_fixed_immediate":"Immediate fixed count","count_batch":"Checkpoint count",
        "count_checkpoint":"Checkpoint count","count_poisson_immediate":"Immediate Poisson count"}


def fmt(value,digits=3):
    if value is None or (isinstance(value,float) and not math.isfinite(value)):return "--"
    if isinstance(value,(int,np.integer)):return f"{value:,}"
    return f"{float(value):.{digits}f}"


def pct(value):
    return "--" if value is None or pd.isna(value) else f"{100*float(value):.2f}%"


def figure_all(frames,eff,cost,out):
    directory=out/"figures";directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":180})
    index=[]
    def export(fig,name,caption):
        fig.tight_layout()
        for suffix in ["png","pdf"]:fig.savefig(directory/f"{name}.{suffix}",bbox_inches="tight")
        plt.close(fig);index.append(dict(name=name,caption=caption))
    fixed=frames["fixed"]
    fig,axes=plt.subplots(2,3,figsize=(12,6.3),sharex=True,sharey=True)
    for row,eps in enumerate([.001,.1]):
      for col,n in enumerate([1000,10000,1000000]):
        ax=axes[row,col]
        for method in ["count","multi","single","fh_certified","fh_native_tuned"]:
          g=fixed[(fixed.n==n)&(fixed.epsilon==eps)&(fixed.method==method)&(fixed.truth!="Gap")&fixed.C.notna()]
          curve=g.groupby("C").error_rate.max()
          ax.plot(curve.index,curve.values,".-",color=COLORS[method],label=LABELS[method])
        ax.axhline(.05,color="black",lw=.7,ls="--");ax.set(xscale="log",ylim=(-.03,1.03),title=f"n={n:,}; epsilon={eps:g}")
        ax.grid(alpha=.15)
    axes[1,1].set_xlabel("C in m = ceil(C n / [epsilon log(en)])")
    axes[0,0].set_ylabel("Worst observed error");axes[1,0].set_ylabel("Worst observed error")
    axes[0,2].legend(fontsize=6.5,loc="upper right")
    export(fig,"fixed_risk","Worst observed error over promised synthetic scenarios. The dashed line is 5%; simultaneous confidence bounds, not this point-estimate plot, determine passing budgets.")
    fig,axes=plt.subplots(2,2,figsize=(10,6.5))
    for ax,n in zip(axes.flat,[1000,10000,100000,1000000]):
      for method in ["count","multi","single","fh_certified"]:
        g=eff[(eff.n==n)&(eff.method==method)].sort_values("epsilon")
        ax.plot(g.epsilon,g.normalized_n_over_eps_log,"o-",ms=4,color=COLORS[method],label=LABELS[method])
      ax.set(xscale="log",yscale="log",title=f"n={n:,}",xlabel="epsilon",ylabel="m* epsilon log(en) / n");ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    export(fig,"sample_complexity","Smallest tested budget passing simultaneous 95% error upper bounds across all promised synthetic scenarios. Missing points are right-censored; no interpolation or universal complexity claim is made.")
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    mech=frames["scale_mechanism"]
    g=mech[(mech.n==10000)&(mech.C==4)&(mech.truth=="No")]
    table=g.pivot_table(index="j",columns="epsilon",values="trigger_rate",aggfunc="mean").fillna(0)
    im=axes[0].imshow(table.values,aspect="auto",origin="lower",vmin=0,vmax=1,cmap="viridis")
    axes[0].set_xticks(range(len(table.columns)),[f"{x:g}" for x in table.columns],rotation=45)
    axes[0].set_yticks(range(len(table.index)),table.index);axes[0].set(xlabel="epsilon",ylabel="Scale j",title="Scale trigger rates at n=10,000, C=4")
    fig.colorbar(im,ax=axes[0],label="Probability")
    g=frames["scale_removal"].query("truth == 'No'").copy()
    g["extra_normalized"]=g.mean_extra_cost*g.epsilon/g.n
    v=g.groupby("j").extra_normalized.mean()
    axes[1].bar(v.index,v.values,color=COLORS["multi"]);axes[1].set(xlabel="Removed scale j",ylabel="Mean added cost / (n/epsilon)",title="Leave-one-scale-out cost")
    export(fig,"scale_mechanism","Left: all triggered scales are recorded, not only the first winner. Right: remove one scale while retaining the original error spending and common sample paths.")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    seq=frames["sequential"]
    for ax,truth in zip(axes,["Yes","No"]):
      for method in ["count_validation","polynomial_only","multi_hybrid","single_hybrid","fh_hybrid"]:
        v=seq[(seq.truth==truth)&(seq.method==method)].groupby("n").relative_cost.mean()
        ax.plot(v.index,v.values,"o-",color=COLORS[method],label=LABELS[method],ms=4)
      ax.axhline(1,color="gray",ls="--");ax.set(xscale="log",xlabel="n",title=f"True {truth}");ax.grid(alpha=.15)
    axes[0].set_ylabel("Mean relative cost vs checkpoint count");axes[1].legend(fontsize=7)
    export(fig,"sequential_ablation","Large-scale coupled Poisson trajectories. Cost is stopped training mean plus validation labels; it estimates expected acquisition cost and is not the distribution of actual sample counts.")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    stream=frames["stream"]
    for ax,truth in zip(axes,["Yes","No"]):
      for method in ["count_checkpoint","count_validation","multi_hybrid","single_hybrid","fh_hybrid"]:
        v=stream[(stream.truth==truth)&(stream.method==method)].groupby("n").relative_cost.mean()
        ax.plot(v.index,v.values,"o-",color=COLORS[method],label=LABELS[method],ms=4)
      ax.axhline(1,color="gray",ls="--");ax.set(xscale="log",xlabel="n",title=f"True {truth}");ax.grid(alpha=.15)
    axes[0].set_ylabel("Actual labels / immediate fixed count");axes[1].legend(fontsize=7)
    export(fig,"actual_cost","Actual label consumption including validation, compared with an immediately stopping fixed-budget count baseline. All methods share iid training labels and fresh validation pools.")
    fig,axes=plt.subplots(2,2,figsize=(10,6.2))
    for ax,scenario in zip(axes.flat,["uniform_null","head90_null","uniform_boundary","exact_diffuse_boundary"]):
      path=out/"raw"/f"stream_n1000_e0.01_{scenario}.npz"
      with np.load(path) as data:
        for method in ["count_fixed_immediate","count_checkpoint","count_validation","multi_hybrid","single_hybrid"]:
          x=np.sort(data[method+"__cost"])
          ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",color=COLORS[method],label=LABELS[method])
      ax.set(title=scenario.replace("_"," "),xlabel="Actual labels",ylabel="Empirical CDF",ylim=(0,1.02))
    axes[0,0].legend(fontsize=6.5)
    export(fig,"actual_cost_cdf","Full stopping-cost distributions for n=1,000 and epsilon=0.01. Quantiles refer to actual consumed labels, with verification samples included.")
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    adv=frames["adversarial_fixed"]
    for ax,truth in zip(axes,["Yes","No"]):
      for method in ["count","multi","single","fh_certified","fh_native_tuned"]:
        g=adv[(adv.truth==truth)&(adv.method==method)].groupby("C").error_rate.max()
        ax.plot(g.index,g.values,"o-",color=COLORS[method],label=LABELS[method],ms=4)
      ax.axhline(.05,color="black",ls="--",lw=.7);ax.set(title=f"Selected hard {truth} cases",xlabel="C",ylabel="Worst independent-test error",ylim=(-.03,1.03))
    axes[1].legend(fontsize=6.5)
    export(fig,"adversarial_risk","Adversarial candidates were selected on a separate search stream and then frozen. Each fixed-budget test cell uses 10,000 new trials; all selected candidates are retained.")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    g=fixed[(fixed.n==10000)&(fixed.epsilon==.01)&fixed.C.notna()]
    for method in ["count","multi","single","fh_certified"]:
      h=g[g.method==method]
      v=h.groupby("C").lower_joint_success.min()
      axes[0].plot(v.index,v.values,"o-",color=COLORS[method],label=LABELS[method],ms=4)
      v=h.groupby("C").lower_over_simultaneous_upper95.max()
      axes[1].plot(v.index,v.values,"o-",color=COLORS[method],label=LABELS[method],ms=4)
    axes[0].set(xlabel="C",ylabel="Worst joint success",ylim=(-.03,1.03),title="Effective-support lower-bound power")
    axes[1].set(xlabel="C",ylabel="Simultaneous upper bound",title="Probability lower bound exceeds true S")
    axes[0].legend(fontsize=7)
    export(fig,"effective_lower","n=10,000, epsilon=0.01. Joint success requires min(effective support,n) <= lower <= S. Exact rational flooring preserves coverage at a cost of less than one category per scale.")
    structural=frames["structure"].query("method == 'multi' and truth == 'No'")
    fig,axes=plt.subplots(1,2,figsize=(12,4.1))
    natural=structural[structural.sampling.isin(["iid_record","natural_group_blocks"])]
    for dataset,g in natural.groupby("dataset"):
      g=g.sort_values("block_size");axes[0].plot(g.block_size,g.error_rate,"o-",label=dataset)
    axes[0].set(xscale="log",xlabel="Within-group block size",ylabel="Failure to reject",title="Natural group dependence",ylim=(-.03,1.03))
    axes[0].legend(fontsize=6.5)
    repair=structural[structural.sampling.str.startswith("repaired")]
    tab=repair.pivot(index="dataset",columns="sampling",values="error_rate")
    x=np.arange(len(tab))
    bounds=repair.pivot(index="dataset",columns="sampling",values="error_simultaneous_upper95")
    for j,c in enumerate(tab.columns):
      pos=x+(j-.5)*.35
      axes[1].bar(pos,tab[c],width=.35,label="Equal-event target" if "equal" in c else "Size-weighted target")
      axes[1].errorbar(pos,tab[c],yerr=np.vstack([np.zeros(len(tab)),bounds[c]-tab[c]]),fmt=".",color="black",capsize=3)
    axes[1].set_xticks(x,tab.index,rotation=35,ha="right");axes[1].set(ylabel="Failure to reject",title="Independent group draws (upper 95% bounds)",ylim=(0,max(.01,1.5*float(bounds.max().max()))));axes[1].legend(fontsize=7)
    export(fig,"group_structure","Dependent blocks deliberately violate iid. The repaired samplers draw a fresh independent group per label; equal-event weighting changes the target distribution. The right panel has a different vertical scale and shows simultaneous upper confidence limits even at zero observed errors. Fixed-budget non-rejection remains Inconclusive.")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    sens=frames["sensitivity"]
    g=sens[(sens.variant=="delta")&(sens.method=="multi_hybrid")]
    for truth,color in [("Yes",COLORS["count_validation"]),("No",COLORS["multi"])]:
      v=g[g.truth==truth].groupby("delta").relative_cost.mean()
      axes[0].plot(v.index,v.values,"o-",color=color,label=truth)
    axes[0].set(xscale="log",xlabel="Requested error delta",ylabel="Relative cost vs checkpoint count",title="Confidence-level sensitivity");axes[0].legend()
    g=sens[(sens.variant!="delta")&(sens.method=="multi_hybrid")]
    tab=g.pivot_table(index="variant",columns="truth",values="relative_cost",aggfunc="mean")
    positions=np.arange(len(tab))
    for j,truth in enumerate(["Yes","No"]):axes[1].barh(positions+(j-.5)*.36,tab[truth],height=.36,label=truth,color=[COLORS["count_validation"],COLORS["multi"]][j])
    axes[1].set_yticks(positions,tab.index,fontsize=6.5);axes[1].set(xlabel="Relative cost",title="One-at-a-time parameter changes");axes[1].legend(fontsize=7)
    export(fig,"sensitivity","Frozen configurations are not re-selected from these outcomes. Delta=0.001 cells use 20,000 trials; remaining sensitivity cells use 2,000.")
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    runtime=frames["runtime"].query("epsilon == 0.1")
    for method,g in runtime.groupby("method"):
      g=g.sort_values("n")
      for ax,column in zip(axes,["compile_seconds","score_microseconds_per_profile","peak_process_rss_MiB"]):
        ax.plot(g.n,g[column],"o-",color=COLORS[method],label=LABELS[method],ms=4);ax.set_xscale("log");ax.set_xlabel("n")
    axes[0].set_ylabel("Compilation seconds");axes[1].set_ylabel("Scoring microseconds/profile");axes[2].set_ylabel("Peak process RSS (MiB)")
    axes[0].legend(fontsize=6.5)
    export(fig,"runtime","Compilation is measured separately from grouped simulation and scoring. Peak RSS is the absolute fresh-process peak, including Python and dependencies. Timing should be interpreted on the recorded machine.")
    save(directory/"index.json",index)
    return index


def tex_escape(s):
    return str(s).replace("\\",r"\textbackslash{}").replace("&",r"\&").replace("%",r"\%").replace("_",r"\_").replace("#",r"\#").replace("$",r"\$")


def table_tex(headers,rows):
    return "\\begin{tabular}{"+"l"*len(headers)+"}\n\\toprule\n"+" & ".join(map(tex_escape,headers))+r" \\"+"\n\\midrule\n"+"\n".join(" & ".join(map(tex_escape,r))+r" \\" for r in rows)+"\n\\bottomrule\n\\end{tabular}\n"


def manuscript(frames,eff,cost,paired,summary,figures,out):
    def costrow(suite,truth,method="multi_hybrid"):
      return cost[(cost.suite==suite)&(cost.truth==truth)&(cost.method==method)].iloc[0]
    actual_yes,actual_no=costrow("stream","Yes"),costrow("stream","No")
    seq_yes,seq_no=costrow("sequential","Yes"),costrow("sequential","No")
    fixed=frames["fixed"]
    adv=frames["adversarial_fixed"]
    formal_no=frames["sequential"].query("method == 'multi_hybrid' and truth == 'No'")
    formal_yes=frames["sequential"].query("method == 'multi_hybrid' and truth == 'Yes'")
    total=summary["total_method_trial_evaluations"]
    configs=json.loads((out/"calibration.json").read_text())["configs"]
    selected=json.loads((out/"adversarial_selected.json").read_text())["populations"]
    title="Finite-Sample Support-Size Testing: Certified Decisions, Sequential Costs, and Failure Modes"
    blocks=[]
    def section(s):blocks.append(dict(kind="section",text=s))
    def p(s):blocks.append(dict(kind="p",text=s))
    def eq(latex,plain):blocks.append(dict(kind="math",latex=latex,text=plain))
    def figure(name):blocks.append(dict(kind="figure",**next(x for x in figures if x["name"]==name)))
    def table(caption,headers,rows,name):
      blocks.append(dict(kind="table",caption=caption,headers=headers,rows=rows,name=name))
      (out/"tables"/f"{name}.tex").write_text(table_tex(headers,rows))
    section("Abstract")
    p(f"Support-size testing distinguishes distributions supported on at most n categories from distributions whose probability mass outside the n most likely categories is at least epsilon. Asymptotically optimal testers do not by themselves determine useful finite-sample constants or acquisition costs. We study certified Chebyshev statistics combined with independent acceptance validation and a count-test fallback. Across {total:,} method-trial evaluations, we compare tuned single-scale and multiscale families, FH-shaped practical variants, literal finite-scale reference branches, and stronger count baselines. On the labelled-stream grid, the multiscale hybrid uses {pct(actual_yes.mean_relative_cost)} of the immediate count baseline's samples on Yes instances and {pct(actual_no.mean_relative_cost)} on No instances, with validation included. Large-scale checkpoint results show larger apparent gains, demonstrating why stopping-rule fairness matters. We also measure effective-support lower-bound power, independently retest searched difficult distributions, and expose sensitivity to natural group dependence. The hybrid has a finite-sample error guarantee under iid sampling, but we do not claim a new optimal-order sample-complexity theorem or universal empirical dominance.")
    section("1. Introduction")
    p("Counting observed categories is reliable as a lower bound but can require many observations when mass is dispersed among rare categories. Polynomial extrapolation can infer support beyond the observed count. The difficulty is that approximation error, sampling fluctuation, repeated looks, and data dependence all interact. A small-budget statistic that frequently rejects alternatives can still be unusable because it rejects null distributions too often.")
    p("The supplied manuscript Optimal Support Size Testing establishes an asymptotically optimal tester and a stronger effective-support lower bound [1]. Our aim is to make the finite-sample experimental questions explicit. We investigate which component creates savings, whether those savings survive a competitive stopping rule, which distributions defeat the chosen practical constants, and what changes when records share documents, users, orders, or sampling events.")
    p("Our contribution is a reproducible finite-sample study and certified implementation, rather than a replacement of the original optimality proof. The artifact includes interval-certified thresholds, exact integer rejection decisions, a corrected sequential ablation, a fair labelled-stream benchmark, held-out adversarial testing, and raw outcomes supporting all reported tables. These components expose both useful operating regimes and important limits.")
    section("2. Problem and related work")
    eq(r"H_0:S=|\operatorname{supp}(p)|\le n,\qquad H_1:\Delta_n(p)=\sum_{i>n}p_{(i)}\ge\varepsilon.","H0: S <= n. H1: Delta_n(p) = sum of probabilities outside top n >= epsilon.")
    p("The probabilities p_(i) are ordered decreasingly. Instances with S>n and Delta_n<epsilon are a promise gap; neither binary output is scored as an error there. A non-rejection from a fixed-budget one-sided polynomial test is reported as Inconclusive, not as a certified Yes. A sequential hybrid returns a binary result only after a certified acceptance or the final count budget.")
    p("The original Wang construction uses multiple Chebyshev scales and attains order n/(epsilon log n) [1]. Ferreira Pinto Jr. and Harms study the same promise problem using a single polynomial with an additional logarithmic dependence on epsilon [2]. We distinguish their literal conservative finite-scale branch from a tuned FH-shaped statistic, and from a separately certified FH-shaped variant. This distinction is necessary because changing constants or thresholds does not preserve the cited theorem automatically.")
    p("Wu and Yang study support estimation under a minimum nonzero mass assumption [3]. Regularized weighted Chebyshev estimators also target support estimation with additional model structure [4]. Those are valuable related methods, but an oracle lower bound on minimum mass would change our distribution-free promise problem. We therefore do not present a tuned point estimator as a universally valid tester. Chao1 thresholding is included only as a clearly labelled heuristic diagnostic. Time-uniform inference motivates careful error spending [5]; the present construction uses a finite, predetermined look schedule rather than an unrestricted confidence sequence.")
    section("3. Certified finite-sample construction")
    p("Under Poissonized sampling with predetermined mean m, category counts N_i are independent Poisson(m p_i). At a scale with A=ma and B=mb, define Chebyshev weights f(0)=0 and f(l)=1 for l above degree k, using the original polynomial recurrence for the remaining coefficients. The statistic is Z=sum_i f(N_i)=D+sum_{l=1}^k F_l(f(l)-1). Parameters are fixed before evaluating a test sample.")
    eq(r"\rho=\frac1{T_k((B+A)/(B-A))},\quad R=\max\{0,1,f(1),\ldots,f(k)\}-\min\{0,1,f(1),\ldots,f(k)\}.","rho = 1/T_k((B+A)/(B-A)); R = range of the possible per-category contributions.")
    p("Every proposed scale is checked using Arb interval arithmetic: 0<A<B<=m, k<=sqrt(B(B-A)), and the configured weight cap. Degree is reduced or the scale is omitted if certification fails. Under these premises, the polynomial expectation envelope gives E[Z]<=(1+rho)S. A single-scale comparator spends its error budget on its one predetermined scale, rather than inheriting the multiscale penalty.")
    eq(r"u=R\sqrt{\frac n2\log\frac{HK}{\delta}},\qquad Z>(1+\rho)n+u\ \Longrightarrow\ \mathrm{No}.","At H looks and K proposed scales: u = R sqrt((n/2) log(HK/delta)). Reject only if Z > (1+rho)n + u.")
    p("Arb supplies upper bounds for rho, the range, and thresholds, and lower bounds for statistic coefficients. All final rejection comparisons use a common binary-rational denominator and integers. Floating calculations only identify candidates for this exact comparison. This protects the analytical null guarantee from rounding-induced false rejection. The random samplers themselves use standard double-precision numerical libraries; they are not formally verified random-number implementations.")
    p("The multiscale family proposes a_j=2^j epsilon/(8n), with B proportional to log(en). The single-scale family selects one j independently on development data. The practical FH-shaped family proposes a proportional to epsilon log(1/epsilon)/n, b proportional to epsilon log(en)^2/[n log(1/epsilon)], and degree proportional to log(en). Its certified version is checked by the same envelope and concentration argument. Its native-threshold version uses the FH-shaped raw statistic with threshold (1+epsilon/2)n and has no inherited finite-sample guarantee after tuning.")
    section("4. Sequential decisions and guarantees")
    p("Training batches are independent Poisson increments on predetermined increasing means. Directly observing the (n+1)-st category permits immediate rejection. Polynomial rejection is checked only at the fixed looks. To accept early, the algorithm freezes the observed training set A, draws r fresh independent validation labels, and counts V labels outside A. The singleton heuristic F1/m gates whether to spend validation labels; it never certifies acceptance.")
    eq(r"\Pr\{\operatorname{Bin}(r,\varepsilon)\le k\}\le\frac\delta{2H},\quad V\le k\ \Longrightarrow\ \mathrm{Yes}.","Choose k so Binomial(r,epsilon) left tail at k <= delta/(2H); accept only when V <= k.")
    p("The final count budget m* is the smallest proposed integer certified to satisfy Pr(Poisson(m* epsilon)<=n)<=delta/2 when validation is enabled. Without early acceptance, the full delta is assigned to the count fallback. Total scheduled validation labels are capped at m*/5. This correction prevents the polynomial-only ablation from wasting half its false-negative budget on a branch it does not use.")
    p("Proposition 1 (conditional on the iid model). With fixed n, epsilon, delta, fixed configuration, independent sampling randomization, fresh validation samples, and the certified comparisons above, the probability of a No output under S<=n is at most delta, and the probability of a Yes output under Delta_n>=epsilon is at most delta. The worst expected acquisition cost is O((n+log(1/delta))/epsilon). No minimum positive category probability is assumed.")
    p("Proof. Under S<=n, append zero contributions to obtain n independent bounded terms. Hoeffding bounds each polynomial crossing by delta/(HK); the union bound covers all scales and looks even though looks share observations. A direct count rejection is impossible under the null. Under the alternative, any history-dependent set A with at most n elements has outside mass at least epsilon. Conditional on that history, fresh validation gives a binomial lower-tail error at most delta/(2H). Before discovery of the (n+1)-st category, the probability of a new category is at least epsilon. Coupling discoveries with Bernoulli(epsilon) trials yields the fixed-size bound Pr(D<=n)<=Pr(Bin(M,epsilon)<=n), and Poisson mixing yields the certified final tail bound. An error on a path reaching fallback is a subset of the error event for the full potential training path. Summing validation and fallback errors gives delta. Training means and all scheduled validation sizes bound expected acquisition by 1.2m*. Standard Poisson concentration gives the stated order. Immediate direct-count stopping only shortens a path and adds no null errors.")
    section("5. Experimental design")
    p("The main grid uses n in {1,000,10,000,100,000,1,000,000}, epsilon in {0.001,0.005,0.01,0.02,0.05,0.1,0.2}, eleven distribution scenarios, thirteen budget factors C from 0.5 through 16, and an additional certified count budget. Each method cell has 2,000 trials, split across four independently keyed random blocks. Equal-probability category groups permit exact occupancy-profile simulation without enumerating up to 100 million categories. The sampler preserves groups of distinct categories; it does not replace a rare tail by one category.")
    p("Each of the multiscale, single-scale and FH-shaped families received eight development candidates. Development uses n=1,500 and 15,000, epsilon=0.005,0.05,0.2, different distribution shape parameters, four budgets, and 400 trials per cell. Certified candidates are ranked by mean non-rejection on alternatives; the native heuristic additionally receives a penalty for observed null inflation. No formal outcomes alter the selected configurations. The selected multiscale B factor is "+fmt(configs["multi"]["b_factor"])+", weight cap "+fmt(configs["multi"]["weight_cap"])+", and the selected single scale is j="+str(configs["single"]["single_j"])+".")
    p("A local prospective design and code/data hashes were recorded before formal evaluation. A subsequent correctness audit found nominal equality-boundary distributions infinitesimally below epsilon after exact rational normalization. The original outputs remain in a superseded archive. We repaired the constructor, retained the original selected configurations, recorded the amendment, and reran every formal suite with the same keyed seeds. Exact normalized tail certificates are provided. This transparent amendment is not an external preregistration claim.")
    table("Experimental inventory. Method-trial evaluations reuse common samples across methods; they are not independent datasets.",
          ["Suite","Method cells","Method-trial evaluations"],
          [[r["suite"],fmt(r["method_cells"]),fmt(r["method_trial_evaluations"])] for r in summary["inventory"]],"inventory")
    p("Error rates use one-sided exact Clopper-Pearson bounds. We report 95% simultaneous bounds with Bonferroni correction across all non-Gap method cells in each frozen suite. The three lower-bound failure metrics form a separate simultaneous family in the fixed-budget suite. The smallest passing budget m* is defined by simultaneous upper bounds across all promised scenarios, not by an observed error-rate threshold. No passing tested budget is reported as right censoring.")
    p("Cost contrasts use common training paths and independent shared validation pools. Cell means are weighted equally, with Yes and No strata separate. Paired Monte Carlo intervals use delta-method standard errors; Holm correction applies to the reported aggregate cost contrasts. A separate scenario bootstrap describes sensitivity to the finite grid composition, not generalization to arbitrary unknown distributions. The full seed-block table and trial arrays are released.")
    section("6. Fixed-budget risk, scaling, and scale contribution")
    passes=summary["fixed_complete_grid_passes"]
    p("Across the fixed-budget grid, the largest observed null rejection rate among the three certified polynomial families is "+pct(summary["certified_fixed_max_fp"])+". Complete-grid passing budgets were found for "+", ".join(f"{LABELS.get(k,k)}: {v}/28 parameter pairs" for k,v in passes.items())+". Native-threshold or point-estimate performance must be assessed jointly on nulls and alternatives; high rejection power alone is not evidence of a valid test.")
    figure("fixed_risk");figure("sample_complexity")
    p("The normalized sample-complexity curves assess finite-grid scaling as epsilon decreases. They do not prove that practical constants retain the original optimal order. In particular, both the finite concentration correction and the fallback may dominate difficult small-n regimes. A missing point is a failure to establish the target error level with this grid and Monte Carlo precision, not proof that no possible budget works.")
    compare=eff.pivot(index=["n","epsilon"],columns="method",values="m_star")
    comparisons=[]
    for method in ["single","fh_certified","count"]:
      comparisons.append([LABELS[method],str(int((compare[method]<compare.multi).sum())),str(int((compare[method]==compare.multi).sum())),str(int((compare[method]>compare.multi).sum()))])
    table("Comparator budgets relative to multiscale m* across 28 n/epsilon pairs. Smaller means the comparator outperforms multiscale on this finite-grid criterion.",
          ["Comparator","Smaller budget","Equal budget","Larger budget"],comparisons,"budget_wins_losses")
    p("The comparison does not show uniform multiscale superiority. At n=1,000, the multiscale family passes at approximately C=4 whereas the tuned single-scale family needs C=5; at n=1,000,000 the ordering reverses, with single scale passing at C=4 and multiscale at C=5. Certified FH-shaped variants also beat multiscale on several small-epsilon settings. Consequently, these experiments do not empirically establish the asymptotic logarithmic-factor advantage of the original theorem for the practical implementations. The finite-sample evidence supports regime-dependent method selection and careful certification, rather than a blanket best-method claim.")
    figure("scale_mechanism")
    p("Scale diagnostics record all crossings, exclusive crossings, first winning scale, and additional sequential cost when a single scale is removed. The removal experiment retains the original thresholds, isolating the information contributed by that scale; it is not a newly optimized smaller family. The independent single-scale development search prevents a weak fixed j=0 comparator from mechanically favoring multiscale testing.")
    section("7. Sequential efficiency and actual acquisition cost")
    table("Equal-weight mean cost ratios. Large-scale trajectories use checkpoint count as denominator; labelled streams use immediate fixed count.",
          ["Suite","Method","Yes ratio","No ratio"],
          [[suite,LABELS.get(method,method),fmt(costrow(suite,"Yes",method).mean_relative_cost),fmt(costrow(suite,"No",method).mean_relative_cost)]
           for suite,methods in [("sequential",["count_validation","polynomial_only","single_hybrid","fh_hybrid","multi_hybrid"]),
                                  ("stream",["count_checkpoint","count_validation","polynomial_only","single_hybrid","fh_hybrid","multi_hybrid"])] for method in methods],"cost_comparison")
    p(f"On the large-scale sequential grid, the full hybrid uses {pct(seq_yes.mean_relative_cost)} and {pct(seq_no.mean_relative_cost)} of checkpoint-count cost in Yes and No cases, respectively. Its maximum observed false-positive rate is {pct(formal_yes.error_rate.max())}, and its maximum observed false-negative rate is {pct(formal_no.error_rate.max())}. These are finite-grid observations alongside, not replacements for, the analytical guarantee.")
    figure("sequential_ablation")
    p(f"In the actual labelled-stream suite, n ranges from 50 to 1,000, epsilon is 0.01 or 0.1, and each cell has 1,000 trials. The hybrid reduces mean cost by {pct(1-actual_yes.mean_relative_cost)} on Yes cases and {pct(1-actual_no.mean_relative_cost)} on No cases compared with immediate fixed count. The worst relative costs in those strata are {fmt(actual_yes.worst_relative_cost)} and {fmt(actual_no.worst_relative_cost)}. A method may therefore lose on individual configurations even when its equal-weight mean improves.")
    figure("actual_cost");figure("actual_cost_cdf")
    comparisons=paired[paired.suite=="stream"]
    table("Paired actual-cost contrasts: multiscale hybrid minus comparator, normalized by immediate fixed-count mean. Negative values favor the hybrid. Intervals describe Monte Carlo uncertainty on this fixed grid.",
          ["Truth","Comparator","Difference","95% MC interval","Holm p"],
          [[r.truth,LABELS[r.comparator],fmt(r.effect),f"[{fmt(r.mc_low95)}, {fmt(r.mc_high95)}]","<1e-300" if r.holm_pvalue<1e-300 else f"{r.holm_pvalue:.2g}"] for r in comparisons.itertuples()],"paired_actual_cost")
    p("The count baseline stops at the exact observation that first reveals category n+1 and otherwise stops at its interval-certified fixed budget. All hybrids also permit immediate direct-count rejection; validation labels are fully charged. This study separates a genuine improvement in inference from savings caused by an artificially delayed baseline. Poisson stopped-mean quantiles from compressed simulations are never presented as actual acquisition quantiles.")
    section("8. Adversarial and effective-support evaluation")
    p(f"The search enumerates head-tail and multi-level shapes with tail-size factors 2,10,100,1,000, tail exponents, boundary ratios, and lognormal nulls. For each of six n/epsilon pairs, three nulls and three alternatives are selected by fixed-budget error and then sequential cost. All {len(selected)} selected distributions are frozen before independent tests with 10,000 trials per fixed-budget cell and 5,000 per sequential cell. Candidate scores, selected probabilities, exact promise checks and raw retest outcomes are retained.")
    p("For the multiscale fixed test, the largest independently observed alternative non-rejection rate at C=4 is "+pct(adv.query("method == 'multi' and truth == 'No' and C == 4").error_rate.max())+". This directly tests whether ordinary benchmark success persists near searched difficult cases. Search is restricted to the stated families; it is not a proof of a least favorable distribution over the full simplex.")
    figure("adversarial_risk")
    eq(r"\Pr(\widehat S>S),\qquad \Pr\{\widehat S<\min(\operatorname{eff}_{\varepsilon}(p),n)\},\qquad \Pr\{\min(\operatorname{eff}_{\varepsilon}(p),n)\le\widehat S\le S\}.","Lower-bound metrics: overcoverage failure; effective-support shortfall; joint success.")
    p("The practical lower bound clips an exact rationally evaluated conservative statistic to [1,n] and floors it. When S<=n the same event controlling all scales implies the lower bound is at most S; when S>n, clipping gives coverage automatically. Flooring avoids an unsafe ceil at an interval boundary but may lose less than one category. The lower effective-support inequality is measured empirically; it is not inherited from the original asymptotic construction.")
    p("Effective-support integer targets are independently recomputed using exact rationally normalized probability weights. The legacy floating-point floor can differ by one at equality. Raw simulation CSVs are immutable; tables/*_exact_metrics.csv are the canonical derived tables for lower-bound analysis, with an explicit audit of any changed targets or trial classifications. This postprocessing changes no algorithm, binary decision, cost, sample or selected parameter.")
    figure("effective_lower")
    lower_rows=[]
    for method,g in fixed[(fixed.n==10000)&(fixed.epsilon==.01)&(fixed.C==4)&fixed.method.isin(["count","multi","single","fh_certified"])].groupby("method"):
      lower_rows.append([LABELS[method],pct(g.lower_joint_success.min()),pct((g.lower_over_errors/g.trials).max()),fmt(g.mean_gain_over_clipped_D.mean(),1)])
    table("Effective-support lower bounds at n=10,000, epsilon=0.01, C=4; all eleven scenarios including Gap are evaluated.",
          ["Method","Worst joint success","Worst over-S rate","Mean gain over clipped D"],lower_rows,"effective_support")
    section("9. Natural groups, held-out groups, and sampling-design repair")
    p("We use NIPS word counts, Enron word counts, MovieLens 1M ratings, Online Retail product rows, BCI tree counts, and the 2007 seine stratum of Inch Lake fish captures [6-10]. Units are documents, users, invoices, spatial quadrats and capture events where available. These published records define finite empirical reference distributions. Their observed support is not asserted to be the support of an external population, and individual records are not automatically independent.")
    data_rows=json.loads((out/"structure_targets.json").read_text())
    table("Finite empirical reference populations used in the new group-structure experiments.",
          ["Dataset","Records","Observed categories","Natural groups"],
          [[r["dataset"],fmt(r["records"]),fmt(r["support"]),fmt(r["groups"])] for r in data_rows],"data_inventory")
    p("The UCI text corpora use a publisher-filtered vocabulary, so their rare tails are not an uncensored language distribution. Retail samples are positive-quantity, positive-price product rows, not quantity-expanded independent transactions. BCI is the published 1982 large-tree subset; years are not pooled. Fish records are restricted to one year and one gear. Original licenses, download hashes, filtering rules and sample units are retained in the data provenance manifests.")
    figure("group_structure")
    p("Natural-group stress samples preserve the original marginal record distribution but draw blocks of 5,20,100 observations from the same group. The largest observed multiscale non-rejection rate over these tested No scenarios is "+pct(summary["structural_max_observed_fn"])+". The iid certificate does not apply to those correlated experiments. The fixed-budget output is Inconclusive when evidence is insufficient; interpreting it as a certified Yes would be incorrect.")
    p("A repair experiment draws a fresh group independently for every retained label. Size-proportional group selection followed by a uniform record preserves the original record distribution. Uniform group selection defines a distinct event-weighted target, which is reported separately. Three group-disjoint splits per dataset additionally test transfer to new empirical frequency spectra: query n is chosen on training groups, while truth and error are evaluated on independently resampled held-out-group spectra. This is not a claim of independent field-level population truth.")
    holdout_rows=[]
    for dataset,g in frames["group_holdout"].query("method == 'multi'").groupby("dataset"):
      holdout_rows.append([dataset,str(len(g)),pct(g[g.truth=="Yes"].error_rate.max()),pct(g[g.truth=="No"].error_rate.max()),pct(g.lower_joint_success.mean())])
    table("Multiscale evaluation on three group-disjoint holdout splits, two training-defined queries and two budgets. A dash means that no cell belongs to that promise stratum.",
          ["Dataset","Cells","Max null error","Max non-rejection","Mean joint lower success"],holdout_rows,"group_holdout")
    section("10. Sensitivity, reference branches, and computational cost")
    figure("sensitivity")
    p("Confidence sensitivity covers delta=0.1,0.05,0.01,0.001; the smallest delta uses 20,000 trials per cell. One-at-a-time changes vary B, degree cap, weight cap, validation allocation, gating and checkpoint density. These outcomes are diagnostic and never reselect the submitted configuration. Failure to establish a small error target by an empirical confidence bound is distinguished from violation of the analytical guarantee.")
    risks=frames["sensitivity"].query("variant == 'delta' and method == 'multi_hybrid'")
    risk_rows=[]
    for delta,g in risks.groupby("delta"):
      risk_rows.append([fmt(delta),str(int(g.trials.iloc[0])),pct(g[g.truth=="Yes"].error_rate.max()),
                        pct(g[g.truth=="No"].error_rate.max()),pct(g.error_simultaneous_upper95.max())])
    table("Confidence-level stress: worst rates and simultaneous upper limits over tested multiscale-hybrid cells. An empirical upper limit can exceed the requested risk even when the analytical guarantee applies.",
          ["Target delta","Trials/cell","Max FP","Max FN","Max simultaneous upper"],risk_rows,"risk_sensitivity")
    refplans=json.loads((out/"reference_plans.json").read_text())
    table("Literal finite-scale reference branches, amplified conservatively to at most 5% error. Costs are prescribed sampling budgets, not simulation runtime.",
          ["n","epsilon","Wang total mean","FH fixed samples"],
          [[fmt(r["n"]),fmt(r["epsilon"]),fmt(r["wang_total_mean"]),fmt(r["fh_total_samples"])] for r in refplans],"reference_budgets")
    p("For the literal Wang constants, every tested realistic n lies in the observed-count branch. For the explicit FH constant a=1/128, all tested reference epsilon values lie in its small-epsilon branch; Proposition 2.2 with n+1 gives a fixed count tester. Independent majority repetition uses the stated worst-case error bounds, including the 1/10 Markov bound in the FH proof. We simulate the actual Poisson or fixed-multinomial models accordingly. These conservative reference costs are useful provenance checks but not the sole practical baseline; the immediate count baseline is much stronger.")
    figure("runtime")
    p("We separate parameter compilation, grouped simulation and decision scoring. Scoring includes integer certification of candidate crossings. Runtime uses fresh processes, one BLAS thread and nine scoring repetitions; peak RSS includes the interpreter and dependencies. Compression makes simulation of enormous known supports computationally feasible, but does not reduce the number of observations that a real sampling procedure must acquire.")
    section("11. Limitations and research claims")
    p("This study establishes a reproducible operating picture for a particular certified construction and finite set of distributions. It does not establish distribution-free acceleration over immediate counting, a minimax lower bound for the hybrid, or a new n/(epsilon log n) guarantee. Development and test shapes belong to overlapping broad distribution families even though parameters and random streams are separated. Adversarial selection explores only a specified family. Actual labelled-stream experiments stop at n=1,000; larger grids use an expected-cost surrogate with an explicit interpretation.")
    p("The empirical data do not provide an independent inventory of all species in a lake or all categories in an unseen population. No new field survey was performed. Group resampling addresses a defined empirical target; it does not automatically correct unknown capture probabilities, drift or unseen categories. Native-threshold FH-shaped and Chao1 baselines are heuristic diagnostics, not guaranteed competitors under the full promise model.")
    p("A top-tier submission additionally requires an independently assessed novelty claim and venue-specific positioning. The proposed finite-sample composition uses established ingredients, so its theoretical novelty should not be presumed from the size of the experimental package. This manuscript is a complete, evidence-linked research draft; it is not a representation that a venue has accepted it or that experiments alone suffice for acceptance.")
    section("12. Reproducibility")
    p("The artifact contains the frozen design, correctness amendment, selected development configurations, source snapshots, dependency versions, data hashes, exact boundary certificates, raw trial arrays, per-cell CSVs, plotting/analysis code, PDF figures and this manuscript in HTML, PDF and LaTeX. The result audit rechecks original source/data hashes and verifies summary statistics against raw outcomes. Formal and smoke outputs are separate. Historical and superseded experiments are not silently relabelled as new results.")
    p("From the project root, create a compatible Python environment, install requirements-lock.txt plus reportlab==4.4.4, and run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m publication.run --phase all --out results_reproduction. Original data can be reconstructed with python run_large_benchmark.py --phase prepare. Raw and derived source datasets, especially MovieLens, are omitted from the portable package; their download scripts, provenance and hashes are included. Reproduction does not require new tuning on formal outcomes.")
    section("13. Conclusion")
    p("Certified polynomial rejection and independent validation can save observations in complementary regimes. The magnitude of the benefit depends strongly on the count stopping rule, distribution shape, confidence level and sampling design. Reporting exact risk control, practical power, actual acquisition cost and dependence failures separately is essential for interpreting support-size testing experiments.")
    section("References")
    references=[
      ("[1] Qisheng Wang. Optimal Support Size Testing. Supplied manuscript 2610.18723v1.pdf.","https://arxiv.org/abs/2610.18723"),
      ("[2] Renato Ferreira Pinto Jr. and Nathaniel Harms. Testing Support Size More Efficiently Than Learning Histograms. TheoretiCS 5, 2026; arXiv:2410.18915v4.","https://arxiv.org/abs/2410.18915v4"),
      ("[3] Yihong Wu and Pengkun Yang. Chebyshev polynomials, moment matching, and optimal estimation of the unseen. arXiv:1504.01227.","https://arxiv.org/abs/1504.01227"),
      ("[4] I (Eli) Chien and Olgica Milenkovic. Support Estimation via Regularized and Weighted Chebyshev Approximations. arXiv:1901.07506v2.","https://arxiv.org/abs/1901.07506v2"),
      ("[5] Steven R. Howard et al. Time-uniform, nonparametric, nonasymptotic confidence sequences. arXiv:1810.08240.","https://arxiv.org/abs/1810.08240"),
      ("[6] David Newman. Bag of Words. UCI Machine Learning Repository. DOI:10.24432/C5ZG6P.","https://archive.ics.uci.edu/dataset/164/bag+of+words"),
      ("[7] F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. DOI:10.1145/2827872.","https://grouplens.org/datasets/movielens/1m/"),
      ("[8] Daqing Chen. Online Retail. UCI Machine Learning Repository. DOI:10.24432/C5BW33.","https://archive.ics.uci.edu/dataset/352/online+retail"),
      ("[9] BCI forest tree counts distributed with vegan; underlying ecological source Condit et al. (2002).", "https://vegandevs.github.io/vegan/reference/BCI.html"),
      ("[10] Derek H. Ogle / FSAdata. InchLake1 fish capture records.","https://fishr-core-team.github.io/FSAdata/reference/InchLake1.html"),
    ]
    for text,url in references:blocks.append(dict(kind="reference",text=text,url=url))
    save(out/"manuscript_content.json",dict(title=title,blocks=blocks))
    return title,blocks


def html_document(title,blocks,out,name):
    css="body{font:16px/1.75 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#203044}h1,h2{line-height:1.3}h2{margin-top:42px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9e1e7;padding:8px;text-align:left}th{background:#eef4f7}figure{margin:28px 0}img{width:100%}figcaption{font-size:13px;color:#526273}pre{padding:16px;background:#f1f5f7;white-space:pre-wrap}a{color:#007b80}.note{background:#edf8f3;padding:18px;border-left:4px solid #078773}"
    chunks=[f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>{html.escape(title)}</title><style>{css}</style><body><h1>{html.escape(title)}</h1><p>Research manuscript · reproducible experimental draft · generated from frozen results</p>"]
    for b in blocks:
      k=b["kind"]
      if k=="section":chunks.append("<h2>"+html.escape(b["text"])+"</h2>")
      elif k=="p":chunks.append("<p>"+html.escape(b["text"])+"</p>")
      elif k=="math":chunks.append("<pre>"+html.escape(b["text"])+"</pre>")
      elif k=="reference":chunks.append(f"<p>{html.escape(b['text'])} <a href='{html.escape(b['url'])}'>Source</a></p>")
      elif k=="table":
        chunks.append("<p><b>"+html.escape(b["caption"])+"</b></p><table><thead><tr>"+"".join("<th>"+html.escape(str(x))+"</th>" for x in b["headers"])+"</tr></thead><tbody>")
        chunks.extend("<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in row)+"</tr>" for row in b["rows"]);chunks.append("</tbody></table>")
      elif k=="figure":
        png=base64.b64encode((out/"figures"/(b["name"]+".png")).read_bytes()).decode()
        chunks.append(f"<figure><img src='data:image/png;base64,{png}'><figcaption>{html.escape(b['caption'])}</figcaption></figure>")
    chunks.append("</body></html>");(out/name).write_text("\n".join(chunks))


def export_tex(title,blocks,out):
    chunks=[r"\documentclass[10pt]{article}",r"\usepackage[margin=0.85in]{geometry}",r"\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,hyperref}",
            r"\title{"+tex_escape(title)+"}",r"\author{Anonymous research draft}",r"\date{}",r"\begin{document}\maketitle"]
    for b in blocks:
      k=b["kind"]
      if k=="section":chunks.append(r"\section*{"+tex_escape(b["text"])+"}")
      elif k=="p":chunks.append(tex_escape(b["text"])+"\n")
      elif k=="math":chunks.append("\\[\n"+b["latex"]+"\n\\]")
      elif k=="reference":chunks.append(tex_escape(b["text"])+r" \url{"+b["url"]+"}\n\n")
      elif k=="figure":chunks.append(r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/"+b["name"]+r".pdf}\caption{"+tex_escape(b["caption"])+r"}\end{figure}")
      elif k=="table":chunks.append(r"\begin{table}[htbp]\centering\small\caption{"+tex_escape(b["caption"])+"}\n\\resizebox{\\linewidth}{!}{\n"+table_tex(b["headers"],b["rows"])+"}\n\\end{table}")
    chunks.append(r"\end{document}");(out/"manuscript.tex").write_text("\n".join(chunks))


def export_pdf(title,blocks,out):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPaper",fontName="Helvetica",fontSize=9.2,leading=13.2,spaceAfter=7))
    styles.add(ParagraphStyle(name="CaptionPaper",fontSize=8,leading=10.5,textColor=colors.HexColor("#475665"),spaceAfter=10))
    styles.add(ParagraphStyle(name="EquationPaper",fontName="Courier",fontSize=8,leading=11,backColor=colors.HexColor("#f1f5f7"),borderPadding=7,spaceAfter=12))
    styles.add(ParagraphStyle(name="CellPaper",fontSize=7,leading=9))
    width=510
    story=[Paragraph(html.escape(title),styles["Title"]),Paragraph("Anonymous research draft · Complete experimental artifact",styles["BodyPaper"]),Spacer(1,10)]
    for b in blocks:
      k=b["kind"]
      if k=="section":story.append(Paragraph(html.escape(b["text"]),styles["Heading1"]))
      elif k in ["p","reference"]:story.append(Paragraph(html.escape(b["text"]),styles["BodyPaper"]))
      elif k=="math":story.append(Paragraph(html.escape(b["text"]),styles["EquationPaper"]))
      elif k=="figure":
        path=out/"figures"/(b["name"]+".png");iw,ih=ImageReader(str(path)).getSize()
        scale=min(width/iw,300/ih)
        story.append(KeepTogether([Image(str(path),width=iw*scale,height=ih*scale),Paragraph(html.escape(b["caption"]),styles["CaptionPaper"])]))
      elif k=="table":
        cells=[[Paragraph(html.escape(str(x)),styles["CellPaper"]) for x in row] for row in [b["headers"],*b["rows"]]]
        t=Table(cells,colWidths=[width/len(b["headers"])]*len(b["headers"]),repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf1f5")),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#82939e")),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([Paragraph(html.escape(b["caption"]),styles["CaptionPaper"]),t,Spacer(1,12)])
    def footer(canvas,doc):
      canvas.saveState();canvas.setFont("Helvetica",8);canvas.setFillColor(colors.gray)
      canvas.drawString(42,25,"OSST finite-sample study · generated from audited experiment outputs")
      canvas.drawRightString(552,25,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(out/"manuscript.pdf"),pagesize=(594,792),leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=42,
                      title=title,author="Anonymous research draft").build(story,onFirstPage=footer,onLaterPages=footer)


def chinese_report(frames,eff,cost,paired,summary,figures,out):
    rows=cost.query("suite == 'stream' and method == 'multi_hybrid'")
    yes=rows[rows.truth=="Yes"].iloc[0];no=rows[rows.truth=="No"].iloc[0]
    blocks=[dict(kind="p",text=f"本轮完成 {summary['total_method_trial_evaluations']:,} 次方法—试次评估。不同方法共享样本，所以该数字不是独立数据集或独立抽样批次数。完整结果由原始输出核对通过，论文稿、图表、逐次数据和复现入口均已生成。"),
            dict(kind="p",text=f"最重要的实际成本结论：在 n=50–1,000、epsilon=0.01/0.1 的逐条样本流实验中，多尺度组合方法相对立即停止的固定预算计数基线，Yes 场景平均使用 {pct(yes.mean_relative_cost)} 的样本，No 场景使用 {pct(no.mean_relative_cost)}。验证样本已计入。该结果不能外推为对所有分布的统一加速。"),
            dict(kind="p",text="一个必须保留的负面结果：多尺度并非所有参数下最省样本。n=1,000 时多尺度通过预算约为 C=4、单尺度为 C=5；到 n=1,000,000 时顺序反转。经过重新认证的 FH 形状版本在若干小 epsilon 配置中也更省样本。现有结果不支持把实用版本写成普遍优于对照的方法，也没有实证确立原理论的对数因子优势。"),
            dict(kind="section",text="研究定位与可交付内容"),
            dict(kind="p",text="这是完整的论文实验稿与可复现材料，不是 CCF-A 录用承诺。当前可支持的贡献是有限样本认证、强基线下的成本评估、困难分布检验和采样结构分析；新方法仍需独立的创新性论证，不能用实验规模替代最优阶证明。英文稿采用算法与实验论文结构，未伪造作者、投稿或录用信息。"),
            dict(kind="section",text="七组实验的完成情况"),
            dict(kind="table",caption="新实验库存",headers=["实验","方法配置数","方法—试次评估"],rows=[[r["suite"],fmt(r["method_cells"]),fmt(r["method_trial_evaluations"])] for r in summary["inventory"]]),
            dict(kind="section",text="多尺度、小 epsilon 与文献对照"),
            dict(kind="p",text="主网格覆盖 n 从 1,000 到 1,000,000，epsilon 最小为 0.001。多尺度、单尺度和 FH 形状家族各获得八个开发候选。FH 原文保守常数对应的计数分支、调参后的原阈值启发式版本、重新认证的实用版本分开标注。单尺度比较不再固定使用较弱的 j=0。")]
    for name in ["fixed_risk","sample_complexity","scale_mechanism"]:blocks.append(dict(kind="figure",**next(f for f in figures if f["name"]==name)))
    blocks.extend([dict(kind="section",text="停止规则与真实采样成本"),dict(kind="p",text="大规模压缩模拟保留跨检查点的联合泊松过程，其训练均值加验证条数用于估计期望成本。逐条样本流实验单独统计真正消耗的标签，并允许所有方法在发现第 n+1 类时立即停止。两种口径不混用，实际成本的 P90/P99 仅从真实标签计数计算。")])
    for name in ["sequential_ablation","actual_cost","actual_cost_cdf"]:blocks.append(dict(kind="figure",**next(f for f in figures if f["name"]==name)))
    blocks.extend([dict(kind="section",text="困难分布、两侧下界与真实分组"),dict(kind="p",text="困难分布在独立搜索流上选择，再冻结为 36 个总体；正式定预算复测每单元 10,000 次，顺序复测每单元 5,000 次。支持集下界分别检查超过 S、低于 min(effective_support,n)、两侧同时成立三个事件。真实数据增加自然分组相关性、独立按组抽样修正及三次完整分组留出。")])
    for name in ["adversarial_risk","effective_lower","group_structure","sensitivity","runtime"]:blocks.append(dict(kind="figure",**next(f for f in figures if f["name"]==name)))
    blocks.extend([dict(kind="section",text="修订透明度与尚不能声称的结论"),
                   dict(kind="p",text="最初运行中，精确有理数审计发现若干名义边界总体因浮点舍入落到承诺区域之外，幅度约 10^-20 到 10^-18。已保留原输出，记录修订，保持开发集选出的参数不变，修正构造并重跑所有正式实验；新边界质量证书随结果发布。这一流程是本地前瞻协议加公开修订，不能称为从未修改的外部预注册。"),
                   dict(kind="p",text="有效支持集的整数目标也经独立有理数复核：部分浮点取整诊断值相差 1，精确重算改变了 1 个方法配置的下界错误计数。原始模拟 CSV 保持不变，论文下界指标以 tables/*_exact_metrics.csv 为准；effective_support_audit.json 完整列出差异。算法参数、抽样、二元判定和样本成本均未改变。"),
                   dict(kind="p",text="目前没有新增实地鱼类调查或独立的全湖物种清单，因此不能宣称估准了全湖鱼种总数。自然分组的修正针对明确的经验目标分布，不自动解决捕获概率偏差、时间漂移或全新未见类别。有限网格实验也不证明组合方法拥有原论文的最优阶。"),
                   dict(kind="section",text="文件入口"),
                   dict(kind="p",text="英文完整稿：manuscript.pdf / manuscript.html / manuscript.tex；逐单元结果：各 suite 的 CSV；逐次输出：raw/；图表：figures/；统计汇总：tables/；协议与源码：design.json、protocol.json、source_snapshot/；审计：result_audit.json；复现说明：REPRODUCE.txt。")])
    html_document("支持集检验：完整实验与论文稿",blocks,out,"report.html")
    path=out/"report.html";s=path.read_text().replace("lang='en'","lang='zh-CN'")
    links="<div class='note'><a href='manuscript.pdf'>英文论文 PDF</a> · <a href='manuscript.html'>英文完整稿</a> · <a href='manuscript.tex'>LaTeX 源码</a> · <a href='summary.json'>结果摘要</a> · <a href='REPRODUCE.txt'>复现说明</a></div>"
    path.write_text(s.replace("<body>","<body>"+links,1))


def package(out):
    deps=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (out/"requirements-publication-lock.txt").write_text(deps)
    instructions="""OSST follow-up experimental artifact

Open report.html for the Chinese report; manuscript.pdf/html/tex contain the English research draft.
All figures are available as PNG and PDF. All experiment summaries are CSV; raw trial arrays are compressed NumPy files.

Reproduce from the project root with Python 3.14 (the exact executed version is recorded in protocol.json):
  python -m venv .venv
  .venv/bin/python -m pip install -r results_publication/requirements-publication-lock.txt
  .venv/bin/python run_large_benchmark.py --phase prepare
  .venv/bin/python -c 'from fish_data import prepare_fish_data; prepare_fish_data()'
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m publication.run --phase all --out results_reproduction

For an isolated phase use --phase fixed/sequential/search/adversarial/stream/structure/sensitivity/reference/runtime.
The adversarial phase requires completed search. To regenerate analysis only:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -m publication.run --phase report

Historical fish data for fish_2007_seine are prepared by the pinned-source command above.

Source datasets and derived matrices are omitted from the archive. Obtain them from original publishers using the
provided preparation scripts; observe their recorded terms. MovieLens redistribution is not authorized by this artifact.

The original formal run was superseded because a rational audit found ulp-level boundary mislabelling. The correction,
frozen original parameter selection, and complete rerun are disclosed in design.json. The superseded outputs remain on
the working machine in results_publication_superseded_boundary_v1 and are not pooled with final results.

Expected-cost proxies and actual labelled-stream costs are different metrics. Native-threshold FH-shaped and Chao1
rows are heuristics. A fixed-budget failure to reject is Inconclusive. No field population census or optimal-order proof
is claimed for the practical hybrid. This is a research draft and experimental artifact, not an acceptance certificate.

Canonical lower-bound analysis uses tables/*_exact_metrics.csv. Raw simulation CSVs retain their floating diagnostic
effective-support integers. effective_support_audit.json documents exact rational re-evaluation; no binary decisions,
sample costs, parameters or random samples are changed by that derived-metric correction.
"""
    (out/"REPRODUCE.txt").write_text(instructions)
    # Keep a complete dependency snapshot, including unchanged legacy helper modules.
    for p in [*ROOT.glob("*.py"),ROOT/"requirements.txt",ROOT/"requirements-lock.txt",ROOT/"pytest.ini"]:
      dst=out/"source_snapshot"/p.name;dst.parent.mkdir(exist_ok=True,parents=True)
      if not dst.exists():dst.write_bytes(p.read_bytes())
    for p in (ROOT/"publication").glob("*.py"):
      dst=out/"source_snapshot/publication"/p.name;dst.parent.mkdir(exist_ok=True,parents=True)
      if not dst.exists() or p.name in {"report.py","aggregate.py"}:dst.write_bytes(p.read_bytes())
    save(out/"analysis_source_hashes.json",{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"publication/report.py",ROOT/"publication/aggregate.py"]})
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="MANIFEST.sha256"}
    (out/"MANIFEST.sha256").write_text("\n".join(f"{digest}  {name}" for name,digest in manifest.items())+"\n")
    archive=ROOT/"artifacts/osst_publication_artifact.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
      for p in sorted(out.rglob("*")):
        if p.is_file():z.write(p,str(p.relative_to(ROOT)))
      for p in [*ROOT.glob("*.py"),ROOT/"requirements.txt",ROOT/"requirements-lock.txt",ROOT/"pytest.ini"]:z.write(p,p.name)
      for p in (ROOT/"publication").glob("*.py"):z.write(p,str(p.relative_to(ROOT)))
      for p in [ROOT/"data/large/sources.json",ROOT/"data/large/dataset_summary.json",ROOT/"data/provenance.json"]:
        if p.exists():z.write(p,str(p.relative_to(ROOT)))
    (archive.with_suffix(".sha256")).write_text(sha(archive)+"  "+archive.name+"\n")
    return archive


def report(out=OUT):
    frames,eff,cost,paired,summary=aggregate(out)
    figures=figure_all(frames,eff,cost,out)
    title,blocks=manuscript(frames,eff,cost,paired,summary,figures,out)
    html_document(title,blocks,out,"manuscript.html")
    export_tex(title,blocks,out);export_pdf(title,blocks,out)
    chinese_report(frames,eff,cost,paired,summary,figures,out)
    artifact=package(out)
    print(f"Report, manuscript and artifact complete: {artifact}",flush=True)
