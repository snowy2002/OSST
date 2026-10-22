"""Build the Chinese comparison report and standalone research figures."""
from fractions import Fraction
from pathlib import Path
import csv
import html
import json
import math
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_parameter_search'


def main():
    cert=json.loads((OUT/'certificate.json').read_text())
    lower=json.loads((OUT/'lower_bound_certificate.json').read_text())
    summary=json.loads((OUT/'experiment_summary.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    with (OUT/'experiments.csv').open() as f:rows=list(csv.DictReader(f))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):
        plt.tight_layout();plt.savefig(figures/f'{name}.png',dpi=160)
        plt.savefig(figures/f'{name}.pdf');plt.close()
    plt.figure(figsize=(7.2,4))
    for C,label in [('star','C = 144 (ln 2)^2'),('70','C = 70')]:
        rs=[r for r in cert['results'] if r['C']==C]
        plt.semilogy([r['L0'] for r in rs],[100*float(Fraction(r['target'])) for r in rs],'-o',label=label)
    plt.axhline(1,color='gray',linestyle='--',label='1% target')
    plt.ylabel('Certified single-run failure upper bound (%)');plt.xlabel('L0')
    plt.xticks([14,15,16,18]);plt.legend();save('certified_bounds')
    plt.figure(figsize=(7.2,4))
    constants=[6,10,20,40,60,69,70]
    worst=[]
    for C in constants:
        selected=[r for r in rows if r['suite']=='adversarial' and r['C']==str(C)]
        worst.append(max(int(r['errors'])/int(r['trials'])*100 for r in selected))
    plt.bar([str(c) for c in constants],worst,color=['#c85945']*5+['#a58226','#087e8b'])
    for i,value in enumerate(worst):plt.text(i,value+1,f'{value:.2f}%',ha='center',fontsize=9)
    plt.ylim(0,107);plt.axhline(1,color='gray',linestyle='--')
    plt.xlabel('C (L0 = 14)');plt.ylabel('Worst observed null rejection rate (%)')
    plt.title('Constructed stress tests: 2,000 trials per cell');save('adversarial_errors')
    plt.figure(figsize=(7.2,4))
    xs=np.linspace(68.5,70.5,400)
    a=np.sqrt(512/xs)*3*np.log(2)
    gamma=a*np.arcsinh(a/32)-np.sqrt(1024+a*a)+32
    plt.plot(xs,gamma,label='Scale-zero coefficient growth exponent')
    plt.axhline(.5,color='gray',linestyle='--')
    plt.axvline(69.006271385,color='#c85945',linestyle=':',label='Necessary boundary 69.00627138...')
    plt.axvline(144*np.log(2)**2,color='#087e8b',linestyle=':',label='Sufficient C = 69.185234...')
    plt.xlabel('C');plt.ylabel('gamma(C)');plt.legend(fontsize=8);save('necessary_sufficient')
    mainrows=[r for r in rows if r['suite'] in ['natural','synthetic','lifted'] and r['C']=='70']
    bycell={}
    for r in mainrows:bycell.setdefault(r['cell'],{})[r['L0']]=r
    improvements=[]
    for cell,methods in bycell.items():
        a,b=methods['14'],methods['16']
        gain=float(a['mean_lower_lo'])-float(b['mean_lower_hi'])
        if gain>1e-7:
            improvements.append(dict(cell=cell,suite=a['suite'],scenario=a['scenario'],n=int(a['n']),
                                     epsilon=a['epsilon'],gain=gain,relative_gain=gain/int(a['n'])))
    (OUT/'paired_lower_improvements.json').write_text(json.dumps(improvements,indent=2)+'\n')
    plt.figure(figsize=(7.2,4))
    for suite,color in [('synthetic','#087e8b'),('lifted','#a58226')]:
        selected=[r for r in improvements if r['suite']==suite]
        plt.scatter([r['n'] for r in selected],[r['relative_gain'] for r in selected],s=24,alpha=.65,label=suite)
    plt.xscale('log');plt.xlabel('n');plt.ylabel('Paired mean lower-bound gain / n')
    plt.title('L0 = 14 versus L0 = 16; identical C = 70 samples');plt.legend();save('paired_gain')
    recommended=[r for r in rows if r['C']=='70' and r['L0']=='14']
    rec_trials=sum(int(r['trials']) for r in recommended)
    rec_poly=sum(int(r['trials']) for r in recommended if r['branch']=='chebyshev')
    cp=max(float(r['errors_upper95_simultaneous']) for r in recommended if r['errors'])
    targets={f'{r["C"]}_{r["L0"]}':float(Fraction(r['target'])) for r in cert['results']}
    table=''.join(f'<tr><td>{"144(ln 2)² ≈ 69.185234" if r["C"]=="star" else "70"}</td>'
                  f'<td>{r["L0"]}</td><td>&lt; {100*float(Fraction(r["target"])):.6g}%</td>'
                  f'<td>{math.ceil(math.exp(r["L0"]-1)):,}</td></tr>' for r in cert['results'])
    stress=''.join(f'<tr><td>{C}</td><td>{p:.2f}%</td></tr>' for C,p in zip(constants,worst))
    selected_summary=[r for r in summary['summary'] if r['C']=='70' and r['L0']==14]
    suite_names=dict(natural='自然经验总体',synthetic='常规构造',lifted='自然形状扩展构造',
                     adversarial='困难构造',diffuse_control='稀疏碰撞对照')
    exp_table=''.join(f'<tr><td>{suite_names[r["suite"]]}</td><td>{r["cells"]}</td><td>{r["trials"]:,}</td>'
                      f'<td>{r["polynomial_trials"]:,}</td><td>{r["errors"]}</td><td>{r["coverage_errors"]}</td></tr>'
                      for r in selected_summary)
    payload=[]
    for r in rows:
        payload.append([r['suite'],r['scenario'],r['C'],int(r['L0']),int(r['n']),r['epsilon'],r['truth'],
                        r['branch'],int(r['trials']),None if r['errors']=='' else int(r['errors']),
                        int(r['coverage_errors']),float(r['mean_lower_lo']),float(r['mean_D'])])
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C₀、L₀ 参数探索与验证</title><style>
body{{font:16px/1.75 system-ui,sans-serif;color:#172a32;background:#f5f7f7;margin:0}}main{{max-width:1100px;margin:auto;padding:36px 24px}}
h1{{font-size:32px}}h2{{font-size:24px;margin-top:36px}}a{{color:#087e8b}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce4e5}}th{{background:#e9f0f0;position:sticky;top:0}}
.lead,.card{{padding:20px 24px;background:white;border-radius:12px;margin:18px 0}}.lead{{border-left:5px solid #087e8b}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{width:100%;background:white;border-radius:10px}}
input,select{{font:inherit;padding:6px;margin:5px;max-width:220px}}.scroll{{overflow:auto;max-height:540px}}code{{background:#e9eeee;padding:2px 5px}}
.muted{{color:#576871;font-size:14px}}@media(max-width:750px){{.grid{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style><main>
<h1>C₀、L₀ 已收敛到怎样的选择？</h1>
<p class="muted">2026-10-22 · 保持原论文构造，仅比较采样常数与分支阈值 · 原始结果保留在独立目录</p>
<div class="lead"><b>综合建议：C₀=70、L₀=14，独立运行一次。</b>
<p>原构造要求对所有 n、ε 和离散分布统一可靠时，新的下界推导排除了 C₀≤69；70 因此是最小可行整数。
L₀=14 已证明足够，单次失败概率小于 <b>0.09%</b>。若只做二元判定、希望少做多项式计算，保留
<b>(70,16)</b> 也合适，其单次界更低：<b>0.0057%</b>。L₀=14 不是已证明的全局最小阈值。</p>
<p>此前依据旧界建议的 27 次或 5 次重复已不再是达到 1% 的必要开销。
与 C₀=1025 的旧方案比较，单次期望样本减少 <b>{100*(1-70/1025):.2f}%</b>，为原来的 1/{1025/70:.2f}。
这是样本预算的比较，不代表 CPU 用时也按同一比例缩短。</p></div>

<h2>可靠性与分支的取舍</h2><p>所有行都是单次运行的统一理论上界，不是实测错误率。
在 iid 泊松采样、完整阶数和原统计量下，覆盖二元承诺检验及支持下界；在二元承诺成立时，两项可同时保证。
ε 是分布分离参数，不是失败概率。</p>
<table><thead><tr><th>C₀</th><th>L₀</th><th>单次失败概率上界</th><th>开始使用多项式的最小 n</th></tr></thead><tbody>{table}</tbody></table>
<div class="grid"><img src="figures/certified_bounds.png" alt="认证错误上界"><img src="figures/necessary_sufficient.png" alt="必要常数与充分常数"></div>
<div class="card"><b>预算比较器</b><p>固定 C₀、n 和 ε 后，L₀ 只改变分支，不改变期望样本量。</p>
<label>C₀ <select id="C"><option value="70">70</option><option value="star">144(ln 2)²</option></select></label>
<label>L₀ <select id="L0"><option>14</option><option>15</option><option>16</option><option>18</option></select></label>
<label>n <input id="n" type="number" min="1" value="1000000"></label>
<label>ε <input id="eps" type="number" min="0.000001" max="0.999999" step="0.01" value="0.1"></label>
<p id="budget"></p><p class="muted">显示值用于比较；精确计算以源码的 Arb 区间为准。m=C₀n/[ε(1+ln n)]。</p></div>

<h2>为何现在可以排除 69？</h2>
<p>新推导利用零号尺度权重的增长速度，构造一个占一半概率的主类与 n−1 个等概率稀有类。
当 C₀ 小于临界值时，配合合法的 εₙ，零号尺度的波动会压过判定阈值，误拒绝概率的下极限至少为 1/2。
固定次数的多数表决也不能解决这个统一保证问题。临界值已用精确有理数夹在
<b>69.00627138～69.00627139</b>；这一步与“旧误差界算不下来”有本质区别，是一个具体的渐近反例证明。</p>
<p>目前充分实数常数为 144(ln 2)²≈69.185234，与上述必要下界相差不足 <b>0.26%</b>。
因此整数 C₀ 的选择已经确定；实数常数仍存在一个小间隙。此结论限定于本项目的原构造、固定有限 L₀&gt;1、
全 n 和全 ε 的统一可靠性，不是所有支持测试算法的信息论最优结论，也不是有限规模上 69 必然失败。</p>
<p>下界证明中的三角阵列中心极限定理使用 <a href="https://people.eecs.berkeley.edu/~jordan/courses/210B-spring07/lectures/stat210b_lecture_2.pdf">Berkeley Stat210B 讲义的定理 9</a>；
所需的 Lindeberg 条件已在本次证明中逐项验证。新推导与证书是可复核的数学工作，尚非证明助手形式化或外部同行评审结果。</p>

<h2>本轮独立实验</h2>
<p>共 {summary['independent_sample_paths']:,} 条独立采样路径、{summary['method_evaluations']:,} 次方法评估，
包含十个自然经验总体、原有构造、分支边界、类别扩展和针对权重的困难构造。四个种子块，每个单元共 2,000 次。
同一个 C₀=70 的不同 L₀ 共用频数，便于直接比较；它们不被重复计为独立采样路径。</p>
<p><b>(70,14) 共 {rec_trials:,} 次试验、{rec_poly:,} 次进入多项式分支，未观察到二元错误或下界覆盖失效。</b>
其全部单元的同时 95% 错误率上界约为 {cp*100:.4f}%（按每个指标族分别控制）。零错误不等于真实错误率为零；
统一保证来自理论证明。</p>
<table><thead><tr><th>数据类型</th><th>单元数</th><th>试验次数</th><th>多项式次数</th><th>二元错误</th><th>下界失效</th></tr></thead><tbody>{exp_table}</tbody></table>
<p>自然数据均作为有限经验分布进行 iid 重采样，原生规模全部进入计数分支；类别扩展明确属于构造数据。
这些实验不证明未知自然总体的真实支持数，也没有验证相关采样。分组频数模拟时间不是实际收集数十亿条样本的时间。</p>

<h2>更小 C₀ 的失败样例与 L₀ 的收益</h2>
<div class="grid"><img src="figures/adversarial_errors.png" alt="困难样例错误率"><img src="figures/paired_gain.png" alt="降低L0提高支持下界"></div>
<table><thead><tr><th>C₀（L₀=14）</th><th>困难构造中的最坏实测误拒绝率</th></tr></thead><tbody>{stress}</tbody></table>
<p>C₀=60 的最坏单元为 n=10¹²、ε=0.1、稀有类目标泊松均值 16，错误率 6.35%；同一总体在 C₀=70 下未观察到错误。
C₀=69 在有限测试中也没有出错，这与极大 n 上的理论反例不矛盾。</p>
<p>在主实验的相同 C₀=70 频数上，L₀=14 相比 L₀=16 有 <b>{len(improvements)}</b> 个配置的平均整数下界得到认证的提高。
增益主要来自 n 在 442,414 到 3,269,017 之间时提前启用多项式；相应的计算量也会增加。
只做二元判定时，没有必要单为提前进入多项式而降低 L₀。</p>

<h2>筛选全部结果</h2><input id="query" placeholder="如 adversarial、60、442414" style="max-width:500px;width:80%"><span id="matches"></span>
<div class="scroll"><table><thead><tr><th>类型/场景</th><th>C₀/L₀</th><th>n / ε</th><th>真值/分支</th><th>次数</th><th>二元错误</th><th>下界失效</th><th>平均下界 / D</th></tr></thead><tbody id="resultrows"></tbody></table></div>

<h2>证据、复核和复现</h2><p>充分界：{cert['checks']:,} 项精确算术检查，完整覆盖 520 个闭区间和两个无穷半直线。
独立审计：{audit['checks']:,} 项检查，重算理论公式、核对原始文件、精确真值和事件；
必要下界的根区间另有精确级数证书和 512 位 Arb 交叉核验。</p>
<p><a href="proof.pdf">完整新证明 PDF</a> · <a href="proof.tex">LaTeX 源码</a> · <a href="certificate.json">充分界证书</a> ·
<a href="lower_bound_certificate.json">必要界证书</a> · <a href="audit.json">独立审计</a> · <a href="experiments.csv">全部实验 CSV</a> ·
<a href="experiment_summary.json">实验摘要</a> · <a href="experiment_protocol.json">冻结协议</a> · <a href="REPRODUCE.txt">复现说明</a> ·
<a href="tests.log">测试日志</a></p>
<p class="muted">困难测试 C₀=6、n=10¹² 曾因 512 位精度不足而明确停止，随后增加自动精度提升并恢复执行。
见 <a href="precision_amendment.json">数值精度修订记录</a>；修订前协议和源码保留，已完成原始频数、种子、预算与判定均未修改。
本轮没有改写旧 results_c7016/ 的数据或结论记录，新证明取代其中较松的重复次数建议。</p>
</main><script>
const data={json.dumps(payload,ensure_ascii=False)}, targets={json.dumps(targets)};
const escapeText=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function filterRows(){{let query=document.getElementById('query').value.toLowerCase();let matches=data.filter(r=>r.join(' ').toLowerCase().includes(query));
document.getElementById('matches').textContent=` ${{matches.length}} 条（显示前 300 条）`;
document.getElementById('resultrows').innerHTML=matches.slice(0,300).map(r=>`<tr><td>${{escapeText(r[0]+' / '+r[1])}}</td><td>${{r[2]}} / ${{r[3]}}</td><td>${{r[4]}} / ${{r[5]}}</td><td>${{r[6]}} / ${{r[7]}}</td><td>${{r[8]}}</td><td>${{r[9]===null?'Gap':r[9]}}</td><td>${{r[10]}}</td><td>${{r[11].toPrecision(6)}} / ${{r[12].toPrecision(6)}}</td></tr>`).join('');}}
function budget(){{let name=document.getElementById('C').value,C=name==='star'?144*Math.log(2)**2:70,L0=+document.getElementById('L0').value,n=+document.getElementById('n').value,eps=+document.getElementById('eps').value;
if(!(Number.isSafeInteger(n)&&n>=1&&eps>0&&eps<1)){{document.getElementById('budget').textContent='请输入正整数 n 和 0<ε<1。';return;}}
let L=1+Math.log(n),m=C*n/(eps*L),branch=L<L0?'简单计数':'多项式';document.getElementById('budget').textContent=`期望样本量约 ${{m.toLocaleString('zh-CN',{{maximumFractionDigits:2}})}}；${{branch}}分支；单次失败概率 < ${{(100*targets[name+'_'+L0]).toPrecision(4)}}%。`;}}
document.getElementById('query').addEventListener('input',filterRows);['C','L0','n','eps'].forEach(id=>document.getElementById(id).addEventListener('input',budget));filterRows();budget();
</script></html>'''
    (OUT/'report.html').write_text(doc)
    result=dict(recommendation=dict(C=70,L0=14,repeats=1,failure_upper=.0009),
                binary_only_alternative=dict(C=70,L0=16,repeats=1,failure_upper=.000057),
                integer_C_optimality_scope='Unchanged construction, fixed finite L0>1, all n and epsilon, fixed odd majority amplification.',
                necessary_C_interval=['69.00627138','69.00627139'],
                sufficient_real_C='144*(ln 2)^2',L0_minimality='Unresolved; 14 is sufficient.',
                recommended_trials=rec_trials,recommended_poly_trials=rec_poly,
                recommended_errors=0,recommended_coverage_errors=0,
                paired_improved_cells=len(improvements),
                independent_sample_paths=summary['independent_sample_paths'],
                exact_checks=cert['checks'],audit_checks=audit['checks'])
    (OUT/'conclusions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for name in ['certify.py','lower_bound.py','audit.py','report.py','explore.py','__init__.py']:
        target=OUT/'source_snapshot/parameter_search'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'parameter_search'/name,target)
    (OUT/'REPRODUCE.txt').write_text('''复现与证据说明（工作目录为 OSST 仓库根目录）

1. 搜索候选（浮点探索，不作为证明）：
   .venv/bin/python -m parameter_search.explore
2. 从头计算充分界的精确有理数证书：
   .venv/bin/python -m parameter_search.certify
3. 从头计算必要下界的根区间，同时用 Arb 交叉检查：
   .venv/bin/python -m parameter_search.lower_bound
4. 重新核验全部原始频数、事件、哈希和充分界公式：
   OPENBLAS_NUM_THREADS=1 .venv/bin/python -m parameter_search.audit
5. 运行测试：
   .venv/bin/python -m pytest -q
6. 重建图表与中文报告：
   .venv/bin/python -m parameter_search.report
   .tools/tectonic-0.17.0/tectonic results_parameter_search/proof.tex --outdir results_parameter_search

实验主程序默认断点续跑当前冻结目录，不会把复用旧单元算成新试验。
若要从零重跑到新目录，保留当前证据，运行：
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
   from pathlib import Path
   from parameter_search import experiments as e
   e.OUT=Path('results_parameter_search_reproduced').resolve()
   e.main()
   PY
同一冻结种子会复现同一组伪随机试验；时间字段、压缩文件元数据可能不同。
需要新随机试验时，在新的输出目录中另设 e.DESIGN['seed']，不要修改已有协议。

自然数据沿用 data/c1025/ 固定来源与许可；源数据/准备文件哈希在实验协议中。
完整原始数据在 raw/，精确概率和单元摘要在 cells/。results_c7016/ 未改动。
算法权重在小 C 的压力测试中需要更高精度，修订历史见 precision_amendment.json。
计数随机数和分组多项式分布由 NumPy/SciPy 双精度生成；数学概率保证不由 PRNG 认证。
数学证书认证有限算术义务；解析证明、渐近下界与标准中心极限定理见 proof.pdf。
这不是 Lean/Coq 形式化，也不是外部同行评审结论。
''')
    print('REPORT COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
