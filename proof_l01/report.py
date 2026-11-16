"""Chinese reassessment after the complete sufficient-constant proof passes."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_proof_l01'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    if summary['status']!='PASS':
        raise ValueError('Do not publish a positive proof report without passed certificates')
    body='''C₀=70、L₀=1：完整充分性证明与可靠性复审

结论：C₀=70、L₀=1 的理论充分性缺口现已补齐。保持原算法的多尺度、次数向上取整、校正项和输出规则，使用实数泊松均值 m=70n/[ε(1+ln n)]，可以证明所有正整数n、所有0<ε<1、所有有限或可数支持分布上，单次运行的联合失败概率严格小于0.331，成功概率严格大于0.669，因此大于2/3。

保证的对象是：估计值落在[min(effε(p),n),min(S(p),n)]内；当S≤n时二元检验接受，当Δn≥ε时拒绝。两个承诺区间内，这些要求同时成立。承诺间隙只要求支持区间正确。

本次是自主建立的C=70充分性证明。它不依赖缺失的ChatGPT Library证书，也不宣称重新证明了C★的最优性或临界值可取。

证明怎样补齐
1. n=1…73：覆盖全部2701个(n,S)。先选定真实概率最大的类别作为锚点，用总概率质量约束控制它的低计数概率；再把多个尺度的上偏差包含在共同的零计数/单例事件中。我们完整证明了同均值归约引理，把任意不相同的泊松均值归约到一维上确界，再用向外舍入区间覆盖整个实数半线。最坏上尾界在n=27，小于0.330679396。
2. n=74：独立桥接证书覆盖7个尺度，上尾小于0.301518。
3. n=75…403：全部329个整数、2570个尺度，用保留零计数负项的精确矩母函数认证，上尾小于0.291304。
4. 7≤L≤14：覆盖700个连续闭区间及6597个次数分支，包含所有取整跳变，上尾不超过10e⁻⁴，即0.183157以下。
5. L≥14：重新计算已有C70证明的260个连续区间及无穷半线界，联合失败小于0.0009。

小L区间的下偏差另外严格控制在1.657×10⁻²⁷以下。将上、下偏差正确合并，再核对403/404及L=14交界，得到全域联合失败<0.331。计算程序检查的是完整区间上的不等式；随机模拟和优化器返回值没有被当成证明。

证明与代码现在是否完整可靠
在上述iid泊松模型和固定算法定义下，C70L1的充分性证明已形成完整链条，并有可复现的计算机辅助证书。小范围与中范围分别做了独立重放；中范围还使用两种不同的多项式非负性方法和不同的系数实现。验证器拒绝缺失的n、S或尺度；大范围同时用有理区间与独立Arb公式复核。
这是通常意义的计算机辅助数学证明，仍依赖解析论证、验证器实现及区间算术库的正确性；没有声称已经通过Lean/Coq等形式化系统证明。审阅者可以逐条审阅文字推导并重放所有证书。
实现方面，实数均值、J/k取整、交替系数、校正、ceil及截断已经交叉验证。实际程序有数值及资源边界，超出能力会明确拒绝；定理的无限n范围不意味着有限机器能执行全部输入。

现有实验应怎样解释
主实验使用MidFish、LTRM、NRMN、EPA NRSA、BBS五个来源的40个固定经验分布，640格×2000次，共128万条主泊松路径。完整算法78次联合失败，最坏单格1.10%，640格的95%同时上界2.27%，与新的分布无关理论保证相容。
259098条路径获得合法且更紧的整数下界；另有17次错误抬高，以及61次零假设下错误拒绝。相同大预算的纯计数原本全部成功，所以这些下界提高不能称为成功率提高。所有78条失败已高精度逐条确认。
额外的1%理论风险计数基线也跑了128万次，35次失败，最坏0.45%，同时95%上界1.32%；C70预算仍是它的8.09–10.59倍。新证明不会改变这一效率比较。算法给出更紧下界的价值，应与额外采样成本和上偏差一并报告。

仍然没有覆盖的结论
• 现场生态总体：当前真值是清理后的经验分布；重采样保证不自动消除观测相关、探测偏差和未观测物种问题。
• 大类别数实证：源记录约979万，但S只有26–603、n只有1–1206；这些数据不能说明大n渐近区间的实际优势。
• 最优常数：本次证明70足够，不要求也不宣称69.006…是这套算法的精确最小常数。
• 发表竞争力：理论充分性、实现一致性和当前经验验证已闭环；相对强基线的优势区间、更多独立数据和明确新意仍需论证。因此仍不能仅凭本轮材料判定可以作为高质量CCF-A论文录用。

建议把后续实验重点放在相同风险下的样本预算—下界紧度曲线、更大类别数与稀有类尾部、独立调查或时间留出验证。继续堆叠同类重采样次数的价值小于明确展示算法在哪些规模与分布上值得额外成本。

复现（无需重新搜索证明参数）
cd /home/snowy/OSST
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proof_l01.verify_all

旧results_real_l01_20261104/theory_review文件记录的是拿到摘要时的审查状态；其中“小n未补齐”的结论已由本次自主证明更新。原实验数据和数值结果保持不变。
'''
    (OUT/'reliability_report.txt').write_text(body)
    paragraphs=body.split('\n\n')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C70L1证明与可靠性复审</title><style>body{max-width:980px;margin:36px auto;padding:0 22px;background:#fafbfc;color:#233348;font:16px/1.8 system-ui,sans-serif}h1{font-size:28px}p{white-space:pre-line}a{color:#165f92}.lead{background:#eaf4ee;border-left:5px solid #3f7855;padding:18px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf1f4}code{background:#edf1f4;padding:3px}</style>'''
    content+='<h1>'+html.escape(paragraphs[0])+'</h1><p class="lead">'+html.escape(paragraphs[1])+'</p>'
    content+='<p><a href="C70_L1_complete_proof.pdf">完整证明PDF（含既有大范围证明附录）</a> · <a href="C70_L1_proof.pdf">本次新证明PDF</a> · <a href="C70_L1_proof.tex">LaTeX源文</a> · <a href="C70_L1_certificate_bundle.zip">证明与证书复现包</a> · <a href="summary.json">总证书</a></p>'
    for p in paragraphs[2:]:
        content+='<p>'+html.escape(p)+'</p>'
    content+='<p><a href="small_audit.json">小n独立审计</a> · <a href="midrange_independent_audit.json">连续区间独立审计</a> · <a href="../results_real_l01_20261104/report.html">真实数据实验报告</a> · <a href="reliability_report.txt">纯文本复审</a></p></html>'
    (OUT/'reliability_report.html').write_text(content)
    print('Wrote reliability report; uniform failure bound',summary['failure_bound_rational'])


if __name__=='__main__':main()
