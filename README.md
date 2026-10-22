这个目录把《Optimal Support Size Testing》中的公式、可执行的有限样本方案和鱼类数据实验放在一起。论文原文保留为 `2610.18723v1.pdf`；抽取的文本在 `artifacts/paper.txt`。当前代码是依据 PDF 和分析编写的实现。

**最新投稿前审查：理论适用范围、原始数据与论文整合。** 入口为 [完整中文审查报告](results_submission_audit/review_report.html)、[理论边界与补充证明](results_submission_audit/theory_scope.pdf)、[英文实验章节预览](results_submission_audit/paper_insert_preview.pdf)、[可整合的 LaTeX 源码](results_submission_audit/paper_insert.tex) 和 [整合说明](results_submission_audit/INTEGRATION.txt)。此前的新证明专项复核保存在 `results_proof_review/`。审查列出 27 项问题及修订建议；以下历史报告应结合这次审查阅读。

原论文的匹配下界需要明确 ε 的适用范围，不能对整个 ε∈(0,1) 作统一最优性断言；新常数需要连同新的系数与集中界证明一起整合。当前 `(70,14)` 的 1,994,000 次试验中，承诺范围内的二元检验有 1,792,000 次；同预算计数消融同样没有错误，多项式的已观察收益是提高支持下界，尚未展示额外的二元判定优势。补充证明还给出：在 n≤10¹²、ε∈(0,1) 和 C₀=70 的预算下，纯计数的统一误差界已小于 1.863×10⁻¹²。零观测错误不能代替最坏情形理论保证，自然经验总体的重采样也不等同于新的野外独立观测。英文章节和整合说明明确保留这些限制；本仓库材料尚不代表论文已提交或通过同行评审。

**2026-10-22 后续探索：推荐 `(C₀,L₀)=(70,14)`，单次运行；整数 C₀ 的上下界已对齐。** 见 [新的中文比较报告](results_parameter_search/report.html)、[充分界与必要界证明 PDF](results_parameter_search/proof.pdf)、[结论摘要](results_parameter_search/conclusions.json) 和 [复现说明](results_parameter_search/REPRODUCE.txt)。新的集中不等式分析证明 `(70,14)` 单次失败概率小于 **0.09%**，`(70,16)` 小于 **0.0057%**，因此达到 1% 不再需要之前依据旧界给出的 27 次或 5 次重复。与 C₀=1025 相比，期望样本量减少约 93.17%；这不是相同比例的 CPU 加速结论。

新的渐近反例把必要常数夹在 **69.00627138～69.00627139**；在保持原统计量、固定有限 L₀>1、要求所有 n 和 ε 统一可靠的范围内，C₀≤69 不可行，故 **70 是最小可行整数 C₀**。固定奇数次独立多数表决也不能消除该反例。充分实数常数仍为 144(ln 2)²≈69.185234，与必要下界相差不足 0.26%。这不是所有算法的信息论最优结论；**L₀=14 已证明充分，但没有证明最小**。只需二元判定时，保留 `(70,16)` 可减少中间规模的多项式计算；需要更早提高支持下界时，`(70,14)` 从 n=442,414 开始启用多项式。

本轮新增 5,116,000 条独立模拟路径、8,848,000 次配对方法评估，覆盖十个自然经验总体、构造数据、切换边界及困难样例。`(70,14)` 的 1,994,000 次试验未观察到检验错误或下界失效；C₀=60 的困难样例错误率达到 6.35%。充分界通过 32,752 项精确算术检查，独立公式/原始数据审计通过 60,725 项检查；必要界另有精确级数证书和 Arb 交叉核验。推导及证书不是证明助手形式化或外部同行评审。源码在 `parameter_search/`，原有 `results_c7016/` 保持为历史实验记录。

**2026-10-22：C₀=70、L₀=16 已接入并完成自然/构造数据实验。** 入口为 [中文实验报告](results_c7016/report.html)、[英文 PDF](results_c7016/experiment_report.pdf)、[结果摘要](results_c7016/summary.json) 和 [完整复现说明](results_c7016/REPRODUCE.txt)。独立实现位于 `c7016/algorithm.py`，保留证明中的实数泊松均值、全部尺度和完整阶数；不通过修改旧版默认常数套用新定理。

主实验覆盖 756 个构造配置、10 个自然经验总体的 236 个配置，以及 120 个保留自然丰度形状的扩展构造配置。新算法共运行 4,208,000 次，其中 1,672,000 次实际进入多项式分支；未观察到承诺检验错误或支持下界失效。三套实验最坏单元的同时 95% 检验错误率上界分别约为 0.2818%、0.2441%、0.4660%。自然数据主实验均为计数分支；扩展类别的结果明确属于合成数据。

相同预算的计数消融也没有承诺检验错误，多项式主要在 554,000 次试验中提高了下界。构造网格上，新单次方案仍比 99% 保证的计数基线多用约 2.44–8.39 倍样本。历史实验按旧误差界将 `(70,16)` 独立运行 27 次，总期望系数为 1890，高于旧版 1025；该重复实验仍保留，但顶部的新证明已经取消达到 1% 时的这项重复开销。自然数据有 42/236 个配置的原始记录数达到平均预算，人工复制观测的相关性压力测试最坏漏报率为 100%。

完整数值接口、原始频数、精确真值、四个独立种子块、置信区间、代码/数据快照、复核结果均保留。125 项回归测试通过；18 次逐类别泊松计数的完整接口核验包含 12 次多项式分支。所有二元判定与下界覆盖事件经过数值认证，部分精确整数下界保存为认证区间。精确理论证书另已从头重算。重复实验路径数的一次纯统计口径修订记录在 `results_c7016/protocol.json`，初版协议和修改前源码保留，未改变任何原始试验或判定。

```bash
.venv/bin/python -m c7016.algorithm plan --n 3269018 --epsilon 0.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.run --phase all --out results_c7016_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.diagnostics --out results_c7016_reproduced
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.validate --out results_c7016_reproduced --proof-and-tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c7016.report --out results_c7016_reproduced
```

**前版：C₀、L₀ 联合优化的完整证明与精确算术证书。** 入口为 [证明 PDF](artifacts/joint_constants_submission/submission_proof.pdf)、[LaTeX 主文件](artifacts/joint_constants_submission/submission_proof.tex)、[算术附录](artifacts/joint_constants_submission/arithmetic_appendix.tex)、[中文说明](artifacts/joint_constants_submission/提交说明.txt) 和 [完整证明包](artifacts/osst_publication_proof.zip)。其参数、系数与期望引理继续用于顶部的新证明；以下误差界为前版较松的充分界。

证明覆盖有限或可数离散分布、所有 n≥1 和 ε∈(0,1)，在文中规定的独立同分布泊松采样及算法条件下成立。C₀=144(ln 2)²≈69.185234 配合 L₀=16 时，单次失败概率严格小于 0.317480；整数方案 (70,16) 的上界为 0.282612，两者成功概率均超过 2/3。(70,20) 的单次失败概率严格小于 0.072224。独立重复与多数表决/中位数放大见证明。这里给出的是充分参数，不声称全局最小，也没有认证向下取整为 69。

精确证书通过 428 项义务检查，独立审计通过 2,491 项检查；验证只使用 Python 标准库的整数和有理数运算。这是数学证明及可复算证书，不是 Lean/Coq 内核形式化，也不代表会议录用保证。

```bash
cd artifacts/joint_constants_submission
python3 exact_certificate.py
python3 -O audit_exact.py
# 安装 Tectonic 后重建 PDF：
tectonic submission_proof.tex
```

**实现与实验版本边界：** 当前 `paper_reference.py` 仍默认 C₀=2²⁶、L₀=1024；现有 C₀=1025 实验也保留 L₀=1024。(70,16) 使用新的 `c7016` 接口与 `results_c7016/`，(70,20) 仍只有证明结果，尚未重跑对应实验。不能仅给旧接口传入 `--c0 70` 就视为运行新定理。不同版本的结果、源码快照和证明分别保留。仓库排除虚拟环境、编译工具缓存及大型跨领域下载/清洗数据；后者可按下方复现命令重新准备，数据来源和摘要保留。

**2026-10-21：C₀=1025 的独立完整实验。** 新入口为 [中文结果与读图说明](results_c1025/report.html)、[实验论文及完整证明 PDF](results_c1025/experiment_paper.pdf)、[LaTeX 主文件](results_c1025/experiment_paper.tex) 和 [复现说明](results_c1025/REPRODUCE.txt)。主实验覆盖 n=1 至 10¹²、五个 epsilon、最多十五种分布场景，以及七类生物的十个公开经验总体。共 9,130,000 条独立模拟路径、39,720,000 次配对方法评估；方法评估次数不等于独立试验次数。完整参考接口执行 1,428 次，代数核验通过 43,801 项，回归测试 112 项通过。

严格保持 L₀=1024，因此现实规模只进入计数分支。C₀=1025 在主网格没有观察到承诺检验错误，同时 95% 错误率上界约为合成 0.107%、自然经验分布 0.0955%。但计数基线仍少用约 35.8–135.2 倍样本；236 个生态配置中只有 2 个原始记录数达到新系数的平均预算。实验稿保留这些限制和成组相关性的失败结果，不把经验分布重采样当作未知自然总体的验证。

```bash
.venv/bin/python paper_reference.py demo --n 1000 --epsilon 0.1 --c0 1025
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m c1025.run --phase all --out results_c1025_reproduced
.venv/bin/python -m c1025.supplement --out results_c1025_reproduced
.venv/bin/python -m c1025.numeric_extra --out results_c1025_reproduced
.venv/bin/python -m c1025.report --out results_c1025_reproduced
```

`paper_reference.py` 的默认系数仍是原版 2²⁶，新常数显式通过 `--c0 1025` 或函数参数 `c0=1025` 传入。原有实验与结果目录保留。

**2026-10-19：完整后续实验与英文论文稿。** 新实验单独保存在 `results_publication/`。入口为 [中文报告](results_publication/report.html)、[英文论文 PDF](results_publication/manuscript.pdf)、[英文 HTML](results_publication/manuscript.html) 和 [LaTeX 源码](results_publication/manuscript.tex)。新增小 epsilon/规模网格、公平调参的多尺度/单尺度/FH 形状对照、逐条停止的强计数基线、独立困难分布复测、有效支持集两侧指标、自然分组及留出数据、置信水平/参数敏感性和运行开销实验。共 69,896,000 次方法—试次评估（不同方法共享样本），101 项测试通过。原始逐次输出、冻结协议、精确边界证书和源码快照全部保留。

在实际标签流的测试网格上，多尺度组合方法相对立即停止的固定预算计数基线，Yes / No 场景平均节省约 31.2% / 14.9% 的样本。多尺度并非所有参数下最优：28 组参数中，单尺度的通过预算有 7 组更小、14 组相同、7 组更大；认证 FH 形状版本也在部分小 epsilon 配置中胜出。论文稿保留这些负面结果，不把有限网格表现写成新的最优阶证明或 CCF-A 录用承诺。

一次名义边界的浮点标签问题已通过精确有理数认证修正，并在保持原开发参数不变的条件下重跑全部正式实验。修订说明见 `results_publication/design.json`、`boundary_correction_audit.json`；旧输出保存在 `results_publication_superseded_boundary_v1/`，没有混入正式结果。

```bash
# 按独立目录完整复现；具体数据准备与版本要求见 REPRODUCE.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m publication.run \
  --phase all --out results_reproduction
# 仅从现有结果重建图表和论文稿
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m publication.run --phase report
```

[完整复现说明](results_publication/REPRODUCE.txt)；[代码、结果、图表和论文压缩包](artifacts/osst_publication_artifact.zip)。压缩包不包含跨领域原始或派生数据；按发布者的许可从固定来源重新获取。

**新增：同时控制误报与漏报的实用组合方法。** [guaranteed.py](guaranteed.py) 保留论文的多项式构造，用认证的阈值提前判断 No；用独立新样本验证 Yes；到最终预算时用计数检验收尾。在固定分布 iid、预定参数以及论文的 Yes/No 承诺下，两种错误分别不超过 `delta`（默认 5%），不需要最小非零概率假设。证明、文献调研、独立实验与消融见 [新方法报告](results_guaranteed/report.html)。

```bash
cd /home/snowy/OSST
# 使用独立开发集选定的参数；原版与旧工程版入口继续保留
.venv/bin/python guaranteed.py plan --n 1000 --epsilon 0.1 --delta 0.05 \
  --config results_guaranteed/selected_config.json
.venv/bin/python guaranteed.py demo --n 1000 --epsilon 0.1 \
  --config results_guaranteed/selected_config.json
# 完整重跑：独立开发集、冻结协议、每正式单元 1000 次及相关性压力测试
.venv/bin/python guaranteed_experiments.py --trials 1000 --development-trials 100
# 逐条输入流：样本不足时输出 Inconclusive，验证样本不重复充当训练样本
.venv/bin/python guaranteed.py csv --file data/example_300_fish.csv \
  --n 8 --epsilon 0.1 --column species
```

这是新的组合算法。它的最坏期望样本界为 `O((n+log(1/delta))/epsilon)`；没有证明缩小常数后仍保留论文的最优阶。正式实验中，验证开销计入样本成本；每个统计 No 都通过整数证书核验；尾概率和参数前提由 Arb 区间计算认证。真实 CSV 是否 iid 仍取决于采样设计，经验分布回测也不等于确定了自然总体的全部类别。不给 `--config` 时使用内置的保守默认值，保证不变，但不应把它的性能与开发集选定配置混为一谈。

**论文对应性审计后的结论：旧大规模实验主要运行工程修正版，不是第 3 节原算法。** 切比雪夫构造相同，但样本预算、阶数和误差修正有改变。严格原版已独立实现在 [paper_reference.py](paper_reference.py)，详细公式表及实际函数调用清单见 [核对报告](results_audit/report.html)。旧实验数值没有改挂到原算法名下；旧协议所对应的源码快照保存在 `results_audit/before_audit/`。

要运行与论文公式对应的版本：

```bash
cd /home/snowy/OSST
.venv/bin/python paper_reference.py plan --n 1000 --epsilon 0.1
.venv/bin/python paper_reference.py demo --n 1000 --epsilon 0.1
# 重跑独立公式测试与函数调用核对，生成 results_audit/report.html
.venv/bin/python audit_paper.py
```

严格版保留 `C0=2^26, L0=1024`、全部尺度和原阶数，不用 Hoeffding 替换原修正。使用 Arb 区间算术检查 `m/J/k` 的取整及最终判定；精度或资源不足时明确报错，不降低参数或猜测结果。`demo` 仅对已知有限分布等价生成泊松类别次数，随机发生器采用双精度；现实采样仍需要论文预算。对真实泊松数据，可提供频数 JSON `{"1":F1,"2":F2,...}`，运行 `paper_reference.py test --fingerprint 文件 --n 1000 --epsilon 0.1 --m 84864619140`；采样模型需由数据采集过程满足，程序无法仅从频数验证。

扩展实验已完成：五套公开数据共 9,893,928 次有效观测，合成支持集最大 100,000,001 类，正式实验包含 1,344 个总体与预算组合、672,000 批抽样和 3,360,000 次方法判定。中文报告、完整结果筛选器和图见 [大规模实验报告](results_large/report.html)，评测前固定的参数与代码哈希见 [协议](results_large/protocol.json)。

旧版小规模实验和鱼类分析保留在 [原实验报告](REPORT.md) 与 `results/`；它们的参数与数值没有被新版结果覆盖。

**运行完整的大规模实验**

```bash
cd /home/snowy/OSST
.venv/bin/python run_large_benchmark.py \
  --trials 500 --calibration-trials 200 --stress-trials 200 --seed 20261017
```

这个入口自动下载固定版本数据、校验 SHA256、清洗、测试、独立校准、冻结评测协议、运行合成及真实频谱实验、做相关性压力实验、生成 PNG/PDF 和自包含 HTML 报告，最后打包代码和结果。当前环境在已有数据缓存时整套实验约 3–4 分钟，其他机器时间会不同。新环境先执行 `python -m pip install -r requirements.txt`。数据存于 `data/large/`，结果存于 `results_large/`，程序和结果包为 `artifacts/osst_large_benchmark.zip`。分发包不包含跨领域原始或派生数据，复现时从发布者下载。

真实数据包括 UCI NIPS 和 Enron 词频、MovieLens 1M 评分、UCI Online Retail 销售行和 vegan BCI 大树普查。所有总体使用正频数类别定义支持集；单次词语、评分、销售行、树木分别对应一次抽样单位，不将商品数量当作独立交易，不拼接不同年份的森林普查。原始记录中的相关性只在单独的压力实验中处理；正式实验从完整经验频数定义的分布独立有放回采样，结论针对该有限参考分布。

`benchmark_core.py` 以等概率类别分组精确模拟 `(F0,...,FK,F>K)`，因此能测试一亿类别而无需枚举一亿个标签。高频箱只保留类别数，不能从它伪造实际总样本数 M；权重始终使用预设 m。`score_profiles` 只接收观测统计量与预设参数，真实概率只供抽样和评测真值使用。

有限样本修正版有 iid 泊松模型下的误报控制，但不因缩小常数而自动获得论文的最坏情形漏报保证。默认正式合成实验在 `C=4` 的预算下最大实测误报为 0%、漏报为 0.6%；全部对应单元的同时单侧置信上界不超过 3.71%。这是完整测试网格上的证据。简单缩常数的版本和 Chao1 阈值化存在严重误报，不能仅看它们的检出率。具体比较、失败场景、独立校准和置信区间均保留在报告和 CSV 中。

也可分阶段运行：

```bash
.venv/bin/python run_large_benchmark.py --phase prepare
.venv/bin/python run_large_benchmark.py --phase calibrate
.venv/bin/python run_large_benchmark.py --phase evaluate
# 仅重建图表、报告和结果包，需要已有结果和数据缓存
.venv/bin/python run_large_benchmark.py --phase report
# 仅重建 HTML，不重复模拟或绘图
.venv/bin/python report_large.py
.venv/bin/python -m pytest -q
```

需要保留另一组运行时用 `--out results_large_other`；HTML 报告从该次 CSV 动态生成。README 中的固定数值只对应上述默认配置。原始数据的来源、许可、清洗规则和剔除数量见 `data/large/sources.json`、`data/large/dataset_summary.json`。

**运行原小规模与鱼类实验**

当前目录已经创建好 `.venv`，可以执行：

```bash
cd /home/snowy/OSST

# 原始流程：公式测试、小规模模拟、鱼类回测、CSV 示例和压缩包
.venv/bin/python run_all.py --trials 500 --seed 20261016

# 论文参数，m=84,864,619,140；只规划，不生成样本标签
.venv/bin/python support_size_demo.py --paper-plan --n 1000 --epsilon 0.1

# 泊松方案：保证针对 iid 的承诺检验；m=10,537
.venv/bin/python support_size_demo.py --baseline-plan --n 1000 --epsilon 0.1 \
  --failure-prob 0.05 --sampling poisson

# 固定样本方案：所需样本数为 10,509
.venv/bin/python support_size_demo.py --baseline-plan --n 1000 --epsilon 0.1 \
  --failure-prob 0.05 --sampling fixed

# 合成分布实验：9 种分布，3 个预算，每种重复 500 次
.venv/bin/python support_size_demo.py --demo --trials 500 --seed 20261016

# 真实鱼类数据回测，固定版本原始数据已保存在 data/
.venv/bin/python support_size_demo.py --fish-demo --trials 500 --seed 20261016

# 公式、概率界、数据清洗和边界行为测试
.venv/bin/python support_size_demo.py --self-test

# 读取一个实际的 300 条捕获记录子样本
.venv/bin/python support_size_demo.py --csv data/example_300_fish.csv \
  --n 5 --epsilon 0.01

# 同一份固定样本，做一次正确的泊松子抽样并运行多项式
.venv/bin/python support_size_demo.py --csv data/example_300_fish.csv \
  --n 8 --epsilon 0.01 --poissonize-fixed --seed 20261016
```

新环境使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-lock.txt` 记录本次运行的实际版本。`pypdf` 只用于核对 PDF，不是算法运行依赖。图表使用英文标注，以便缺少中文字体的环境也能直接运行；源代码和说明使用中文。

**文件和接口**

| 文件 | 作用 |
|---|---|
| `support_size.py` | 频数直方图、论文计划、切比雪夫递推、多尺度统计量、尾概率基线、Chao1 |
| `paper_reference.py` | 第 3 节原算法；原参数、任意精度区间运算及输出认证 |
| `test_paper_reference.py` | 独立有理数闭式系数、原参数、边界和原版输出核对 |
| `audit_paper.py` | 公式对应表、全函数清单、真实调用记录与审计报告 |
| `support_size_demo.py` | 命令行入口，兼容用户分析中的主要开关 |
| `experiments.py` | 合成实验、逐鱼不放回回测、事件分组回测、经验分布泊松实验 |
| `fish_data.py` | 固定 Git 版本下载、SHA256 校验、逐条 CSV 输入、鱼类清洗 |
| `test_support_size.py` | 关键数学恒等式和数据处理测试 |
| `run_all.py` | 一条命令完整重跑，保存示例 JSON、测试日志和 SHA256 清单实验包 |
| `data/provenance.json` | 数据来源、版本、哈希、清洗与采样限制 |

逐条输入 CSV 至少有 `species` 一列。也可以输入 `species,count`，并加 `--count-column count`。count 必须是个体数；捕捞重量、单位努力捕获量 CPUE、物种名录的记录数都不能直接当作抽样次数。可用 `--species-column` 指定其他列名。

`--sampling fixed` 为默认模式，不使用依赖泊松独立性的多项式保证。`--sampling poisson --m 300` 表示采样前就确定了均值 300、实际总数 M 服从该泊松模型；程序不会自动把观测到的 M 填进 m。不能把一份恰好 300 行的文件改个开关就宣称其经过了泊松化。

已有固定 N 条记录时，可以用 `--poissonize-fixed`：程序事先依据 N 选 m，使 `Pr(Poi(m)>N)<=0.005`，独立抽取 M，再从池中随机选 M 条。M>N 时返回 `overflow`，保留确定性的直接观察下界，并将多项式结论标为样本不足；不会截断 M 或重新抽到满意为止。如果原始 N 条是固定预算的 iid 样本，这与从无限 iid 序列读取前 M 条的过程可耦合，整个运行过程保留单侧误报控制。不能仅筛选成功运行来重新宣称未经调整的条件错误率。

子抽样不能把有偏或相关的现场捕获变成 iid；真实数据的输出仍是描述性的。`fish_polynomial_lower_bounds.csv` 单独记录这一适配器在实际记录子样本上的表现，预设上限 n=20、epsilon=0.01，不使用已知真实种数来挑选上限。

`--assume-iid` 只是用户对采样设计的声明，程序无法验证鱼群是否独立、平稳或有渔具偏好。未声明时，真实 CSV 的输出明确标为描述性分析。即使声明，保证也只在该假设确实成立时适用。先筛选同一采样总体，再运行；不要混合不同年份、湖泊、渔具后称其为全湖丰度分布。

计数检验的 `No` 表示已经看到了超过 n 种；`Yes` 只有在预算满足设定的漏报上界时返回，否则返回 `Inconclusive`。这是承诺检验的 Yes：不能据此排除 `S>n` 但 `Delta_n<epsilon` 的中间区域。输出的漏报界是 No 分布下的事前错误概率上界，不是当前结论的后验概率。

**论文公式的实现**

设 `F[l]` 是出现 l 次的类别数，`D=sum(F.values())`。每个尺度的原始统计量为

\[
 Z_j=D+\sum_{\ell=1}^{k_j}F_\ell(f_j(\ell)-1).
\]

代码使用无量纲变量 `A=m*a, B=m*b`，以 80 位 Decimal 精度递推

\[
u_{t+1,\ell}=2\alpha u_{t,\ell}-2\beta\ell u_{t,\ell-1}-u_{t-1,\ell},
\qquad \alpha=\frac{B+A}{B-A},\quad\beta=\frac2{B-A},
\]

并计算 `f(l)=1-u[k,l]/u[k,0]`。`f(0)=0`、`f(l>k)=1`。最后用 `math.fsum` 聚合。

`paper_statistic` 实现原文的 `max(D, max_j (Z_j-delta_j*n)/(1+delta_j))` 这一修正形式，但旧版实验向它传入缩小后的参数，只用于 naive 对照，不能视为完整原算法。严格入口使用 `reference_parameters → reference_statistic → reference_test`；两者共用 `paper_correction`，避免修正公式漂移。`clipped_paper_lower_bound` 单独实现原文的整数输出并由严格入口调用，不能拿它再与 n 比较来拒绝。

`paper_plan` 现在委托严格版区间参数计算，使用原文的 `C0=2^26, L0=1024`；在现实规模 n 下进入直接计数分支。测试以压缩频数实际执行 `n=10^445` 的全部三个多项式尺度，而不只是打印参数。这是确定性公式核对，不是在现实中采集如此多的类别。`make_scales(mode="paper")` 不再接受任意的小预算；大 L 分支明确要求使用严格入口，不转为浮点近似。

用 `N_i ~ Poisson(m*p_i)` 直接生成类别计数，与先生成 `M~Poisson(m)` 再生成 M 条独立标签同分布。采样成本由模拟支持集 S 决定；排序聚合最坏 `O(S log S)`、空间 `O(S)`。因此，即使 m 约为 849 亿，已知 1,000 类的模拟也能很快完成。**这是模拟加速，真实观测仍需要获取对应的样本，不能把计算加速当作样本复杂度降低。**

**有限样本下如何缩小实际预算**

计数基线可将 Fact 2.2 中的 Markov 界进一步收紧。在 No 情形下，任意至多 n 个已见类别之外都至少还有 epsilon 质量。因此，在收集到第 n+1 种之前，每次采样发现新种的条件概率至少为 epsilon。

用独立均匀随机数耦合发现过程与成功概率 epsilon 的 Bernoulli 过程：在尚未超过 n 种时，每个 Bernoulli 成功都对应一次新类别发现。故固定 M 次采样时

\[
 \Pr(D\le n)\le \Pr(\operatorname{Bin}(M,\varepsilon)\le n).
\]

对独立的 `M~Poisson(m)` 混合，利用泊松稀疏化得到

\[
 \boxed{\Pr(D\le n)\le \Pr(\operatorname{Poi}(m\varepsilon)\le n).}
\]

`count_plan` 二分求出尾概率不超过 alpha 的最小整数预算。Yes 情形下 D 不可能大于 S，故误报恒为 0。这是基于论文 Fact 2.2 的发现过程思想额外推导的有限样本改进，并非声称原文给出了这些新数值。

当 `n=1000, epsilon=.1, alpha=.05`，原 Markov 方案为 200,200，改进泊松方案为 **10,537**，固定样本方案为 **10,509**。常数误差率下该基线仍为 `O(n/epsilon)`，没有取得论文的渐近对数优势。

若实地采样，可以事先按方案设定预算，见到第 n+1 种即提前结束并拒绝。不要在每个时刻重算无校正的多项式阈值并反复检验；那样不继承下述单次误报保证。

**多项式的有限样本修正**

简单缩小常数容易产生大权重、振荡和严重误报。实验保留一个 `naive` 对照：`B=log(en)`、最多 8 阶，却继续使用原文 delta 的修正。其误报已在结果中显现，不能用作可靠检验。

`practical` 模式仍用 `a_j=2^j epsilon/(8n)`，默认 `B=min(log(en),m)`，但作了以下处理：

1. 用实际的 `rho_j=1/T_k((B+A)/(B-A))`，不假定截断阶数后仍有 `rho<=delta`。
2. 用 `ceil(acosh(1/delta)/acosh((B+A)/(B-A)))` 求逼近目标需要的阶数，最多 12 阶；根据权重幅度再降阶，使 `max|f|<=20`。
3. 同时要求 `k<=sqrt(B(B-A))` 和 `B<=m`，保持论文 Lemma 2.3 的期望上界条件。参数在看数据前固定。
4. 用实际权重的范围宽度 `R_j=max{0,1,f_j(1),...}-min{0,1,f_j(1),...}` 控制随机误差。

设一共 K 个尺度。S<=n 时可补入贡献恒为 0 的虚拟类别，使项数为 n。各类别的泊松计数独立，Lemma 2.3 给出 `E Z_j <= (1+rho_j) S`。对这些有界独立项应用 [Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830)，并对 K 个尺度取联合界：

\[
t_j=R_j\sqrt{\frac n2\log\frac K\alpha},\qquad
L=\max\left\{D,\max_j\frac{Z_j-t_j}{1+\rho_j}\right\}.
\]

于是 **S<=n 时 `Pr(L>n)<=alpha`**；各尺度共用样本不影响联合界。`max(D, min(n, max(1,ceil(L))))` 还是一个置信度至少 `1-alpha` 的支持集下界：当 S<=n 时上述事件同时给出 L<=S；当 S>n 时截断部分天然小于 S，而 D 也小于 S。

这个修正保证误报和下界覆盖，**没有在缩小预算下证明统一的漏报率，也没有保留原论文有效支持集下界的双侧保证**。因此实用多项式接口返回 `No` 或 `Inconclusive`。实验表中的 `failure_to_reject` 统计 No 分布上未能拒绝的比例，不能理解成已经有概率保证的 Yes。

以上工程修正版的误报结论是精确算术中的数学保证。该修正版仍使用高精度系数和双精度聚合，没有建立区间舍入误差认证；新增的 `paper_reference.py` 原版入口才使用区间计算。它们的适用保证不能互相替代。

完整运行后，`artifacts/osst_experiment_bundle.zip` 包含代码、文档、论文、固定版本数据和结果，排除了 `.venv` 及临时下载工具。包内 `MANIFEST.sha256` 可用于逐文件校验，包旁也保存了压缩包的 SHA256。解压后按安装说明建环境即可重跑。更改 `--seed` 或 `--trials` 后，以对应生成的 CSV/JSON 为准；`REPORT.md` 保留默认实验的记录。

**鱼类数据与估计目标**

原始来源是 [Derek H. Ogle 的 InchLake1](https://fishr-core-team.github.io/FSAdata/reference/InchLake1.html)，记录 2007 和 2008 年 5 月的捕获个体，含渔具、年份、捕获事件和物种。下载脚本固定 Git commit，并核对 SHA256；原始文件按上游 GPL (>=2) 的包许可保留署名。

清洗仅去掉 7 条 `Rusty Crayfish`，剩 **4,887 条鱼类记录、记录内共 9 种鱼**。不因某些个体体长相同而去重。CSV 没有逐尾时间戳，`record_id` 只是源文件行号，不能据此声称知道每次上钩的时间顺序。

物种数点估计增加了 Chao1 作为生态学对照：

\[
\widehat S_{\mathrm{Chao1,BC}}=D+\frac{F_1(F_1-1)}{2(F_2+1)}.
\]

这是偏差修正的点估计，不是保证不超过真值的置信下界。稀有种完全未入样时仍可能明显低估，单次样本也可能高估。依据可参见 [Chiu & Chao 的相关推导和讨论](https://pmc.ncbi.nlm.nih.gov/articles/PMC4741086/)。

鱼类实验分为三种，输出文件明确标记采样方式：

| 实验 | 输出 | 可支持的结论 |
|---|---|---|
| 从同年同渔具记录随机不放回抽取 M 条 | `fish_subsample_results.csv` | 能否恢复这组参考记录中的物种数；不套用泊松保证 |
| 按完整捕获事件划分，约一半事件作训练数据 | `fish_event_holdout.csv` | 保留事件聚集结构后，恢复完整参考记录的效果 |
| 将完整组内频率当作已知 p，再独立泊松抽样 | `fish_testing_results.csv` | 在来自真实频率形状的经验分布上检查算法；不是新的现场捕获 |

前两种的对照真值仅是参考数据内已记录的种数。第三种测试中 S 和 Delta_n 也仅针对经验分布；Gap 行不计算错误率。所有阈值的结果用于分别评估，没有将多次检验合成一个未经校正的总体判断。

如果第 i 种鱼的丰度是 A_i、被所用渔具捕获的机会为 q_i，观测概率一般与 `A_i*q_i` 成比例。钓鱼无法探测的种类可能有 q_i=0。由此产生的偏差，无法用支持集检验公式自动消除。要验证全湖鱼种总数，还需要独立的全湖物种调查，以及覆盖不同栖息地、渔具和时间的采样设计。
