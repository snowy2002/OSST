#!/usr/bin/env python3
"""重跑公式测试及调用跟踪，生成论文到代码的审计表；不重写正式实验数值。

调用计数来自列明的审计工作负载，不把没有执行的下载/预处理分支称为死代码。
函数自身不靠测试结果调参。完整实验历史代码另保存在 before_audit。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CORE = {"paper_reference.py", "support_size.py", "benchmark_core.py"}


def formula_rows():
    return [
        ("Problem 1", "Yes: S≤n; No: Δ_n≥ε; Gap 不计错", "benchmark_core.py:Population.truth", "定义一致；真值仅在模拟评测中使用"),
        ("Theorem 1.2", "eff_ε=min{k: Δ_k≤ε}", "support_size.py:effective_support", "定义一致；不是未知总体的可观测输入"),
        ("§3", "L=log(en), C0=2^26, L0=2^10", "paper_reference.py:_plan_at_precision", "原常数；自然对数"),
        ("§3", "m=ceil(C0 n/(εL)); m≠实际 M", "paper_reference.py:reference_test", "认证取整并强制检查输入预算"),
        ("§3", "M~Poi(m); Ni 独立~Poi(mp_i)", "support_size.py:poisson_fingerprint", "已知模拟分布时等价生成次数；现实模型须由采样设计满足"),
        ("式 (5), §3", "D=Σ 1{Ni>0}; L<L0 时 tildeS=D", "paper_reference.py:reference_statistic", "原分支；不强行启用多项式"),
        ("§3", "a0=ε/(8n), b=64L/m, J=floor(log2(b/(2a0)))", "paper_reference.py:_plan_at_precision", "原式；J 取整有区间认证"),
        ("§3", "aj=2^j a0, δj=2^(-j-2), kj=ceil(√(b/aj) log(2/δj))", "paper_reference.py:_plan_at_precision", "遍历 j=0,…,J；不截断 kj"),
        ("式 (2)", "P(x)=-Tk((b+a-2x)/(b-a))/Tk((b+a)/(b-a))", "paper_reference.py:reference_weights", "代数等价的阶乘系数递推；独立有理数闭式展开核对"),
        ("式 (3)", "f(0)=0; f(l)=1+cl l!/m^l (1≤l≤k); f(l>k)=1", "paper_reference.py:reference_weights", "零次数不贡献，高次数贡献 1"),
        ("式 (4), §3", "Zj=Σi fj(Ni)=D+Σl Fl(fj(l)-1)", "paper_reference.py:reference_statistic", "按频数聚合是精确代数改写；不遗漏高次数类别"),
        ("式 (8)", "Q(x)=E[f(Poi(mx))]=1+exp(-mx)P(x)", "test_paper_reference.py:test_poisson_expectation_identity_in_interval_arithmetic", "证明恒等式，在独立测试中核对；检验时不需要已知 p"),
        ("Lemma 2.3/4.1", "0<a≤b/2≤1/2, √(B(B-A))≥k, ρ≤δ", "paper_reference.py:_plan_at_precision", "逐尺度认证参数前提；不靠容差放过失败"),
        ("§3", "tildeS=max(D,maxj (Zj-δj n)/(1+δj))", "support_size.py:paper_correction", "原修正，由 reference_statistic 聚合全部尺度"),
        ("§3", "tildeS≤n→Yes；tildeS>n→No", "paper_reference.py:reference_test", "用未截断统计量判定；边界无法认证则加精度或报错"),
        ("§3", "Shat=min(max(ceil(tildeS),1),n)", "support_size.py:clipped_paper_lower_bound", "单独数值输出；No 分支下代数化简为 n"),
        ("工程改动", "较小 m、B、受限 k、实际 ρ、Hoeffding 修正", "support_size.py:hoeffding_statistic", "不等同于第 3 节；旧大规模实验主要评测这一版本"),
        ("论文外对照", "计数尾概率基线、Chao1、固定池适配器", "support_size.py:count_plan/chao1/poissonized_fixed_test", "不是原论文算法步骤；保留用于比较与数据接入"),
    ]


def inventory():
    definitions, callsites = [], defaultdict(set)
    files = sorted(p for p in ROOT.glob("*.py") if p.name != "audit_paper.py")
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                if name:
                    callsites[name].add(f"{path.name}:{node.lineno}")
        if path.name.startswith("test_"):
            continue

        def visit(node, prefix=""):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix+child.name
                    definitions.append({"file": path.name, "function": name, "line": child.lineno,
                                        "doc": (ast.get_docstring(child) or "").split("\n")[0]})
                    visit(child, name+".<locals>.")
                elif isinstance(child, ast.ClassDef):
                    visit(child, prefix+child.name+".")
                else:
                    visit(child, prefix)
        visit(tree)
    return definitions, callsites


class Calls:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.phase = ""
        self.sources = {str(p): p.name for p in ROOT.glob("*.py") if p.name != "audit_paper.py"}

    def profile(self, frame, event, argument):
        if event == "call" and frame.f_code.co_filename in self.sources:
            key = (self.sources[frame.f_code.co_filename], frame.f_code.co_qualname)
            self.counts[key][self.phase] += 1

    def run(self, phase, callback):
        self.phase = phase
        previous = sys.getprofile()
        sys.setprofile(self.profile)
        try:
            return callback()
        finally:
            sys.setprofile(previous)


def workloads(out: Path, calls: Calls):
    import numpy as np
    import pytest
    from benchmark_core import synthetic_cases, Population, stable_rng
    from benchmark_data import DATA
    from large_experiments import evaluate, real_views
    from paper_reference import main as paper_main, reference_parameters, reference_test
    from support_size import Fingerprint
    from support_size_demo import main as demo_main

    reference_runs = []

    def original():
        for n in [1000, 10**445]:
            plan = reference_parameters(n, "0.1")
            for label, fp in [("empty", Fingerprint({})), ("many_singletons", Fingerprint({1: n})),
                              ("exact_n", Fingerprint({20: n})), ("exceeds_n", Fingerprint({20: n+1}))]:
                reference_runs.append({"case": label, "kind": "人为构造的频数，仅作确定性公式和边界检查",
                                       **reference_test(fp, n, "0.1", m=plan.m)})
        stream = io.StringIO()
        with redirect_stdout(stream):
            paper_main(["demo", "--n", "1000", "--epsilon", "0.1"])
        save_json(out / "original_parameter_demo.json", json.loads(stream.getvalue()))
    calls.run("paper_original", original)
    save_json(out / "reference_runs.json", reference_runs)

    def legacy():
        # 调用实际 CLI 和原实验入口；所有输出隔离到 audit/smoke，不改正式结果。
        for args in [["--demo", "--trials", "2"], ["--fish-demo", "--trials", "2"],
                     ["--paper-plan"], ["--baseline-plan"],
                     ["--csv", str(ROOT/"data/example_300_fish.csv"), "--n", "8", "--epsilon", ".01"],
                     ["--csv", str(ROOT/"data/example_300_fish.csv"), "--n", "8", "--epsilon", ".01", "--poissonize-fixed"],
                     ["--csv", str(ROOT/"data/example_300_fish.csv"), "--n", "8", "--epsilon", ".01", "--sampling", "poisson", "--m", "300"]]:
            # 最后一个 CSV 调用只核对分支，不宣称真实固定数据具有泊松模型。
            demo_main(args+["--out", str(out/"smoke")])
    with (out / "entrypoint_trace.log").open("w") as log, redirect_stdout(log):
        calls.run("legacy_entrypoints_smoke", legacy)

    def batch():
        config = json.loads((ROOT/"results_large/calibration.json").read_text())["selected_config"]
        rows = []
        for n, eps in [(1000, .1), (1000000, .05)]:
            for name, pop in synthetic_cases(n, eps):
                m = math.ceil(4*n/(eps*(1+math.log(n))))
                rows.extend(evaluate(pop, n, eps, m, 3, stable_rng(20261018, "audit", n, name),
                                     config, {"case": name, "audit_only": True}))
        for name in ["nips", "enron", "movielens", "retail", "bci"]:
            with np.load(DATA/"prepared"/f"{name}_counts.npz") as data:
                pop = Population.from_counts(data["counts"])
            for view, current, n in real_views(pop, .1):
                m = math.ceil(4*n/(.1*(1+math.log(n))))
                rows.extend(evaluate(current, n, .1, m, 3, stable_rng(20261018, "audit", name, view),
                                     config, {"case": name+"/"+view, "audit_only": True}))
        save_json(out / "batch_trace_cases.json", {"note": "每单元仅 3 次，用于调用跟踪，不能当作性能评测结论。", "rows": rows})
    calls.run("batch_evaluate_smoke", batch)

    with (out/"tests.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        result = calls.run("tests", lambda: pytest.main([str(ROOT/filename) for filename in
                           ["test_support_size.py", "test_benchmark.py", "test_paper_reference.py"]]+["-q"]))
    if result:
        raise RuntimeError("公式测试失败，见 results_audit/tests.log")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    import csv
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(out, summary, functions):
    def table(headers, rows, identifier=""):
        return (f'<table id="{identifier}"><thead><tr>'+"".join(f"<th>{html.escape(str(h))}</th>" for h in headers)+
                "</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{html.escape(str(c))}</td>" for c in row)+"</tr>" for row in rows)+"</tbody></table>")
    contents = ["""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>论文与程序逐项核对</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#223342}
table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:left;border:1px solid #dae1e7;padding:8px;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf4f6}code,pre{background:#f1f5f7;padding:5px}pre{white-space:pre-wrap;padding:16px}.finding{border-left:5px solid #137f73;padding:14px;background:#edf8f5}input{padding:10px;width:75%}a{color:#096b8c}
</style><body><h1>论文与程序逐项核对</h1>
<p class="finding"><b>结论：之前大规模实验使用工程修正版，不能称为严格复现。新增 paper_reference.py 按论文第 3 节保留原常数、原分支、原阶数及原判定。</b></p>
<p>切比雪夫递推和频数聚合在代数上没有改变论文估计器；旧版改变的是统计参数、阶数选择与随机误差修正。
本次没有把原实验的误判率改挂到严格原版名下。</p>
<p>严格入口以 Arb 区间计算认证参数取整和离散输出。算力或精度不够就报错，不截断阶数或猜测结论。
这是给定频数的数值认证；采样必须符合论文的 iid 泊松模型，有限测试也不是整份程序的形式化证明。</p>""",
        f"<p><b>测试：</b>{html.escape((out/'tests.log').read_text().strip().splitlines()[-1])}。独立有理数闭式展开、泊松期望恒等式、原常数、取整、分支临界点、预算和输出边界均已核对。</p>",
        "<h2>原公式对应</h2>", table(["论文位置", "数学定义", "代码位置/函数", "核对结论"], formula_rows()),
        """<h2>哪些函数实际用到了</h2>
<p>下表是实际调用记录。跟踪范围：严格原版两个分支及 CLI 模拟；旧版合成/鱼类/CSV 入口各少量运行；
大规模算法的 evaluate 实际调用路径（合成和五套经验分布，每单元 3 次）；全套测试。
这些小运行只用于核对调用关系，没有重新估计正式实验误判率。</p>
<p><b>不能要求所有函数在每次运行中都调用。</b>小 L 分支没有多项式计算；批量版把逐项求和写成矩阵运算；
Q(x)、方差和证明界只用于证明或测试；下载与预处理仅在缺少缓存时运行。未在此次跟踪中出现，不等于没有用途。</p>
<p>特别核对：reference_weights 只在原版大 L 分支调用；paper_correction 在原版及 naive 对照调用；
paper_statistic 旧版仅用于 naive 对照；hoeffding_statistic 才是工程版修正。
clipped_paper_lower_bound 现在由严格原版调用，旧批量版用等价的向量表达式。</p>
<label>筛选函数：<input id="filter" placeholder="例如 reference_weights、paper_statistic、support_size.py"></label>""",
        table(["文件:行 / 函数", "严格原版", "旧版入口", "批量评测", "测试", "状态"],
              [[f"{r['file']}:{r['line']} / {r['function']}", r['paper_original'], r['legacy_entrypoints_smoke'],
                r['batch_evaluate_smoke'], r['tests'], r['status']] for r in functions], "functions"),
        """<h2>运行严格原版</h2><pre>cd /home/snowy/OSST
.venv/bin/python paper_reference.py plan --n 1000 --epsilon 0.1
.venv/bin/python paper_reference.py demo --n 1000 --epsilon 0.1
.venv/bin/python audit_paper.py</pre>
<p>n=1000、ε=0.1 时，原版 m=84,864,619,140，走直接计数分支。
demo 对已知有限分布直接生成泊松类别计数，所以可以运行；它没有降低现实观测需求。
n=10^445 的多项式分支用人为构造的压缩频数检查了全部三个尺度，阶数为 6、6、5；这不是如此规模的现场采样实验。</p>
<p>文件：<a href="formula_mapping.csv">公式表</a> · <a href="function_usage.csv">全函数清单及静态候选调用点</a> ·
<a href="reference_runs.json">两个原版分支的计算记录</a> · <a href="original_parameter_demo.json">原预算模拟</a> ·
<a href="tests.log">测试日志</a> · <a href="audit_summary.json">文件哈希与跟踪范围</a> ·
<a href="paper_page6.png">论文第 6 页</a></p>
<p>旧实验协议中的源码哈希对应 before_audit 保存的原文件。正式结果没有因本次审计被重新命名为原算法结果。
函数清单中的静态调用点只按名称匹配，可能有重名，不能代替实际调用证据。</p>
<script>document.getElementById('filter').addEventListener('input', e=>{const q=e.target.value.toLowerCase();
document.querySelectorAll('#functions tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></body></html>"""]
    (out/"report.html").write_text("\n".join(contents))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results_audit")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    calls = Calls()
    workloads(out, calls)
    definitions, sites = inventory()
    functions = []
    for definition in definitions:
        counts = calls.counts[(definition["file"], definition["function"])]
        data = {**definition, **{key: counts[key] for key in
                ["paper_original", "legacy_entrypoints_smoke", "batch_evaluate_smoke", "tests"]}}
        production = sum(count for phase, count in counts.items() if phase != "tests")
        data["status"] = "实际入口已调用" if production else "此次仅在测试调用" if counts["tests"] else "此次未覆盖；需结合静态入口"
        data["static_candidate_calls"] = "; ".join(sorted(sites[definition["function"].split(".")[-1]]))
        functions.append(data)
    write_csv(out/"function_usage.csv", functions)
    write_csv(out/"formula_mapping.csv", [dict(zip(["paper", "formula", "implementation", "finding"], row)) for row in formula_rows()])
    sources = sorted([*ROOT.glob("*.py"), ROOT/"requirements-lock.txt", ROOT/"requirements.txt", ROOT/"pytest.ini"])
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "paper_sha256": hashlib.sha256((ROOT/"2610.18723v1.pdf").read_bytes()).hexdigest(),
               "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "functions": len(functions), "statuses": dict(Counter(r["status"] for r in functions)),
               "core_not_executed": [r["function"] for r in functions if r["file"] in CORE and not sum(calls.counts[(r["file"], r["function"])].values())],
               "tests": (out/"tests.log").read_text().strip().splitlines()[-1],
               "trace_scope": "原版两分支和 demo、旧版各入口少量运行、evaluate 小批量路径、全测试；不是全部 CLI/下载分支覆盖或整套程序形式化证明。",
               "historical_results": "results/results_large 数值保持历史结果；before_audit 源码快照对应旧 protocol 哈希。"}
    save_json(out/"audit_summary.json", summary)
    render(out, summary, functions)
    print(json.dumps({"report": str(out/"report.html"), "functions": len(functions),
                      "statuses": summary["statuses"], "core_not_executed": summary["core_not_executed"],
                      "tests": summary["tests"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
