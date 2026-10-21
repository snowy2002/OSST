#!/usr/bin/env python3
"""中文命令行入口。示例见 README.md；默认只打印帮助。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from fish_data import load_counts_csv
from support_size import (Fingerprint, count_plan, count_test, describe_sample,
                          make_scales, hoeffding_statistic, paper_plan, poissonized_fixed_test)


def main(argv=None):
    p = argparse.ArgumentParser(description="支持集大小检验及鱼类捕获数据实验")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--paper-plan", action="store_true", help="查看论文完整参数；不执行巨量抽样")
    action.add_argument("--baseline-plan", action="store_true", help="用更紧的尾概率界规划计数方案")
    action.add_argument("--demo", action="store_true", help="运行工程修正版及对照实验（不是论文原算法）")
    action.add_argument("--fish-demo", action="store_true", help="下载/读取真实捕获数据并回测")
    action.add_argument("--self-test", action="store_true", help="运行公式与边界测试")
    action.add_argument("--csv", type=Path, help="读取自己的逐条类别 CSV")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--failure-prob", type=float, default=0.05)
    p.add_argument("--sampling", choices=["fixed", "poisson"], default="fixed")
    p.add_argument("--m", type=float, help="事先指定的泊松均值，不能用实际 M 替代")
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--species-column", default="species")
    p.add_argument("--count-column", help="可选的整数个体数列；不是重量或 CPUE")
    p.add_argument("--assume-iid", action="store_true", help="显式声明数据来自固定分布独立采样；程序不能验证")
    p.add_argument("--poissonize-fixed", action="store_true", help="从固定样本 CSV 做一次泊松子抽样，再运行多项式")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = p.parse_args(argv)
    try:
        if args.poissonize_fixed and (args.csv is None or args.sampling != "fixed" or args.m is not None):
            raise ValueError("--poissonize-fixed 需要 --csv、--sampling fixed，且不能另设 --m")
        if args.paper_plan:
            result = paper_plan(args.n, args.epsilon)
        elif args.baseline_plan:
            result = count_plan(args.n, args.epsilon, args.failure_prob, args.sampling)
        elif args.demo or args.fish_demo:
            if args.trials < 1:
                raise ValueError("trials 必须是正整数")
            from experiments import run_synthetic, run_fish
            if args.demo:
                result = run_synthetic(args.out, args.n, args.epsilon,
                                       args.trials, args.seed, args.failure_prob)
            else:
                result = run_fish(args.out, args.trials, args.seed)
        elif args.self_test:
            import pytest
            return pytest.main([str(Path(__file__).with_name(name)) for name in
                                ["test_support_size.py", "test_benchmark.py", "test_paper_reference.py"]]+["-q"])
        elif args.csv:
            counts = load_counts_csv(args.csv, args.species_column, args.count_column)
            fp = Fingerprint.from_counts(counts.values())
            result = describe_sample(fp)
            result["count_test"] = count_test(fp, args.n, args.epsilon, alpha=args.failure_prob,
                                               sampling=args.sampling, m=args.m)
            result["guarantee_status"] = ("conditional_on_declared_iid_sampling" if args.assume_iid
                                          else "descriptive_only_iid_not_established")
            result["sampling_note"] = "真实捕获数据通常含渔具偏差/事件内相关；数学保证取决于采样模型"
            if args.sampling == "poisson":
                scales = make_scales(args.n, args.epsilon, args.m)
                score = hoeffding_statistic(fp, scales, args.n, args.m, args.failure_prob)
                result["practical_polynomial"] = {
                    "score": score, "decision": "No" if score > args.n else "Inconclusive",
                    "support_lower_bound": max(fp.observed, min(args.n, max(1, math.ceil(score)))),
                    "alpha": args.failure_prob,
                    "guarantee": "泊松化 iid 下的单侧误报控制；此预算没有统一漏报保证",
                    "scales": len(scales)}
            elif args.poissonize_fixed:
                import numpy as np
                result["poissonized_polynomial"] = poissonized_fixed_test(
                    counts.values(), args.n, args.epsilon,
                    np.random.default_rng(args.seed), alpha=args.failure_prob)
        else:
            p.print_help()
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as e:
        p.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
