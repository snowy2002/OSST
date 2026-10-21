#!/usr/bin/env python3
"""一条命令下载、校准、运行大规模实验、绘图并打包。

默认正式单元重复 500 次；开发集 200 次。原来的小实验结果不会被覆盖。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from benchmark_data import DATA, ROOT, prepare_all, save_json, sha256
from large_experiments import (calibrate, finish_results, run_dependence_stress, run_real_large,
                               run_richness, run_synthetic_large, write_protocol)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = int(value)
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def bundle(out):
    """便携包只有程序、协议、汇总和图，不分发任何跨领域原始/派生数据。"""
    target = ROOT / "artifacts" / "osst_large_benchmark.zip"
    target.parent.mkdir(exist_ok=True)
    paths = [(p, p.name) for p in ROOT.iterdir() if p.is_file() and p.suffix in {".py", ".md", ".txt"}]
    paths += [(p, "results_large/"+p.name) for p in out.iterdir() if p.is_file()]
    paths += [(DATA/p, "data/large/"+p) for p in ["sources.json", "dataset_summary.json"]]
    manifest = "\n".join(f"{sha256(p)}  {arc}" for p, arc in sorted(paths, key=lambda x: x[1]))+"\n"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in paths:
            z.write(p, "OSST/"+arc)
        z.writestr("OSST/MANIFEST.sha256", manifest)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError("压缩包损坏")
        if any(x.endswith(("_counts.npz", "_groups.npz", "ratings.dat", "ml-1m.zip")) for x in z.namelist()):
            raise ValueError("便携包不应含原始或派生 MovieLens 数据")
    target.with_suffix(".sha256").write_text(sha256(target)+"  "+target.name+"\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_large")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--calibration-trials", type=int, default=200)
    parser.add_argument("--stress-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--phase", choices=["all", "prepare", "calibrate", "evaluate", "report"], default="all")
    args = parser.parse_args(argv)
    if min(args.trials, args.calibration_trials, args.stress_trials) < 1:
        parser.error("重复次数必须 >=1")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.phase in ["all", "prepare"]:
        prepare_all()
    if args.phase == "prepare":
        return 0
    if args.phase in ["all", "calibrate", "evaluate"]:
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_support_size.py", "test_benchmark.py"],
                               cwd=ROOT, text=True, capture_output=True)
        (out / "tests.log").write_text(tests.stdout+tests.stderr)
        print(tests.stdout, end="", flush=True)
        if tests.returncode:
            raise SystemExit(tests.returncode)
    if args.phase in ["all", "calibrate"]:
        config = calibrate(out, args.seed+39019, args.calibration_trials)
    elif args.phase == "report":
        protocol = json.loads((out / "protocol.json").read_text())
        if sha256(out / "calibration.json") != protocol["calibration_sha256"]:
            raise ValueError("校准文件已改变，不能用于重建旧测试的报告；请重新 evaluate")
        config = protocol["config"]
    else:
        config = json.loads((out / "calibration.json").read_text())["selected_config"]
    if args.phase == "calibrate":
        return 0
    if args.phase in ["all", "evaluate"]:
        protocol = write_protocol(out, config, args.trials, args.seed, args.stress_trials, DATA)
        synthetic = run_synthetic_large(out, config, args.trials, args.seed)
        real = run_real_large(out, config, args.trials, args.seed)
        stress = run_dependence_stress(out, config, args.stress_trials, args.seed)
        richness = run_richness(out, config, args.trials, args.seed)
    else:
        synthetic, real, stress, richness = [read_rows(out/name) for name in
                                            ["synthetic_large.csv", "real_large.csv", "dependence_stress.csv", "richness_curves.csv"]]
    summary = finish_results(out, synthetic, real, stress, richness, config)
    if args.phase != "report":
        save_json(out / "execution.json", {"seconds": time.perf_counter()-start,
                                            "python": sys.version, "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}})
    from report_large import generate_report
    report = generate_report(out)
    target = bundle(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"实验结果：{out}\n中文交互报告：{report}\n程序及结果包：{target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
