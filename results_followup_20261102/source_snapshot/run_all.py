#!/usr/bin/env python3
"""一条命令运行验证、模拟、真实数据实验和 CSV 示例，并生成便携实验包。

python run_all.py --trials 500
更改 trials/seed 后，以 CSV/JSON 为准；REPORT.md 保留已记录的默认实验。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from experiments import run_fish, run_synthetic, write_json

ROOT = Path(__file__).resolve().parent


def build_bundle(out: Path) -> Path:
    """打包项目文件与结果，排除环境、下载工具和临时研究页面。"""
    paths = [x for x in ROOT.iterdir() if x.is_file() and x.suffix in {".py", ".md", ".txt", ".pdf"}]
    paths += [x for folder in [ROOT / "data", out] for x in folder.iterdir() if x.is_file()]
    paths = sorted(set(paths))
    entries = []
    for p in paths:
        if p.parent == out:
            arc = Path("results") / p.name
        else:
            arc = p.relative_to(ROOT)
        entries.append((p, arc))
    manifest = "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {arc.as_posix()}" for p, arc in entries) + "\n"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    target = artifacts / "osst_experiment_bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in entries:
            z.write(p, str(Path("OSST") / arc))
        z.writestr("OSST/MANIFEST.sha256", manifest)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError("压缩包 CRC 检查未通过")
    (artifacts / "osst_experiment_bundle.sha256").write_text(
        hashlib.sha256(target.read_bytes()).hexdigest() + "  " + target.name + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("trials 必须是正整数")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("1/4 验证关键公式、边界行为和固定数据版本", flush=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "test_support_size.py"),
                            str(ROOT / "test_paper_reference.py")],
                           cwd=ROOT, capture_output=True, text=True)
    (out / "tests.log").write_text(tests.stdout + tests.stderr)
    print(tests.stdout, end="", flush=True)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    print("2/4 合成分布实验", flush=True)
    synthetic = run_synthetic(out, trials=args.trials, seed=args.seed)
    print("3/4 真实鱼类数据回测", flush=True)
    fish = run_fish(out, args.trials, args.seed)
    print("4/4 验证真实 CSV 入口、保存运行信息并打包", flush=True)
    examples = {}
    for name, extra in [("fixed_csv", []), ("poissonized_csv", ["--poissonize-fixed"])]:
        command = [sys.executable, str(ROOT / "support_size_demo.py"), "--csv",
                   str(ROOT / "data/example_300_fish.csv"), "--n", "8", "--epsilon", ".01",
                   "--seed", str(args.seed), *extra]
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT)
        examples[name] = json.loads(result.stdout)
        write_json(out / f"example_{name}.json", examples[name])
    # Gap 行不计错，数值输出必须有限，确保交付结果可机器读取。
    for filename in ["fish_testing_results.csv", "synthetic_results.csv"]:
        with (out / filename).open(newline="") as f:
            for row in csv.DictReader(f):
                if row["truth"] == "Gap":
                    assert row["error_rate"] == ""
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "seed": args.seed, "trials": args.trials,
               "tests": tests.stdout.strip(), "synthetic": synthetic, "fish": fish,
               "report_scope": "REPORT.md 数值对应默认 500 次和种子 20261016；其他配置以结果 CSV/JSON 为准",
               "fixed_csv_example": examples["fixed_csv"],
               "poissonized_csv_example": examples["poissonized_csv"]}
    write_json(out / "run_summary.json", summary)
    bundle = build_bundle(out)
    print(f"完成。结果目录：{out}\n便携实验包：{bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
