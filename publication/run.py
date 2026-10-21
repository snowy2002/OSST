"""Run the prospective follow-up without overwriting historical experiments.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m publication.run --phase all
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

from publication.protocol import ROOT,OUT,freeze_design,freeze_evaluation,save,sha
from publication.suites import (development,fixed_suite,sequential_suite,adversarial_search,adversarial_test)
from publication.stream import stream_suite
from publication.structure import structure_suite
from publication.sensitivity import sensitivity_suite
from publication.reference import reference_suite
from publication.runtime import runtime_suite

PHASES={"fixed":fixed_suite,"sequential":sequential_suite,"search":adversarial_search,
        "adversarial":adversarial_test,"stream":stream_suite,"structure":structure_suite,
        "sensitivity":sensitivity_suite,"reference":reference_suite,"runtime":runtime_suite}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["all","prepare","report",*PHASES],default="all")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--quick",action="store_true",help="Smoke test only; refuses official output directory")
    args=parser.parse_args();out=args.out.resolve()
    if args.quick and out==OUT.resolve():raise ValueError("Use --out results_publication_smoke for quick tests")
    out.mkdir(parents=True,exist_ok=True);(out/"raw").mkdir(exist_ok=True)
    if args.phase=="report":
      from publication.report import report
      report(out);return
    freeze_design(out)
    if args.phase in ["all","prepare"]:
      if not (out/"calibration.json").exists():
        tests=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
        (out/"tests.log").write_text(tests.stdout+tests.stderr)
        if tests.returncode:raise RuntimeError("Tests failed; see tests.log")
        configs=development(out,args.quick)
      else:
        configs=json.loads((out/"calibration.json").read_text())["configs"]
      freeze_evaluation(configs,out)
      if args.phase=="prepare":return
    else:
      configs=json.loads((out/"calibration.json").read_text())["configs"]
      freeze_evaluation(configs,out)
    phases=list(PHASES) if args.phase=="all" else [args.phase]
    for phase in phases:
      marker=out/f"execution_{phase}.json"
      if marker.exists():
        previous=json.loads(marker.read_text())
        if previous.get("status")=="complete":
          print(f"{phase}: already complete, preserving frozen results",flush=True);continue
      record=dict(phase=phase,status="running",mode="smoke" if args.quick else "formal",
                  started_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/"protocol.json"))
      save(marker,record);t=time.perf_counter()
      try:
        PHASES[phase](configs,out,args.quick)
      except Exception as exc:
        record.update(status="failed",exception=repr(exc),elapsed_seconds=time.perf_counter()-t)
        save(marker,record);raise
      record.update(status="complete",elapsed_seconds=time.perf_counter()-t,
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
      save(marker,record)
    if args.phase=="all":
      from publication.report import report
      report(out)


if __name__=="__main__":main()
