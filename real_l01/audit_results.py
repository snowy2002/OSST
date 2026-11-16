"""Audit saved real-data experiments without changing their frozen sources.

Run from the repository with::

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m real_l01.audit_results

Writes only results_real_l01_20261104/results_audit.json. Exact integer/rational
event checks do not call the experiment's outcome or Population helpers. The
78 observed failing paths are additionally recomputed using the same Arb
formula implementation at higher precision; this is not a second symbolic
implementation. One complete cell per data source receives a PRNG replay.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import poisson

from real_l01 import algorithm as alg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_real_l01_20261104"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(*keys):
    return int.from_bytes(hashlib.sha256("|".join(map(str, keys)).encode()).digest()[:8], "little")


def main():
    start = time.perf_counter()
    results = json.loads((OUT / "results.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    populations = json.loads((OUT / "prepared/populations.json").read_text())
    lookup = {p["id"]: p for p in populations}
    failures, mean_diagnostics, high_precision = [], [], []
    checked, gain = Counter(), Counter()
    sampled_sources = {}

    def check(condition, kind, case=None, detail=None):
        checked[kind] += 1
        if not bool(condition):
            failures.append({"check": kind, "case": case, "detail": detail})

    check(sha(OUT / "prepared/populations.json") == protocol["population_sha256"], "population_sha256")
    check(len(results) == len(protocol["cases"]) == 640, "case_count")
    check(len({r["id"] for r in results}) == len(results), "unique_case_ids")
    frozen = json.loads((OUT / "source_hashes.json").read_text())
    for name, digest in frozen.items():
        check(sha(OUT / "source_snapshot" / name) == digest, "frozen_source_sha256", detail=name)
        check(sha(ROOT / name) == digest, "current_source_sha256", detail=name)

    for r, case in zip(results, protocol["cases"]):
        cid = r["id"]
        pop = lookup[r["population_id"]]
        counts = sorted(pop["counts"], reverse=True)
        N, S, n, eps, trials = sum(counts), len(counts), r["n"], Fraction(r["epsilon"]), r["trials"]
        tail = Fraction(sum(counts[n:]), N)
        truth = "Yes" if S <= n else "No" if tail >= eps else "Gap"
        remain, eff = N, None
        for i, v in enumerate(counts, 1):
            remain -= v
            if Fraction(remain, N) <= eps:
                eff = i
                break
        check(all(r[k] == v for k, v in case.items()), "protocol_case_match", cid)
        check(truth == r["truth"] and S == r["S"] and eff == r["eff"] and tail == Fraction(r["tail_exact"]),
              "exact_empirical_truth", cid)
        m = 70 * n / (float(eps) * (1 + math.log(n)))
        check(math.isclose(r["m"], m, rel_tol=1e-14), "poisson_mean_formula", cid)
        path = OUT / "raw" / f"{cid}.npz"
        check(sha(path) == r["raw_sha256"], "raw_sha256", cid)
        with np.load(path, allow_pickle=False) as z:
            D, F, M, block, degree = z["observed"], z["low"], z["M"], z["block"], r["degree"]
            check(D.shape == M.shape == block.shape == (trials,) and F.shape == (trials, degree), "fingerprint_shapes", cid)
            check(all(np.issubdtype(x.dtype, np.integer) for x in [D, F, M, block]), "integer_fingerprints", cid)
            check(np.all((0 <= D) & (D <= S) & (D <= M)) and np.all(F >= 0) and np.all(M >= 0), "fingerprint_ranges", cid)
            check(np.all(F.sum(axis=1) <= D), "sum_F_le_D", cid)
            weighted = F @ np.arange(1, degree + 1)
            check(np.all(weighted <= M), "sum_lF_le_M", cid)
            check(np.all(M - weighted >= (degree + 1) * (D - F.sum(axis=1))), "high_count_remainder_feasible", cid)
            sizes = np.full(4, trials // 4) + (np.arange(4) < trials % 4)
            check(np.array_equal(block, np.repeat(np.arange(4), sizes)), "four_seed_blocks", cid)
            expected_seeds = [seed("real_l01_v1", cid, r["population_id"], j) for j in range(4)]
            check(r["seeds"] == expected_seeds and len(set(expected_seeds)) == 4, "seed_derivation", cid)
            check(r["baseline_seed"] == seed("real_l01_v1_count_budget", cid, r["population_id"]), "baseline_seed_derivation", cid)
            check(math.isclose(float(M.mean()), r["mean_actual_M"], rel_tol=1e-15)
                  and math.isclose(float(M.var(ddof=1)), r["var_actual_M"], rel_tol=1e-14), "M_summary_matches_raw", cid)
            mu, total = trials * r["m"], int(M.sum())
            pmean = min(1., 2 * min(float(poisson.cdf(total, mu)), float(poisson.sf(total - 1, mu))))
            mean_diagnostics.append({"case": cid, "dataset": r["dataset"], "planned_m": r["m"],
                "sample_mean_M": float(M.mean()), "mean_z": (float(M.mean()) - r["m"]) / math.sqrt(r["m"] / trials),
                "exact_two_sided_total_poisson_p": pmean})
            check(pmean >= .05 / len(results), "poisson_M_mean_bonferroni_95", cid,
                  {"p": pmean, "cutoff": .05 / len(results)})

            for mode in ["full", "single", "count", "no_count"]:
                lo, rej = z[mode + "_lower"], z[mode + "_reject"]
                over, short = z[mode + "_over"], z[mode + "_short"]
                binary, joint, stats = z[mode + "_binary"], z[mode + "_joint"], r["stats"][mode]
                check(np.issubdtype(lo.dtype, np.integer) and np.all((1 <= lo) & (lo <= n)), "output_range_" + mode, cid)
                check(all(v.shape == (trials,) and v.dtype == np.dtype(bool) for v in [rej, over, short, binary, joint]),
                      "event_bool_shapes_" + mode, cid)
                check(np.array_equal(over, lo > min(S, n)), "over_from_output_" + mode, cid)
                check(np.array_equal(short, lo < min(eff, n)), "short_from_output_" + mode, cid)
                expected_binary = rej if truth == "Yes" else ~rej if truth == "No" else np.zeros(trials, dtype=bool)
                check(np.array_equal(binary, expected_binary), "binary_from_truth_" + mode, cid)
                check(np.array_equal(joint, binary | over | short), "joint_union_" + mode, cid)
                check(np.all(lo[rej] == n), "reject_implies_clipped_n_" + mode, cid)
                check(all(stats[k] == int(z[mode + "_" + k].sum()) for k in ["reject", "over", "short", "binary", "joint"]),
                      "event_summary_" + mode, cid)
                check(stats["block_joint_failures"] == [int(joint[block == j].sum()) for j in range(4)], "block_summary_" + mode, cid)
                check(math.isclose(stats["mean_lower"], float(lo.mean()), rel_tol=1e-15)
                      and stats["q05_lower"] == float(np.quantile(lo, .05))
                      and stats["q95_lower"] == float(np.quantile(lo, .95)), "lower_summary_" + mode, cid)

            check(np.array_equal(z["count_lower"], np.clip(D, 1, n)) and np.array_equal(z["count_reject"], D > n),
                  "count_baseline_definition", cid)
            check(np.all(z["full_lower"] >= z["single_lower"])
                  and np.all(z["single_lower"] >= z["count_lower"])
                  and np.all(z["full_lower"] >= z["no_count_lower"]), "ablation_output_order", cid)
            B, Blo = z["count_budget_observed"], z["count_budget_lower"]
            Brej = B > n
            Bbinary = Brej if truth == "Yes" else ~Brej if truth == "No" else np.zeros(trials, dtype=bool)
            Bshort = Blo < min(eff, n)
            check(np.all((0 <= B) & (B <= S)) and np.array_equal(Blo, np.clip(B, 1, n)), "budget_count_definition", cid)
            check(np.array_equal(z["count_budget_binary"], Bbinary) and np.array_equal(z["count_budget_short"], Bshort),
                  "budget_count_events", cid)
            check(r["stats"]["count_budget"]["binary"] == int(Bbinary.sum())
                  and r["stats"]["count_budget"]["short"] == int(Bshort.sum())
                  and r["stats"]["count_budget"]["joint"] == int((Bbinary | Bshort).sum())
                  and r["stats"]["count_budget"]["over"] == 0, "budget_count_summary", cid)
            higher, fullfail, countfail = z["full_lower"] > z["count_lower"], z["full_joint"], z["count_joint"]
            check(int(higher.sum()) == r["full_better_than_count"], "higher_than_count_summary", cid)
            check(int((z["full_lower"] > z["single_lower"]).sum()) == r["full_better_than_single"], "higher_than_single_summary", cid)
            masks = {
                "higher_full_output": higher,
                "higher_output_full_joint_failure": higher & fullfail,
                "higher_output_full_joint_success": higher & ~fullfail,
                "higher_output_full_and_count_both_joint_success": higher & ~fullfail & ~countfail,
                "higher_output_rescues_count_joint_failure": higher & ~fullfail & countfail,
                "higher_output_creates_full_joint_failure": higher & fullfail & ~countfail,
                "any_rescued_count_joint_failure": ~fullfail & countfail,
                "any_created_full_joint_failure": fullfail & ~countfail,
            }
            for key, mask in masks.items():
                gain[key] += int(mask.sum())

            for idx in np.flatnonzero(fullfail):
                plan = alg.parameters("70", 1, n, r["epsilon"], bits=1024)
                exact = alg.exact_output(int(D[idx]), F[idx], plan, "full", precisions=(1024, 2048, 4096, 8192))
                exact_reject = exact["decision"] == "No"
                exact_over, exact_short = exact["lower"] > min(S, n), exact["lower"] < min(eff, n)
                exact_binary = exact_reject if truth == "Yes" else not exact_reject if truth == "No" else False
                check(exact["lower"] == int(z["full_lower"][idx]) and exact_reject == bool(z["full_reject"][idx])
                      and exact_over == bool(z["full_over"][idx]) and exact_short == bool(z["full_short"][idx])
                      and exact_binary == bool(z["full_binary"][idx]), "failed_path_high_precision_confirmation", cid,
                      {"index": int(idx)})
                high_precision.append({"case": cid, "trial_index": int(idx), "dataset": r["dataset"], "truth": truth,
                    "S": S, "n": n, "eff": eff, "exact": exact, "over": exact_over, "short": exact_short,
                    "binary": exact_binary, "count_joint": bool(countfail[idx]), "higher_output_than_count": bool(higher[idx])})

            if r["dataset"] not in sampled_sources:
                cp = np.asarray(pop["counts"], dtype=np.int64)
                probs = cp / cp.sum()
                regen, regD, regF = [], [], []
                for b, size in enumerate(sizes):
                    draws = np.random.default_rng(expected_seeds[b]).poisson(r["m"] * probs, size=(int(size), S))
                    regen.append(draws.sum(axis=1))
                    regD.append((draws > 0).sum(axis=1))
                    # Per-frequency equality counts avoid retaining a large
                    # bincount array for species with very high Poisson means.
                    regF.append(np.column_stack([(draws == ell).sum(axis=1) for ell in range(1, degree + 1)]))
                check(np.array_equal(np.concatenate(regen), M) and np.array_equal(np.concatenate(regD), D)
                      and np.array_equal(np.concatenate(regF), F), "one_case_per_source_seed_replay", cid)
                sampled_sources[r["dataset"]] = mean_diagnostics[-1]

    audit = {
        "status": "PASS" if not failures else "FAIL",
        "n_cells": len(results), "n_paths": sum(r["trials"] for r in results),
        "checks": dict(checked), "failures": failures,
        "population_sha256": sha(OUT / "prepared/populations.json"), "results_sha256": sha(OUT / "results.json"),
        "frozen_source_count": len(frozen),
        "mean_M_diagnostics": {
            "method": "For each cell, sum(M) ~ Poisson(trials*m); exact doubled smaller tail and 0.05/640 threshold; PRNG floating point model as recorded in run protocol. This is a diagnostic, not proof of RNG quality.",
            "max_abs_z": max(abs(x["mean_z"]) for x in mean_diagnostics),
            "minimum_p": min(x["exact_two_sided_total_poisson_p"] for x in mean_diagnostics),
            "one_seed_replayed_case_per_source": sampled_sources, "all_cells": mean_diagnostics,
        },
        "gain_interpretation": {
            "counts": dict(gain),
            "valid_higher_output_definition": "full_lower > count_lower AND full_joint == false; this means a valid tighter output under empirical truth, not necessarily rescue of a failed count result.",
            "rescued_definition": "full_joint == false AND count_joint == true; higher-output restriction separately reported.",
        },
        "high_precision_failure_confirmation": {
            "count": len(high_precision), "starting_bits": 1024,
            "method": "Recompute each full-failure path with Arb exact_output from original fingerprint, independent of stored vectorized score intervals; geometry/weights are same certified-arithmetic implementation.",
            "paths": high_precision,
        },
        "limits": [
            "Boolean truth/event audit is independent exact integer/rational recomputation. Failure score replay uses the same Arb formula implementation at higher precision, not a second symbolic estimator implementation.",
            "Only one entire cell per data source had RNG replay; all 640 cells had fingerprint, hash, block and truth/event checks.",
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUT / "results_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "cells": len(results), "failure_paths_rechecked": len(high_precision),
                      "gain": dict(gain), "failed_checks": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
