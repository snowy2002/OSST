"""Replay the complete finite-n MGF certificate without numerical optimization.

    .venv/bin/python proof_l01/verify_finite.py

Only saved rational t/eta proposals are reused. Geometry, polynomial
positivity, Chernoff bounds, and the final union bounds are recomputed.
Missing/duplicated n or scales are rejected before any arithmetic verification.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flint import arb, ctx
from parameter_search.algorithm import geometry
from proof_l01 import mgf

OUT = ROOT / "results_proof_l01"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_coverage(document):
    """Check the exact finite index set and every scale; never use zip truncation."""
    require(document.get("C") == 70 and document.get("L0") == 1, "Wrong C/L0")
    require(document.get("rigorous") is True, "Certificate artifact not marked rigorous")
    rows = document.get("rows")
    require(isinstance(rows, list), "Missing rows list")
    require([r.get("n") for r in rows] == list(range(75, 404)),
            "n coverage must be the exact ordered sequence 75..403")
    for row in rows:
        n = row["n"]
        _, expected = geometry("70", 1, n, 512)
        supplied = row.get("scales")
        require(isinstance(supplied, list), f"n={n}: missing scales list")
        require([r.get("j") for r in supplied] == [s.j for s in expected],
                f"n={n}: scales must be complete, ordered and unique")
        require(len(supplied) == len(expected), f"n={n}: scale count mismatch")
        for index in range(len(expected)):
            actual, target = supplied[index], expected[index]
            require(actual.get("k") == target.k, f"n={n}, j={target.j}: wrong degree")
            require(Fraction(actual["delta"]) == target.delta, f"n={n}, j={target.j}: wrong delta")
            require(isinstance(actual.get("t"), str) and isinstance(actual.get("eta"), str),
                    f"n={n}, j={target.j}: t/eta must be exact rational strings")
            require(Fraction(actual["t"]) > 0 and Fraction(actual["eta"]) > 0,
                    f"n={n}, j={target.j}: nonpositive t/eta")
    return rows


def negative_controls(document):
    results = []
    for name in ["delete_one_n", "delete_one_scale"]:
        altered = copy.deepcopy(document)
        if name == "delete_one_n":
            del altered["rows"][0]
        else:
            del altered["rows"][0]["scales"][0]
        try:
            validate_coverage(altered)
        except ValueError as exc:
            results.append({"test": name, "rejected": True, "reason": str(exc)})
        else:
            raise ArithmeticError(f"Negative control was accepted: {name}")
    return results


def no_optimization(*args, **kwargs):
    raise ArithmeticError("Optimization is forbidden in certificate replay")


def main():
    start = time.perf_counter()
    path = OUT / "mgf_certificates.json"
    document = json.loads(path.read_text())
    # This also turns accidental calls into a hard failure rather than silently
    # reoptimizing the supplied certificate and changing what is being checked.
    mgf.minimize_scalar = no_optimization
    mgf.numerical_scale = no_optimization
    rows = validate_coverage(document)
    negative = negative_controls(document)
    verified = []
    scale_count = 0
    with ctx.workprec(512):
        maximum = arb(0)
        for row in rows:
            n = row["n"]
            _, scales = geometry("70", 1, n, 512)
            total = arb(0)
            for index in range(len(scales)):
                scale, stored = scales[index], row["scales"][index]
                t, eta = Fraction(stored["t"]), Fraction(stored["eta"])
                certificate, bound = mgf.certify_scale(n, scale, t, eta, bits=512)
                require(certificate["passed"], f"Positivity certificate failed n={n}, j={scale.j}")
                require(bound >= 0, f"Negative probability bound n={n}, j={scale.j}")
                total += bound
                scale_count += 1
            require(total < mgf.ball(Fraction(8, 25)), f"Upper-tail bound >= .32 for n={n}")
            maximum = maximum.max(total)
            verified.append({"n": n, "scale_count": len(scales), "passed": True,
                             "union_bound_arb": total.str(65)})
    audit = {
        "status": "PASS", "C": 70, "L0": 1, "n_range_inclusive": [75, 403],
        "n_count": len(verified), "scale_count": scale_count,
        "coverage_exact_and_no_duplicates": True,
        "geometry_rebuilt_and_k_delta_checked": True,
        "saved_rational_t_eta_positive": True,
        "all_polynomial_certificates_recomputed": True,
        "all_union_bounds_strictly_below": "8/25",
        "maximum_union_bound_arb": maximum.str(65),
        "optimization_executed": False, "optimization_calls_disabled": True,
        "negative_controls": negative, "rows": verified,
        "certificate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mgf_code_sha256": hashlib.sha256((ROOT / "proof_l01/mgf.py").read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "Upper deviation only, uniformly over S<=n and arbitrary independent Poisson means. Lower deviation and other n ranges require separate arguments.",
    }
    (OUT / "finite_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["status", "n_count", "scale_count", "maximum_union_bound_arb",
                                           "optimization_executed", "negative_controls", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
