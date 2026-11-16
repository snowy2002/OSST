"""Certify the n=74 bridge, without changing the n=75..403 certificates.

First generation (numerical proposals followed by rigorous checks)::
    .venv/bin/python proof_l01/bridge74.py --generate

Default mode independently replays saved rational parameters, no optimization::
    .venv/bin/python proof_l01/bridge74.py
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flint import arb, ctx
from parameter_search.algorithm import geometry
from proof_l01 import mgf

OUT = ROOT / "results_proof_l01"
CERTIFICATE = OUT / "n74_certificate.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(document):
    require(document.get("C") == 70 and document.get("L0") == 1 and document.get("n") == 74,
            "Expected exactly C=70, L0=1, n=74")
    require(document.get("rigorous") is True, "Expected rigorous certificate artifact")
    _, geometry_scales = geometry("70", 1, 74, 512)
    require(len(geometry_scales) == 7, "Reconstructed n=74 geometry does not have seven scales")
    saved = document.get("scales")
    require(isinstance(saved, list) and len(saved) == 7, "Exactly seven saved scales required")
    require([s.get("j") for s in saved] == list(range(7)), "Scales j=0..6 must be complete, ordered and unique")
    for index in range(7):
        value, actual = saved[index], geometry_scales[index]
        require(value.get("k") == actual.k, f"Scale {index}: degree mismatch")
        require(Fraction(value["delta"]) == actual.delta, f"Scale {index}: delta mismatch")
        require(isinstance(value.get("t"), str) and isinstance(value.get("eta"), str),
                f"Scale {index}: t/eta must be rational strings")
        require(Fraction(value["t"]) > 0 and Fraction(value["eta"]) > 0,
                f"Scale {index}: positive t/eta required")
    return geometry_scales


def generate():
    start = time.perf_counter()
    scales, certs = geometry("70", 1, 74, 512)[1], []
    with ctx.workprec(512):
        union = arb(0)
        for scale in scales:
            proposal = mgf.numerical_scale(74, scale)
            t = Fraction(round(proposal["t"] * 10**12), 10**12)
            eta = Fraction(math.ceil((proposal["eta"] * 1.002 + 1e-9) * 10**12), 10**12)
            certificate, bound = mgf.certify_scale(74, scale, t, eta, bits=512)
            require(certificate["passed"], f"n=74 scale {scale.j}: polynomial proof failed")
            certs.append(certificate)
            union += bound
        require(union < mgf.ball(Fraction(8, 25)), "n=74 upper-tail union is not <.32")
    document = {
        "rigorous": True, "C": 70, "L0": 1, "n": 74, "scale_count": 7,
        "scales": certs, "passed": True, "union_bound_arb": union.str(65),
        "union_bound_midpoint": float(union), "strict_target": "8/25",
        "proposal_method": "Nonrigorous MGF optimization followed by rational rounding and 512-bit Arb polynomial certification. Optimizer is not trusted as a proof.",
        "elapsed_seconds": time.perf_counter() - start,
    }
    validate(document)
    OUT.mkdir(exist_ok=True)
    CERTIFICATE.write_text(json.dumps(document, indent=2) + "\n")


def replay():
    start = time.perf_counter()
    document = json.loads(CERTIFICATE.read_text())
    scales = validate(document)

    def forbid(*args, **kwargs):
        raise ArithmeticError("Optimization forbidden during n=74 certificate replay")

    mgf.numerical_scale = forbid
    mgf.minimize_scalar = forbid
    malformed = copy.deepcopy(document)
    del malformed["scales"][0]
    try:
        validate(malformed)
    except ValueError as exc:
        negative = {"test": "delete_one_scale", "rejected": True, "reason": str(exc)}
    else:
        raise ArithmeticError("Missing-scale negative control incorrectly accepted")

    verified = []
    with ctx.workprec(512):
        union = arb(0)
        for index in range(7):
            stored = document["scales"][index]
            t, eta = Fraction(stored["t"]), Fraction(stored["eta"])
            certificate, bound = mgf.certify_scale(74, scales[index], t, eta, bits=512)
            require(certificate["passed"], f"n=74 scale {index}: replay failed")
            union += bound
            verified.append({"j": index, "k": scales[index].k, "delta": str(scales[index].delta),
                             "t": str(t), "eta": str(eta), "passed": True, "bound_arb": bound.str(65)})
        require(union < mgf.ball(Fraction(8, 25)), "Recomputed upper tail not <.32")
        require(union < mgf.ball(Fraction(1, 3)), "Recomputed upper tail not <1/3")
    audit = {
        "status": "PASS", "C": 70, "L0": 1, "n": 74, "scale_count": 7,
        "scale_indices": list(range(7)), "full_scale_coverage_unique_and_ordered": True,
        "positive_rational_t_eta_checked": True, "geometry_k_delta_rebuilt": True,
        "optimization_executed_in_replay": False, "optimization_calls_disabled": True,
        "all_polynomial_certificates_recomputed": True, "union_bound_arb": union.str(65),
        "union_bound_midpoint": float(union), "strict_target": "8/25", "strictly_below_one_third": True,
        "scales": verified, "negative_control": negative,
        "certificate_sha256": hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest(),
        "mgf_code_sha256": hashlib.sha256((ROOT / "proof_l01/mgf.py").read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "Upper deviation for n=74, all S<=74 and arbitrary independent Poisson means. Combined with separate lower-tail control in the full theorem.",
    }
    (OUT / "n74_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ["status", "n", "scale_count", "union_bound_arb",
                                                "optimization_executed_in_replay", "negative_control"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    replay()


if __name__ == "__main__":
    main()
