#!/usr/bin/env python3
"""Interval checks for constants in c0_reduction_proof.tex, not a formal proof checker.

Run from OSST: .venv/bin/python artifacts/verify_c0_reduction.py
Uses exact rationals and 4096-bit Arb balls; the tiny positive increment must
never be represented as a binary64 addition to 1024.
"""
import json
from pathlib import Path
from flint import arb, ctx, fmpq


def main():
    checks = []
    def gt(name, value, bound):
        assert value > bound, (name, value, bound)
        checks.append(dict(name=name, relation=">", value=value.str(35), bound=bound.str(35) if isinstance(bound, arb) else str(bound)))
    def lt(name, value, bound):
        assert value < bound, (name, value, bound)
        checks.append(dict(name=name, relation="<", value=value.str(35), bound=bound.str(35) if isinstance(bound, arb) else str(bound)))

    with ctx.workprec(4096):
        eta = arb(fmpq(9, 4 * 10**219))
        c = arb(1024) + eta
        h = eta / 1024
        t = arb(fmpq(1022999, 1000))
        log100 = arb(100).log()
        gt("strictly positive increment", c - 1024, 0)
        # Three ranges n <= e^40; e^40 < n <= e^t; e^t < n < e^1023.
        x1 = arb(1024) / 41
        gt("small-n Chernoff exponent", x1 - 1 - x1.log(), log100)
        h2 = arb(1024) / (t + 1) - 1
        gt("middle-n Chernoff exponent", arb(40).exp() * h2**2 / (2*(1+h2)), log100)
        gt("near-boundary Chernoff exponent", t.exp() * h**2 / (2*(1+h)), log100)
        # Uniform bounds for the polynomial branch L >= 1024.
        lt("degree divided by L", 3*arb(2).log()/arb(2).sqrt() + arb(1)/1024, arb(3)/2)
        lt("b upper bound at L=1024", arb(1024)**2 * arb(-1023).exp()/16, 1)
        lt("joint concentration bound at L=1024", arb(128)/3 * arb(1).exp() * arb(1024)**4 * (-arb(23)/32*1024).exp(), arb(1)/100)
        lt("large expected occupancy count error", 2*arb(-1023).exp(), arb(1)/100)
        lt("heavy top-r case", arb(-31*1024-1).exp() + (-arb(1024)*arb(1023).exp()/1024).exp(), arb(1)/100)
        # Exact integer immediately below the branch cutoff.
        n = arb(1023).exp().ceil().unique_fmpz() - 1
        ell = arb(n).log()+1
        gt("lower-bound n logarithm", ell, 1023)
        lt("lower-bound n is in count branch", ell, 1024)
        d = 1024*arb(n)/ell - arb(n)
        gt("Poisson mean excess for C=1024", d, 0)
        lt("Poisson mean excess bound", d, arb(1)/1023)
        gt("counterexample error lower limit", (-arb(1)/1023).exp()/16, arb(1)/100)

    output = dict(passed=True, precision_bits=4096,
                  C_exact="1024 + 9/(4*10^219)",
                  scope="Scalar inequalities only. The analytic proof is in c0_reduction_proof.tex; no simulations are evidence for its universal claims.",
                  checks=checks)
    dest = Path(__file__).with_name("c0_reduction_checks.json")
    dest.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
