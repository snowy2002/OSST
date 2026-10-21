#!/usr/bin/env python3
"""Exercise adaptive precision through a nontrivial ceil, not just clipping."""
from pathlib import Path
import hashlib
import json
import sys
import time
from fractions import Fraction

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from audit_c0_1025 import ball, independent_plan, independent_weights, integer
import paper_reference as reference
from support_size import Fingerprint
from flint import arb,ctx


def main():
    start=time.perf_counter()
    old=reference.C0
    try:
        reference.C0=1025
        with ctx.workprec(4096):
            n=integer(arb(1023).exp(),'floor')+1
            ep=Fraction(1,10)
            plan=independent_plan(n,ep,True)
            observed=n//100
            fp=Fingerprint({1:observed})
            score=arb(observed)
            for s in plan['scales']:
                weights,_,_=independent_weights(s['A'],s['B'],s['k'])
                z=observed*weights[1]
                score=score.max((z-ball(s['delta'])*n)/(1+ball(s['delta'])))
            assert score>observed and score<n
            lower=integer(score,'ceil')
            assert score<lower and score>lower-1 # genuinely noninteger
            original=reference.reference_parameters(n,ep,max_bits=32768)
            actual=reference.reference_test(fp,n,ep,m=original.m,max_bits=32768,max_degree=10000)
            assert actual['decision']=='Yes'
            assert actual['support_lower_bound']==lower
            output=dict(passed=True,case='polynomial wins over D, below n, noninteger score',
                        initial_precision_bits=original.precision_bits,
                        final_precision_bits=actual['precision_bits'],
                        independent_precision_bits=4096,
                        score_divided_by_n=(score/n).str(30),
                        observed_divided_by_n=(arb(observed)/n).str(30),
                        output_digits=len(str(lower)),
                        lower_bound_sha256=hashlib.sha256(str(lower).encode()).hexdigest(),
                        elapsed_seconds=time.perf_counter()-start,
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    finally:
        reference.C0=old
    (HERE/'fractional_output.json').write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
