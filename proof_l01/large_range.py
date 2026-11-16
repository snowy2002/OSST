"""Recompute the C=70 half-line certificate without changing old artifacts."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
from parameter_search import certify as c

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_proof_l01'


def main():
    started=time.monotonic()
    c.CHECKS.clear()
    C=c.I(70)
    c.require(C>69 and C<=70,'C range')
    c.require(256*c.I(14)**2/C>1,'J nonnegative')
    c.require(64*c.I(14)**2/(C*c.I(13).exp())<1,'endpoint domain')
    c.require((512/C).sqrt()*3*c.LOG2+c.I(Q(1,14))<6,'degree condition')
    rows=[]
    for i in range(140,400):
        rows.append(c.slab(C,i))
        if i%50==49:
            print('completed through L=',Q(i+1,10),flush=True)
    tail=c.half_line(C,False,40)
    bounds={key:c.imax(tail[key],*(r[key] for r in rows))
            for key in ['upper','lower_finite','lower_general','joint']}
    c.require(bounds['joint']<c.I(Q(9,10000)),'C70 large-range joint risk below 0.0009')
    encoded=c.encode(dict(C=70,L_range='[14,infinity)',passed=True,
                         closed_slabs=rows,half_line=tail,bounds=bounds))
    previous=json.loads((ROOT/'results_parameter_search/certificate.json').read_text())
    c.require(encoded['closed_slabs']==previous['complete_slabs']['70'],'independent fresh reproduction of saved finite values')
    c.require(encoded['half_line']==previous['half_lines']['70'],'independent fresh reproduction of saved half-line values')
    encoded.update(checks=len(c.CHECKS),check_descriptions=c.CHECKS,
                   arithmetic='160-bit dyadic integer/rational outward intervals',
                   runtime_seconds=time.monotonic()-started,
                   source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ['proof_l01/large_range.py','parameter_search/certify.py',
                                    'artifacts/joint_constants_submission/exact_certificate.py',
                                    'artifacts/joint_constants_submission/rational_interval.py',
                                    'results_parameter_search/proof.tex',
                                    'artifacts/joint_constants_submission/submission_proof.tex']})
    OUT.mkdir(exist_ok=True)
    (OUT/'large_range.json').write_text(json.dumps(encoded,indent=2)+'\n')
    print('PASS',len(c.CHECKS),'checks',encoded['bounds']['joint'],flush=True)


if __name__=='__main__':
    main()
