"""Independent coefficient oracle, full API contract, and ablation event semantics."""
from fractions import Fraction
import numpy as np
import pytest
from flint import arb,ctx
from benchmark_core import Profiles
from c7016.populations import Population
from parameter_search import algorithm as alg
from support_size import Fingerprint
from followup import core


@pytest.mark.parametrize('k',[1,2,9,30])
def test_positive_closed_form_vs_recurrence(k):
    from paper_reference import reference_weights
    with ctx.workprec(768):
        A,B=arb('0.3'),arb(50)
        direct=core.closed_form_weights(A,B,k)
        recurrent=reference_weights(A,B,k)
        assert all(x.overlaps(y) and (x-y).abs_upper()<arb('1e-100') for x,y in zip(direct,recurrent))


@pytest.mark.parametrize('n',[442413,442414,3269018])
def test_scale_enclosures_and_ablation_match_direct(n):
    plan=alg.parameters('70',14,n,'.1')
    low=np.zeros((3,plan.degree),np.int64)
    if plan.degree:low[1,0]=n//10;low[2,1]=n//20
    fp=Profiles(np.array([n,n//2,n//3]),low)
    a,b=core.scale_intervals(fp,plan);x,y=alg.score_intervals(fp,plan)
    assert np.array_equal(x,a.max(axis=1)) and np.array_equal(y,b.max(axis=1))
    for mode in ['full','single','count']+(['no_count'] if plan.scales else []):
        cols=core.columns(plan,mode)
        for i in range(3):
            exact=core.arb_value(fp.observed[i],low[i],plan,mode,bits=1024)
            with ctx.workprec(1024):
                assert exact>=arb(float(a[i,cols].max())) and exact<=arb(float(b[i,cols].max()))


def test_full_api_distinguishes_mean_total_and_clipped_output():
    p=alg.parameters('70',14,1000,'.1')
    fp=Fingerprint({2:1001})
    result=core.test_fingerprint(fp,1000,'.1',mean=p.m)
    assert result['decision']=='No' and result['lower']==1000 and result['M']==2002
    with pytest.raises(ValueError):core.test_fingerprint(fp,1000,'.1',mean=arb(fp.total))
    with pytest.raises(ValueError):core.test_fingerprint(fp,1000,'.1',mean=p.m,L0=13)


def test_gap_has_no_binary_error_but_keeps_coverage():
    p=alg.parameters('70',14,2,'.1')
    pop=Population((1,2),(Fraction(99,100),Fraction(1,200)))
    fp=Profiles(np.array([1,2,3]),np.zeros((3,0),np.int64))
    a,b=core.scale_intervals(fp,p)
    events,_=core.outcomes(fp,p,pop,a,b)
    assert pop.truth(2,'.1')[0]=='Gap'
    assert not events['binary'].any()
    assert np.array_equal(events['joint'],events['over']|events['short'])
