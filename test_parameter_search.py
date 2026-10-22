import numpy as np
import pytest
from flint import arb,ctx
from benchmark_core import Profiles,stable_rng
from c7016 import algorithm as old
from parameter_search import algorithm as new


@pytest.mark.parametrize('C,L0,n',[('star',14,442414),('70',14,442414),('70',15,1202605),
                                 ('70',16,3269018),('20',14,10**12),('6',14,10**12)])
def test_generic_enclosures_and_boundary(C,L0,n):
    plan=new.parameters(C,L0,n,'.1')
    rng=stable_rng(202610224199,C,L0,n)
    low=rng.integers(0,100,size=(8,plan.degree),dtype=np.int64)
    low[0]=0
    D=low.sum(axis=1)+rng.integers(0,50,size=8);D[0]=n
    fp=Profiles(D,low)
    a,b=new.score_intervals(fp,plan)
    for i in range(8):
        exact=new.arb_score(D[i],low[i],plan,1024)
        with ctx.workprec(1024):
            assert exact>=arb(float(a[i])) and exact<=arb(float(b[i]))
    result,ref=new.compare(fp,plan,a,b,n)
    assert not result[0]


def test_unchanged_70_16_results_and_star_mean():
    for n in [1,3269017,3269018,10**12]:
        p=old.parameters(n,'.01');q=new.parameters('70',16,n,'.01')
        assert p.m.overlaps(q.m) and p.degree==q.degree
        fp=Profiles(np.array([n]),np.zeros((1,q.degree),dtype=np.int64))
        a,b,_,_=old.score_intervals(fp,p);c,d=new.score_intervals(fp,q)
        assert np.array_equal(a,c) and np.array_equal(b,d)
    with ctx.workprec(512):
        p=new.parameters('star',14,1000,'.01')
        assert p.m.overlaps(144*arb(2).log()**2*1000/(arb('.01')*(1+arb(1000).log())))


def test_all_branch_boundaries():
    for L0,boundary in [(14,442414),(15,1202605),(16,3269018),(18,24154953)]:
        assert new.parameters('70',L0,boundary-1,'.1').branch=='observed_count'
        assert new.parameters('70',L0,boundary,'.1').branch=='chebyshev'
