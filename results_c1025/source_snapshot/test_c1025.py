"""Regression checks for explicit C0, occupancy coupling and exact truth."""
from fractions import Fraction
import numpy as np
import pytest
from benchmark_core import uniform,stable_rng
from paper_reference import reference_parameters,reference_test
from support_size import Fingerprint
from c1025.core import budget,sample_path,scenarios,truth,count_decisions,exact_occupancy_cdf

def test_explicit_constant_does_not_mutate_original():
    old=reference_parameters(100,'0.1');new=reference_parameters(100,'0.1',c0=1025)
    assert old.sampling_constant==2**26 and new.sampling_constant==1025
    assert 65000<old.m/new.m<66000
    assert reference_parameters(100,'0.1').m==old.m
    assert new.as_dict()['C0']==1025
    result=reference_test(Fingerprint({1:101}),100,'0.1',m=new.m,c0=1025)
    assert result['decision']=='No' and result['C0']==1025
    with pytest.raises(ValueError):reference_test(Fingerprint({1:100}),100,'0.1',m=old.m,c0=1025)
    with pytest.raises(ValueError):reference_parameters(100,'0.1',c0=True)

def test_n1_exact_rational_rounding():
    assert reference_parameters(1,Fraction(1,4),c0=1025).m==4100

def test_grouped_joint_law_and_pgf():
    pop=uniform(5);paths=sample_path(pop,[1,10,10000],1000,stable_rng(56))
    assert np.all(paths[1]<=paths[10]) and np.all(paths[10]<=paths[10000])
    assert np.all(paths[10000]==5)
    # P(D=0) is the no-observation probability exp(-m), independent of p.
    assert exact_occupancy_cdf(pop,3,0)==pytest.approx(np.exp(-3),rel=1e-12)
    plan=reference_parameters(3,'0.1',c0=1025)
    reject,lower=count_decisions(np.array([0,1,3,4]),plan)
    assert reject.tolist()==[False,False,False,True] and lower.tolist()==[1,1,3,3]

@pytest.mark.parametrize('n',[1,2,1000,10**12])
@pytest.mark.parametrize('eps',[.001,.25])
def test_exact_promise_labels_and_budget(n,eps):
    for name,pop in scenarios(n,eps):
        label,tail,eff=truth(pop,n,eps)
        assert 1<=eff<=pop.support
        if label=='No':assert tail>=Fraction(str(eps))
        if label=='Gap':assert pop.support>n and tail<Fraction(str(eps))
    assert budget(n,str(eps),'count_chernoff_99')<budget(n,str(eps),'C1025')
