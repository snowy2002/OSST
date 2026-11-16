"""Unchanged multiscale statistic with a real Poisson mean.

Removing the legacy b<=1 guard extends the domain of the same polynomial; it
does not establish the missing uniform finite-n probability certificate.
"""
from flint import ctx
from c7016.algorithm import ball
from paper_reference import exact_inputs
from parameter_search.algorithm import Plan, constant, geometry, compiled, score_intervals
from followup.core import scale_intervals, outcomes, exact_output, arb_value


def parameters(C='70', L0=1, n=None, epsilon=None, bits=512):
    n, eps = exact_inputs(n, epsilon)
    if isinstance(L0, bool) or int(L0) != L0:
        raise ValueError('L0 must be an integer')
    with ctx.workprec(bits):
        L, scales = geometry(str(C), int(L0), n, bits)
        m = constant(C)*n/(ball(eps)*L)
        return Plan(str(C), int(L0), n, eps, L, m, scales, bits)
