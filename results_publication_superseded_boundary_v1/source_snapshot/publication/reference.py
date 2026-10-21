"""Strict finite-scale reference branches, with explicit confidence amplification."""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np

from benchmark_core import sample_profiles, stable_rng, uniform, headtail
from paper_reference import reference_parameters, reference_test
from support_size import Fingerprint
from publication.core import near_boundary, trial_summary
from publication.protocol import OUT, DESIGN, SEEDS, save
from publication.suites import finish


def majority_repetitions(error, target=Fraction(1,20)):
    p=Fraction(error)
    for R in range(1,501,2):
      bound=sum(Fraction(math.comb(R,k))*p**k*(1-p)**(R-k) for k in range(R//2+1,R+1))
      if bound<=target:return R,bound
    raise ArithmeticError("Amplification limit")


def fixed_occupancy(pop,N,trials,rng):
    """Fixed-size multinomial sampling, not a Poisson substitute."""
    p=np.repeat(pop.probabilities,pop.multiplicities)
    out=[]
    for start in range(0,trials,32):
      counts=rng.multinomial(N,p,size=min(32,trials-start))
      out.extend(np.count_nonzero(counts,axis=1).tolist())
    return np.array(out)


def reference_suite(configs,out=OUT,quick=False):
    d=DESIGN["reference"]
    rows=[]
    Rw,bw=majority_repetitions(Fraction(1,3))
    Rf,bf=majority_repetitions(Fraction(1,10))
    plans=[]
    for n in d["sizes"][:1] if quick else d["sizes"]:
      for eps in d["epsilons"][:1] if quick else d["epsilons"]:
        wp=reference_parameters(n,str(eps))
        if wp.branch!="observed_count":raise ValueError("Reference suite expects the literal small-L branch")
        # FH v4 Section 2.1 applies because epsilon <= n^(-1/128).
        if not eps<=n**(-1/128):raise ValueError("FH small-epsilon premise failed")
        ep=Fraction(str(eps));ratio=10*(n+1)/ep
        N=(ratio.numerator+ratio.denominator-1)//ratio.denominator
        plans.append(dict(n=n,epsilon=eps,wang=wp.as_dict(),wang_repetitions=Rw,
                          wang_total_mean=Rw*wp.m,wang_amplified_bound=str(bw),
                          fh_branch="Proposition 2.2, n+1 tester, fixed count",fh_per_run_samples=N,
                          fh_repetitions=Rf,fh_total_samples=Rf*N,fh_amplified_bound=str(bf),
                          fh_single_run_bound="1/10 from the Markov argument in its proof"))
        for scenario,pop in [("uniform_null",uniform(n)),("uniform_boundary",uniform(math.ceil(n/(1-eps))+1)),
                             ("exact_diffuse_boundary",near_boundary(n,eps)),("rare_null",headtail(.999,n-1))]:
          T=40 if quick else d["trials"]
          truth=pop.truth(n,eps)
          votes=np.zeros(T,np.int64)
          for rep in range(Rw):
            fp=sample_profiles(pop,wp.m,T,2,stable_rng(SEEDS["reference"],"wang",n,eps,scenario,rep))
            votes+=fp.observed>n
            # A genuine per-category Poisson draw audits the original entrypoint.
            # Never pretend that the compressed >K bin supplies a real total M.
            if rep==0:
              rates=wp.m*np.repeat(pop.probabilities,pop.multiplicities)
              counts=stable_rng(SEEDS["reference"],"scalar_audit",n,eps,scenario).poisson(rates)
              fingerprint=Fingerprint.from_counts(counts)
              answer=reference_test(fingerprint,n,str(eps),m=wp.m)
              if (answer["decision"]=="No")!=bool(fingerprint.observed>n):raise AssertionError(answer)
          rejection=votes>Rw//2
          wang_votes=votes.copy()
          rows.append(dict(method="wang_strict_amplified",scenario=scenario,n=n,epsilon=eps,truth=truth,
                           sampling="poisson",cost_metric="prescribed_total_poisson_mean",repetitions=Rw,
                           **trial_summary(rejection,np.full(T,Rw*wp.m),truth)))
          votes=np.zeros(T,np.int64)
          for rep in range(Rf):
            observed=fixed_occupancy(pop,N,T,stable_rng(SEEDS["reference"],"fh",n,eps,scenario,rep))
            votes+=observed>n
          rejection=votes>Rf//2
          np.savez_compressed(out/"raw"/f"reference_n{n}_e{eps:g}_{scenario}.npz",wang_votes=wang_votes,fh_votes=votes)
          rows.append(dict(method="fh_strict_small_epsilon_amplified",scenario=scenario,n=n,epsilon=eps,truth=truth,
                           sampling="fixed",cost_metric="actual_prescribed_fixed_samples",repetitions=Rf,
                           **trial_summary(rejection,np.full(T,Rf*N),truth)))
          print(f"strict references n={n} eps={eps} {scenario}",flush=True)
    finish(out,"reference",rows)
    save(out/"reference_plans.json",plans)
