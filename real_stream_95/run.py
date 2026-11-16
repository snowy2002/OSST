"""Replay actual empirical observation labels, with and without replacement.

Parameters depend on available budget only, never on the reference richness.
Each replicate constructs a literal label stream. Batched bincount is exactly
the sufficient statistic from reading those labels one at a time.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from flint import arb, ctx
from scipy.stats import beta

from benchmark_core import Profiles
from real_l01 import algorithm as alg
from real_l01.run import profiles_from_draws
from followup.core import exact_output, scale_intervals, compare
from guaranteed import certified_count_budget
from c7016.algorithm import ball

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results_real_stream_95_20261116'
FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
EPS = Fraction(1, 10)
REPEATS = 23


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def populations():
    records = json.loads((ROOT/'results_real_l01_20261104/prepared/populations.json').read_text())
    for meta in json.loads((ROOT/'data/c1025/datasets.json').read_text()):
        with np.load(ROOT/f'data/c1025/prepared/{meta["dataset"]}.npz', allow_pickle=False) as z:
            counts = z['counts'].astype(np.int64)
        records.append(dict(id=meta['dataset'], dataset=meta['dataset'], counts=counts.tolist(),
                            N=int(counts.sum()), S=len(counts), group={'historical_population': True}))
    for r in records:
        c = np.asarray(r['counts'])
        if np.any(c <= 0) or c.sum() != r['N'] or len(c) != r['S']:
            raise ValueError('Invalid empirical abundance vector')
    return records


def effective(counts, eps=EPS):
    total, tail = sum(counts), sum(counts)
    for k, c in enumerate(sorted(counts, reverse=True), 1):
        tail -= c
        if Fraction(tail, total) <= eps:
            return k


@lru_cache(maxsize=4096)
def budget_plan(B, copies=1, method='osst'):
    # P(Poi(mu)>B) <= exp(-(B-mu)^2/(2B)) <= 1/1000.
    # Reserve chosen by integers only: d^2>=14B and ln(1000)<7.
    d = math.isqrt(14*B)
    if d*d < 14*B:
        d += 1
    limit = B-d
    if limit < 1:
        return None
    if method == 'count':
        # Lower-tail Chernoff proposal, with a final direct Arb certificate.
        x = float(EPS)*limit
        n = max(0, math.floor(x-math.sqrt(2*x*math.log(1/.049))))
        if n < 1:
            return None
        with ctx.workprec(256):
            tail = (limit*ball(EPS)).gamma_upper(n+1, regularized=1)
            if not tail < ball(Fraction(49, 1000)):
                raise ArithmeticError('Count risk certificate failed')
        return dict(n=n, m=limit, copies=1, budget=B, mean_limit=limit,
                    tail_certificate=tail.str(50))
    if copies*700 > limit:
        return None
    left, right = 1, max(2, limit+1)
    while left+1 < right:
        n = (left+right)//2
        if copies*700*n/(1+math.log(n)) <= limit:
            left = n
        else:
            right = n
    plan = alg.parameters('70', 1, left, EPS)
    with ctx.workprec(512):
        if not copies*plan.m <= limit:
            raise ArithmeticError('Budget selection not certified')
    return dict(n=left, m=float(plan.m), copies=copies, budget=B, mean_limit=limit)


def score(draws, n):
    plan = alg.parameters('70', 1, n, EPS)
    fp = profiles_from_draws(draws, plan.degree)
    a, b = scale_intervals(fp, plan)
    lo, hi = a.max(axis=1), b.max(axis=1)
    out = np.clip(np.ceil(lo), 1, n).astype(np.int64)
    ambiguous = np.flatnonzero(out != np.clip(np.ceil(hi), 1, n))
    for i in ambiguous:
        out[i] = exact_output(fp.observed[i], fp.low[i], plan)['lower']
    reject, refined = compare(fp, plan, lo, hi, n, 'full')
    return out, reject, fp, len(ambiguous)+refined


def interval(k, n, alpha=.05):
    return (0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k)))


def summarize(values, M, B, r, mode, fraction, method, n, reject, theoretical):
    valid = np.isfinite(values)
    N, S = r['N'], r['S']
    eff = effective(r['counts'])
    v = values[valid]
    close = valid & (np.abs(values-S) <= .05*S)
    exact = valid & (values == S)
    lower = min(eff, n) if n is not None else None
    upper = min(S, n) if n is not None else None
    coverage = valid & (values >= lower) & (values <= upper) if n is not None else np.zeros(len(values), bool)
    tail = Fraction(sum(sorted(r['counts'], reverse=True)[n:]), N) if n is not None else None
    truth = ('Yes' if S <= n else 'No' if tail >= EPS else 'Gap') if n is not None else 'NA'
    joint = coverage.copy()
    if truth == 'Yes':
        joint &= ~reject
    elif truth == 'No':
        joint &= reject
    ci = interval(int(close.sum()), len(values))
    return dict(population=r['id'], dataset=r['dataset'], sampling=mode,
        fraction=fraction, budget=B, N=N, S=S, effective_90=eff, method=method,
        n=n, epsilon=str(EPS), truth=truth, trials=len(values), completed=int(valid.sum()),
        unavailable=int((~valid).sum()), mean_samples=float(M.mean()),
        mean=float(v.mean()) if len(v) else None,
        median=float(np.median(v)) if len(v) else None,
        q025=float(np.quantile(v,.025)) if len(v) else None,
        q975=float(np.quantile(v,.975)) if len(v) else None,
        bias=float(np.mean(v-S)) if len(v) else None,
        mae=float(np.mean(np.abs(v-S))) if len(v) else None,
        rmse=float(np.sqrt(np.mean((v-S)**2))) if len(v) else None,
        mape=float(np.mean(np.abs(v-S))/S) if len(v) else None,
        within5_count=int(close.sum()), within5_rate=float(close.mean()),
        within5_ci95_lo=ci[0], within5_ci95_hi=ci[1],
        exact_rate=float(exact.mean()), over_rate=float(np.mean(valid & (values>S))),
        lower_interval_coverage=float(coverage.mean()) if n is not None else None,
        joint_success_rate=float(joint.mean()) if n is not None else None,
        theory_joint_success_iid=theoretical if mode=='iid_with_replacement' else None,
        theoretical_target='clipped effective-support interval and promised binary decision; NOT 5% relative error',
        output_type='point-estimate heuristic' if method=='chao1_bc' else
            'observed species' if method=='observed' else 'certified-form lower bound')


def run_population(r, trials, out):
    counts = np.asarray(r['counts'], dtype=np.int64)
    N, S = int(counts.sum()), len(counts)
    pool = np.repeat(np.arange(S, dtype=np.int32), counts)
    budgets = [max(1, int(N*f)) for f in FRACTIONS]
    plans = [{k:budget_plan(B, copies=23 if k=='amplified' else 1,
                          method='count' if k=='count95' else 'osst')
              for k in ['single','amplified','count95']} for B in budgets]
    rows, refinements = [], 0
    for mode in ['without_replacement', 'iid_with_replacement']:
        # All algorithms see prefixes of the same stream, pairing comparisons.
        census = np.zeros((6,trials,S), dtype=np.int64)
        draws = {k:np.zeros((6,trials,23 if k=='amplified' else 1,S),dtype=np.int64)
                 for k in ['single','amplified','count95']}
        sizes = {k:np.zeros(v.shape[:-1], dtype=np.int64) for k,v in draws.items()}
        overflow = {k:np.zeros((6,trials),bool) for k in draws}
        for t in range(trials):
            rng = np.random.default_rng(seed('direct_stream_v1',r['id'],mode,t))
            labels = rng.permutation(pool) if mode=='without_replacement' else pool[rng.integers(0,N,size=N)]
            prev, accum = 0, np.zeros(S,np.int64)
            for fidx,B in enumerate(budgets):
                accum += np.bincount(labels[prev:B], minlength=S)
                census[fidx,t] = accum
                prev = B
                for k in draws:
                    p = plans[fidx][k]
                    if p is None:
                        continue
                    rr = np.random.default_rng(seed('poisson_sizes_v1',r['id'],mode,t,fidx,k))
                    ms = rr.poisson(p['m'], size=p['copies'])
                    sizes[k][fidx,t] = ms
                    if ms.sum() > B:
                        overflow[k][fidx,t] = True
                        continue  # Inconclusive, retained in all-rate denominators.
                    start = 0
                    for j,M in enumerate(ms):
                        draws[k][fidx,t,j] = np.bincount(labels[start:start+M], minlength=S)
                        start += M
            if mode=='without_replacement' and not np.array_equal(accum,counts):
                raise ArithmeticError('A shuffled full record stream changed its multiset')
        raw = {'census_counts':census}
        for k in draws:
            raw[k+'_counts'] = draws[k]
            raw[k+'_sizes'] = sizes[k]
            raw[k+'_overflow'] = overflow[k]
        for fidx,(fraction,B) in enumerate(zip(FRACTIONS,budgets)):
            v = census[fidx]
            D = np.count_nonzero(v,axis=1)
            f1,f2 = (v==1).sum(axis=1),(v==2).sum(axis=1)
            chao = D + f1*(f1-1)/(2*(f2+1))
            for method,vals in [('observed',D),('chao1_bc',chao)]:
                rows.append(summarize(vals,np.full(trials,B),B,r,mode,fraction,method,None,None,None))
                raw[f'{fidx}_{method}'] = vals
            for k in draws:
                p = plans[fidx][k]
                method = {'single':'C70_L1','amplified':'C70_L1_median23','count95':'count95'}[k]
                methods = [method,'C70_L14'] if k=='single' else [method]
                if p is None:
                    for name in methods:
                        vals = np.full(trials,np.nan)
                        rows.append(summarize(vals,np.zeros(trials),B,r,mode,fraction,name,None,None,None))
                        raw[f'{fidx}_{name}'] = vals
                    continue
                n = p['n']
                arr = draws[k][fidx].reshape(-1,S)
                D = np.count_nonzero(arr,axis=1)
                if k=='count95':
                    vals, reject = np.clip(D,1,n), D>n
                    theory = .95
                else:
                    vals,reject,fp,nref = score(arr,n)
                    refinements += nref
                    if k=='amplified':
                        vals = np.median(vals.reshape(trials,23),axis=1)
                        reject = reject.reshape(trials,23).sum(axis=1)>=12
                        theory = 1-float(amplification_failure())-.001
                    else:
                        theory = .668
                vals = vals.astype(float)
                vals[overflow[k][fidx]] = np.nan
                totalM = sizes[k][fidx].sum(axis=1)
                consumed = np.where(overflow[k][fidx],0,totalM)
                rows.append(summarize(vals,consumed,B,r,mode,fraction,method,n,reject,theory))
                raw[f'{fidx}_{method}'] = vals
                raw[f'{fidx}_{method}_reject'] = reject
                if k=='single':
                    # This benchmark's maximum possible n remains below e^13.
                    if 1+math.log(n)>=14:
                        raise ValueError('L14 needs its actual polynomial branch')
                    old = np.clip(D,1,n).astype(float)
                    old[overflow[k][fidx]] = np.nan
                    rows.append(summarize(old,consumed,B,r,mode,fraction,'C70_L14',n,D>n,.9981))
                    raw[f'{fidx}_C70_L14'] = old
                    raw[f'{fidx}_C70_L14_reject'] = D>n
        np.savez_compressed(out/'raw'/f'{r["id"]}_{mode}.npz', **raw)
    dump(out/'plans'/f'{r["id"]}.json',dict(population=r['id'],plans=plans,budgets=budgets,refinements=refinements))
    pd.DataFrame(rows).to_csv(out/'cells'/f'{r["id"]}.csv',index=False)
    return rows


def amplification_failure():
    p = Fraction(331,1000)
    return sum(Fraction(math.comb(23,k))*p**k*(1-p)**(23-k) for k in range(12,24))


def historical(out):
    base = ROOT/'results_real_l01_20261104'
    results = json.loads((base/'results.json').read_text())
    rows = []
    for r in results:
        if r['regime'] not in ['at_support','twice_support']:
            continue
        with np.load(base/'raw'/f'{r["id"]}.npz') as z:
            for key in ['full','count','single']:
                values = z[key+'_lower']
                S = r['S']
                rows.append(dict(population=r['population_id'],dataset=r['dataset'],
                    regime=r['regime'],epsilon=r['epsilon'],n=r['n'],S=S,eff=r['eff'],
                    m=r['m'],N=r['source_N'],method=key,mean=float(values.mean()),
                    q025=float(np.quantile(values,.025)),q975=float(np.quantile(values,.975)),
                    mape=float(np.abs(values-S).mean()/S),
                    within5_rate=float((np.abs(values-S)<=.05*S).mean()),
                    exact_rate=float((values==S).mean()),over_rate=float((values>S).mean()),
                    oracle_n=True))
    pd.DataFrame(rows).to_csv(out/'historical_richness_diagnostic.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--trials',type=int,default=128)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();out=args.out
    for d in ['raw','cells','plans']:
        (out/d).mkdir(parents=True,exist_ok=True)
    pops=populations()
    if args.limit:
        pops=pops[:args.limit]
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
        ROOT/'real_l01/algorithm.py',ROOT/'followup/core.py',ROOT/'parameter_search/algorithm.py',
        ROOT/'real_l01/run.py',ROOT/'guaranteed.py',ROOT/'c7016/algorithm.py']}
    protocol=dict(version=1,populations=pops,trials=args.trials,fractions=FRACTIONS,epsilon=str(EPS),
        randomization='Literal recorded labels; uniform permutations and iid uniform record-index draws; separate deterministic seeds.',
        parameters='Only N and the fixed available-budget fraction enter plan selection; S/eff never enter selection.',
        confidence='128 fixed independent replicate streams per population and model; methods and fractions paired. Pointwise two-sided 95% CP intervals, plus separate Bonferroni family across all rows. Not simultaneous across endpoints.',
        sampling_caveat='No iid OSST probability guarantee for without-replacement field-file replay. Full-file richness is a reference, not unknown ecosystem truth.',
        unavailable='Keep budget-unavailable and Poisson-cap-abort trials as no output, counted unsuccessful in total success rates.',
        probability_statement='Risk guarantees concern a clipped effective-support interval and binary promises, never relative error of richness.',
        amplification=dict(repeats=23,failure_exact=str(amplification_failure()),
                           failure_float=float(amplification_failure()),cap_failure_bound=.001),
        chao='Unclipped bias-corrected Chao1: D+F1*(F1-1)/(2*(F2+1)); descriptive point estimate, no CI or coverage claim.',
        source_sha256=sources)
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Protocol differs; choose a new output directory')
    dump(path,protocol)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    start=time.monotonic();allrows=[]
    for i,r in enumerate(pops):
        cell=out/'cells'/f'{r["id"]}.csv'
        if cell.exists():
            rows=pd.read_csv(cell).to_dict('records')
        else:
            rows=run_population(r,args.trials,out)
        allrows.extend(rows)
        print(f'{i+1}/{len(pops)} {r["id"]} N={r["N"]} elapsed={time.monotonic()-start:.1f}s',flush=True)
    frame=pd.DataFrame(allrows)
    intervals=[interval(int(k),args.trials,.05/len(frame)) for k in frame.within5_count]
    frame['within5_sim95_lo']=[v[0] for v in intervals]
    frame['within5_sim95_hi']=[v[1] for v in intervals]
    frame.to_csv(out/'results.csv',index=False)
    historical(out)
    dump(out/'execution.json',dict(populations=len(pops),independent_streams=len(pops)*2*args.trials,
        literal_labels_generated=sum(r['N'] for r in pops)*2*args.trials,
        method_budget_rows=len(frame),method_budget_evaluations=len(frame)*args.trials,
        paired_fractions_and_methods=True,seconds=time.monotonic()-start))


if __name__=='__main__':
    main()
