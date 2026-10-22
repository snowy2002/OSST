"""Exactly normalized finite populations and grouped Poisson fingerprints."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from scipy.stats import poisson

from benchmark_core import Profiles, headtail
from c1025.core import scenarios as previous_scenarios


@dataclass(frozen=True)
class Population:
    multiplicities: tuple[int, ...]
    probabilities: tuple[Fraction, ...]

    def __post_init__(self):
        pairs = [(int(g), Fraction(p)) for g, p in zip(self.multiplicities, self.probabilities)]
        if (len(self.multiplicities) != len(self.probabilities) or not pairs
                or any(g <= 0 or p <= 0 for g, p in pairs)):
            raise ValueError('Positive group sizes and probabilities required')
        mass = sum(g*p for g, p in pairs)
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        object.__setattr__(self, 'multiplicities', tuple(g for g, _ in pairs))
        object.__setattr__(self, 'probabilities', tuple(p/mass for _, p in pairs))
        if self.support > 2**53:
            raise ValueError('This experimental sampler requires support <=2^53')

    @property
    def support(self):
        return sum(self.multiplicities)

    @classmethod
    def from_float_population(cls, population):
        return cls(tuple(map(int, population.multiplicities)),
                   tuple(Fraction.from_float(float(p)) for p in population.probabilities))

    @classmethod
    def from_counts(cls, counts):
        counts = np.asarray(counts)
        if counts.ndim != 1 or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError('Integer category abundances required')
        values, repeats = np.unique(counts[counts > 0], return_counts=True)
        return cls(tuple(map(int, repeats)), tuple(Fraction(int(v)) for v in values))

    def tail(self, n):
        remaining = n
        tail = Fraction(0)
        for g, p in zip(self.multiplicities, self.probabilities):
            keep = max(0, min(g, remaining))
            remaining -= keep
            tail += (g-keep)*p
        return tail

    def effective(self, epsilon):
        remainder = Fraction(str(epsilon))
        discard = 0
        for g, p in zip(self.multiplicities[::-1], self.probabilities[::-1]):
            if g*p <= remainder:
                discard += g
                remainder -= g*p
            else:
                discard += min(g, remainder//p)
                break
        return max(1, self.support-int(discard))

    def truth(self, n, epsilon):
        eps = Fraction(str(epsilon)); tail = self.tail(n)
        label = 'Yes' if self.support <= n else 'No' if tail >= eps else 'Gap'
        return label, tail, self.effective(eps)

    def split_categories(self, copies):
        """Synthetic clones of each label; this is not additional natural data."""
        return Population(tuple(g*copies for g in self.multiplicities),
                          tuple(p/copies for p in self.probabilities))

    def as_dict(self):
        return dict(multiplicities=list(self.multiplicities),
                    probabilities_exact=list(map(str, self.probabilities)), support=self.support)


def synthetic_cases(n, eps):
    cases = [(name, Population.from_float_population(p))
             for name, p in previous_scenarios(n, float(eps))]
    if n > 1:
        for fraction in [Fraction(1, 10), Fraction(1, 2), Fraction(1)]:
            tail = Fraction(str(eps))*fraction
            cases.append((f'epsilon_scaled_rare_{fraction}_null',
                          Population((1, n-1), (1-tail, tail/(n-1)))))
    for name, pop in cases:
        expected = 'Yes' if name.endswith('null') else 'Gap' if name.startswith('gap') else 'No'
        if pop.truth(n, eps)[0] != expected:
            raise ArithmeticError(f'Exact construction label mismatch: {name}, {n}, {eps}')
    return cases


def sample_profiles(pop, mean, trials, degree, rng):
    """Multinomial(g; Poi(mean*p) bins 0..K, >K), independently per group.

    This is the exact sufficient-statistic law in real arithmetic. SciPy/NumPy
    probabilities and PRNG are binary64 approximations. The >K bucket retains
    its number of labels, not their total observations; M is not reconstructed.
    """
    if not math.isfinite(mean) or mean <= 0 or degree < 0 or trials < 1:
        raise ValueError('Invalid sampling parameters')
    low = np.zeros((trials, degree), dtype=np.int64)
    observed = np.zeros(trials, dtype=np.int64)
    max_mass_error = 0.
    for g, p in zip(pop.multiplicities, pop.probabilities):
        lam = mean*float(p)
        if degree == 0:
            observed += rng.binomial(g, -np.expm1(-lam), size=trials)
            continue
        bins = np.r_[poisson.pmf(np.arange(degree+1), lam), poisson.sf(degree, lam)]
        mass = float(bins.sum())
        max_mass_error = max(max_mass_error, abs(mass-1))
        if not np.isfinite(mass) or abs(mass-1) > 1e-10 or np.any(bins < 0):
            raise ArithmeticError('Invalid Poisson bin probabilities')
        bins /= mass
        # Put the largest cell last: NumPy assigns the residual there. This
        # avoids turning ~1e-15 normalization error into false ultra-rare labels
        # when the population has billions/trillions of categories.
        order = np.r_[np.flatnonzero(np.arange(len(bins)) != np.argmax(bins)), np.argmax(bins)]
        draws = rng.multinomial(g, bins[order], size=trials)
        inverse = np.argsort(order)
        observed += g-draws[:, inverse[0]]
        low += draws[:, inverse[1:degree+1]]
    return Profiles(observed, low), max_mass_error


def expected_profiles(pop, mean, degree):
    D = sum(g*(-np.expm1(-mean*float(p))) for g, p in zip(pop.multiplicities, pop.probabilities))
    low = sum((g*poisson.pmf(np.arange(1, degree+1), mean*float(p))
               for g, p in zip(pop.multiplicities, pop.probabilities)), np.zeros(degree))
    return D, low
