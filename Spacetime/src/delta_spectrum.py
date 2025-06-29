# Utility functions for Δ-spectrum analysis of near-misses to Beal's conjecture.
# This module follows the Preservation Constraint Equation (PCE) paradigm and
# mirrors physical resonance searches. The implementation closely matches the
# production script provided by the user, but is organized into smaller
# functions for clarity.

from __future__ import annotations

import json
import math
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


@dataclass
class NearMiss:
    """Container for an approximate Beal triple.

    Real-world analogy: each triple is like a resonance candidate in a
    laboratory experiment.  The ``delta`` value measures how far the
    configuration is from a true solution.
    """

    A: int
    B: int
    C_approx: int
    x: int
    y: int
    z: int
    delta: int
    smallest_prime: int
    gcd_abc: int
    lambda4_weight: float = 1.0

    @property
    def is_coprime(self) -> bool:
        return self.gcd_abc == 1

    @property
    def log_delta(self) -> float:
        return math.log10(max(1, self.delta))


class OptimizedPrimeCache:
    """Cache for smallest prime factor lookups.

    A physical analogy is a lookup table of resonance modes that speeds up
    repeated searches in an experiment.
    """

    def __init__(self, cache_limit: int = 1_000_000) -> None:
        self.cache: Dict[int, int] = {}
        self.cache_limit = cache_limit

    def smallest_prime_factor(self, n: int) -> int:
        """Return the smallest prime factor of ``n`` using caching."""
        if n in self.cache:
            return self.cache[n]
        if n == 0:
            result = 0
        else:
            n_abs = abs(n)
            if n_abs == 1:
                result = 1
            elif n_abs % 2 == 0:
                result = 2
            else:
                result = n_abs
                i = 3
                while i * i <= n_abs:
                    if n_abs % i == 0:
                        result = i
                        break
                    i += 2
        if n <= self.cache_limit:
            self.cache[n] = result
        return result


class DeltaSpectrumAnalyzer:
    """Compute Δ-spectrum statistics for Beal near-misses."""

    def __init__(self, invariants_path: Optional[Path] = None,
                 prime_cache_limit: int = 1_000_000) -> None:
        self.invariants_path = invariants_path
        self.lambda4_items: List[dict] = []
        self.near_misses: List[NearMiss] = []
        self.prime_cache = OptimizedPrimeCache(prime_cache_limit)
        if invariants_path and invariants_path.exists():
            self._load_invariants()

    def _load_invariants(self) -> None:
        """Load invariant catalog to determine λ=4 resonance weight."""
        with self.invariants_path.open() as f:
            data = json.load(f)
        if isinstance(data, dict) and 'invariants' in data:
            invariants = []
            for cat in data['invariants'].values():
                if isinstance(cat, dict):
                    for sub in cat.values():
                        if isinstance(sub, list):
                            invariants.extend(sub)
                elif isinstance(cat, list):
                    invariants.extend(cat)
        else:
            invariants = data if isinstance(data, list) else []

        def lambda4_compatible(item: dict) -> bool:
            txt = json.dumps(item).lower()
            patterns = [
                'λ=4', 'lambda=4', 'eigenmode 4', 'lambda_4', '"lambda": 4'
            ]
            return any(p in txt for p in patterns)

        self.lambda4_items = [inv for inv in invariants if lambda4_compatible(inv)]

    def _compute_lambda4_weight(self, A: int, B: int, C: int) -> float:
        """Heuristic weight for λ=4 resonance."""
        if not self.lambda4_items:
            return 1.0
        sample = json.dumps(self.lambda4_items[:5])
        if str(A) in sample or str(B) in sample:
            return 2.0
        return 1.0

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _calculate_entropy(self, counts: List[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        pvals = [c / total for c in counts if c > 0]
        return -sum(p * math.log2(p) for p in pvals)

    def search_near_misses(self, x: int = 3, y: int = 5, z: int = 7,
                           limit: int = 50, coprime_only: bool = True,
                           max_delta: Optional[int] = None) -> List[NearMiss]:
        """Locate triples ``A^x + B^y ≈ C^z`` within ``limit``.

        A direct analogue is tuning two experimental knobs ``A`` and ``B`` to
        see whether the outcome resonates with some ``C`` mode.
        """
        self.near_misses.clear()
        for A in range(1, limit + 1):
            for B in range(A, limit + 1):
                if coprime_only and math.gcd(A, B) != 1:
                    continue
                S = A ** x + B ** y
                C_float = S ** (1 / z)
                for C in (int(C_float), int(C_float) + 1):
                    if C <= 0:
                        continue
                    delta = abs(S - C ** z)
                    if delta == 0:
                        continue
                    if max_delta and delta > max_delta:
                        continue
                    sp = self.prime_cache.smallest_prime_factor(delta)
                    gcd_abc = math.gcd(math.gcd(A, B), C)
                    weight = self._compute_lambda4_weight(A, B, C)
                    self.near_misses.append(NearMiss(
                        A=A, B=B, C_approx=C, x=x, y=y, z=z,
                        delta=delta, smallest_prime=sp,
                        gcd_abc=gcd_abc, lambda4_weight=weight
                    ))
        return self.near_misses

    def analyze_spectrum(self) -> Dict[str, object]:
        """Return statistics of the Δ-spectrum with λ=4 weighting."""
        if not self.near_misses:
            return {}
        primes = [nm.smallest_prime for nm in self.near_misses]
        weights = [nm.lambda4_weight for nm in self.near_misses]
        prime_counts = collections.Counter(primes)
        weighted_counts = collections.defaultdict(float)
        for nm in self.near_misses:
            weighted_counts[nm.smallest_prime] += nm.lambda4_weight
        forbidden = self._detect_forbidden_bands(prime_counts)
        resonance = self._detect_resonance_peaks(weighted_counts, prime_counts)
        return {
            'total_samples': len(self.near_misses),
            'prime_distribution': dict(prime_counts),
            'weighted_prime_distribution': dict(weighted_counts),
            'lambda4_boost_factor': float(np.mean(weights)),
            'prime_entropy': self._calculate_entropy(list(prime_counts.values())),
            'forbidden_bands': forbidden,
            'resonance_peaks': resonance,
        }

    def _detect_forbidden_bands(self, prime_counts: Dict[int, int],
                                threshold_factor: float = 0.3) -> List[int]:
        if not prime_counts:
            return []
        mean_freq = np.mean(list(prime_counts.values()))
        threshold = threshold_factor * mean_freq
        max_prime = max(prime_counts)
        bands = []
        for p in range(2, min(max_prime + 1, 50)):
            if self._is_prime(p) and prime_counts.get(p, 0) < threshold:
                bands.append(p)
        return bands

    def _detect_resonance_peaks(self, weighted: Dict[int, float],
                                regular: Dict[int, int]) -> List[int]:
        peaks = []
        for p, w in weighted.items():
            if p in regular and w / regular[p] > 1.5:
                peaks.append(p)
        return sorted(peaks)
