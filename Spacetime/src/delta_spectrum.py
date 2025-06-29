"""Δ-Spectrum utilities for near-miss analysis.

This module implements a streamlined version of the Beal-conjecture
search.  The procedure parallels scanning for resonant frequencies in a
physical system and assigns λ=4 weights guided by the SMUG framework.
"""

from __future__ import annotations

import json
import math
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

DEFAULTS = {
    "x": 3,
    "y": 5,
    "z": 7,
    "limit": 50,
    "coprime_only": True,
    "max_delta": None,
}

@dataclass
class NearMiss:
    """Store a single ``A^x + B^y \approx C^z`` near miss."""

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
        """Return True if ``A``, ``B`` and ``C`` share no factor."""
        return self.gcd_abc == 1

    @property
    def log_delta(self) -> float:
        """Base-10 log of ``delta``."""
        return math.log10(max(1, self.delta))

class OptimizedPrimeCache:
    """Smallest-prime cache mimicking a sieve."""

    def __init__(self, cache_limit: int = 1_000_000) -> None:
        self.cache: Dict[int, int] = {}
        self.cache_limit = cache_limit

    def smallest_prime_factor(self, n: int) -> int:
        """Return the smallest prime factor of ``n``."""
        if n in self.cache:
            return self.cache[n]
        if n == 0:
            return 0
        n = abs(n)
        if n == 1:
            result = 1
        elif n % 2 == 0:
            result = 2
        else:
            result = n
            i = 3
            while i * i <= n:
                if n % i == 0:
                    result = i
                    break
                i += 2
        if n <= self.cache_limit:
            self.cache[n] = result
        return result

def _lambda4_weight(A: int, B: int, invariants: List[dict]) -> float:
    """Heuristic λ=4 compatibility weight."""
    if not invariants:
        return 1.0
    body = json.dumps(invariants[:5]).lower()
    return 2.0 if str(A) in body or str(B) in body else 1.0

def _is_prime(n: int) -> bool:
    """Return True if ``n`` is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

class ProductionDeltaAnalyzer:
    """Search and summarise Δ-spectrum data."""

    def __init__(self, invariants_path: Optional[Path] = None) -> None:
        self.invariants_path = invariants_path
        self.lambda4_items: List[dict] = []
        self.near_misses: List[NearMiss] = []
        self.prime_cache = OptimizedPrimeCache()
        if invariants_path and invariants_path.exists():
            self._load_invariants()

    def _load_invariants(self) -> None:
        """Load λ=4-related invariants."""
        with self.invariants_path.open() as f:
            data = json.load(f)
        if isinstance(data, dict) and "invariants" in data:
            items = []
            for cat in data["invariants"].values():
                if isinstance(cat, list):
                    items.extend(cat)
        else:
            items = data if isinstance(data, list) else []
        self.lambda4_items = items

    def _search_single(self, A: int, params: Dict) -> List[NearMiss]:
        """Return near misses for one ``A`` value."""
        x, y, z = params["x"], params["y"], params["z"]
        limit = params["limit"]
        coprime_only = params["coprime_only"]
        max_delta = params["max_delta"]
        misses: List[NearMiss] = []
        for B in range(A, limit + 1):
            if coprime_only and math.gcd(A, B) != 1:
                continue
            S = A ** x + B ** y
            C_float = S ** (1 / z)
            for C in [int(C_float), int(C_float) + 1]:
                delta = abs(S - C ** z)
                if delta == 0:
                    continue
                if max_delta and delta > max_delta:
                    continue
                sp = self.prime_cache.smallest_prime_factor(delta)
                gcd_abc = math.gcd(math.gcd(A, B), C)
                weight = _lambda4_weight(A, B, self.lambda4_items)
                misses.append(
                    NearMiss(
                        A=A,
                        B=B,
                        C_approx=C,
                        x=x,
                        y=y,
                        z=z,
                        delta=delta,
                        smallest_prime=sp,
                        gcd_abc=gcd_abc,
                        lambda4_weight=weight,
                    )
                )
        return misses

    def search_near_misses(self, **kwargs) -> List[NearMiss]:
        """Compute near misses using the given parameters."""
        params = {**DEFAULTS, **kwargs}
        limit = params["limit"]
        self.near_misses = []
        for A in range(1, limit + 1):
            self.near_misses.extend(self._search_single(A, params))
        return self.near_misses

    def analyze_spectrum(self) -> Dict:
        """Return basic statistics of the Δ-spectrum."""
        if not self.near_misses:
            return {}
        primes = [nm.smallest_prime for nm in self.near_misses]
        counts = collections.Counter(primes)
        weights = [nm.lambda4_weight for nm in self.near_misses]
        return {
            "total_samples": len(self.near_misses),
            "prime_distribution": dict(counts),
            "lambda4_boost_factor": float(np.mean(weights)),
            "prime_entropy": self._entropy(list(counts.values())),
        }

    def _entropy(self, counts: List[int]) -> float:
        """Shannon entropy of ``counts``."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -float(sum(p * math.log2(p) for p in probs))

