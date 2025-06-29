"""Δ-Spectrum computation for Beal's conjecture.

This module encapsulates the production-ready analysis pipeline for searching
near-miss solutions to :math:`A^x + B^y = C^z`.  The routines are designed to
mirror scientific data collection: prime factors act as spectroscopic lines,
and the weighted distributions emulate resonance detection in a physical
experiment.
"""

from __future__ import annotations

import json
import math
import collections
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import concurrent.futures
import warnings

import numpy as np
from tqdm.auto import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PARQUET_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    JOBLIB_AVAILABLE = False


DEFAULTS = {
    "x": 3,
    "y": 5,
    "z": 7,
    "limit": 50,
    "coprime_only": True,
    "max_delta": None,
    "progress": True,
    "parallel": False,
    "n_jobs": 4,
}


@dataclass
class NearMiss:
    """Record of a near-miss triple.

    Parameters correspond to integers ``A`` and ``B`` with an approximate ``C``.
    The ``delta`` represents the deviation from an exact power identity, akin to
    measuring the mismatch in a resonance experiment.
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
    """Cache smallest prime factors to mimic spectral lookup tables."""

    def __init__(self, cache_limit: int = 1_000_000) -> None:
        self.cache: Dict[int, int] = {}
        self.cache_limit = cache_limit

    def smallest_prime_factor(self, n: int) -> int:
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


class ProductionDeltaAnalyzer:
    """Driver class for Δ-spectrum exploration.

    The analyzer orchestrates the search for near-miss triples and evaluates the
    distribution of their prime deviations.  It mirrors how experimental physicists
    scan parameter spaces and catalogue resonance peaks.
    """

    def __init__(self, invariants_path: Optional[Path] = None, prime_cache_limit: int = 1_000_000) -> None:
        self.invariants_path = invariants_path
        self.lambda4_items: List[Dict] = []
        self.near_misses: List[NearMiss] = []
        self.prime_cache = OptimizedPrimeCache(prime_cache_limit)
        if invariants_path and invariants_path.exists():
            self._load_invariants()

    def _load_invariants(self) -> None:
        try:
            with self.invariants_path.open() as f:
                data = json.load(f)
            invariants: List[Dict] = []
            if isinstance(data, dict) and "invariants" in data:
                for category in data["invariants"].values():
                    if isinstance(category, dict):
                        for subcategory in category.values():
                            if isinstance(subcategory, list):
                                invariants.extend(subcategory)
                    elif isinstance(category, list):
                        invariants.extend(category)
            elif isinstance(data, list):
                invariants = data

            def lambda4_compatible(item: Dict) -> bool:
                txt = json.dumps(item).lower()
                patterns = [
                    "λ=4",
                    "λ = 4",
                    "lambda=4",
                    "lambda = 4",
                    "eigenmode 4",
                    "λ-4",
                    "lambda-4",
                    "lambda_4",
                    '"lambda": 4',
                    '"eigenvalue": 4',
                ]
                return any(p in txt for p in patterns)

            self.lambda4_items = [inv for inv in invariants if lambda4_compatible(inv)]
        except Exception as exc:  # pragma: no cover - reading failure
            warnings.warn(f"Failed to load invariants: {exc}")
            self.lambda4_items = []

    # ---- search logic --------------------------------------------------
    def _compute_lambda4_weight(self, A: int, B: int, C: int) -> float:
        if not self.lambda4_items:
            return 1.0
        base_weight = 1.0
        for item in self.lambda4_items[:5]:
            txt = json.dumps(item)
            if str(A) in txt or str(B) in txt:
                base_weight *= 2.0
                break
        return base_weight

    def _search_chunk(self, A_range: range, **kwargs) -> List[NearMiss]:
        x = kwargs.get("x", 3)
        y = kwargs.get("y", 5)
        z = kwargs.get("z", 7)
        limit = kwargs.get("limit", 50)
        coprime_only = kwargs.get("coprime_only", True)
        max_delta = kwargs.get("max_delta")
        progress = kwargs.get("progress", False)

        chunk_misses: List[NearMiss] = []
        A_iter = tqdm(A_range, desc=f"A={A_range.start}-{A_range.stop-1}", disable=not progress)
        for A in A_iter:
            for B in range(A, limit + 1):
                if coprime_only and math.gcd(A, B) != 1:
                    continue
                S = A ** x + B ** y
                C_float = S ** (1 / z)
                for C in [int(C_float), int(C_float) + 1]:
                    if C <= 0:
                        continue
                    delta = abs(S - C ** z)
                    if delta == 0:
                        continue
                    if max_delta and delta > max_delta:
                        continue
                    sp = self.prime_cache.smallest_prime_factor(delta)
                    gcd_abc = math.gcd(math.gcd(A, B), C)
                    w = self._compute_lambda4_weight(A, B, C)
                    chunk_misses.append(
                        NearMiss(A, B, C, x, y, z, delta, sp, gcd_abc, w)
                    )
        return chunk_misses

    def search_near_misses(self, **kwargs) -> List[NearMiss]:
        params = {**DEFAULTS, **kwargs}
        x, y, z = params["x"], params["y"], params["z"]
        limit = params["limit"]
        parallel = params["parallel"] and JOBLIB_AVAILABLE
        n_jobs = params["n_jobs"]
        progress = params["progress"]

        if parallel:
            chunk_size = max(1, limit // n_jobs)
            A_ranges = [range(start, min(start + chunk_size, limit + 1)) for start in range(1, limit + 1, chunk_size)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._search_chunk)(r, **params) for r in A_ranges
            )
            self.near_misses = [m for chunk in results for m in chunk]
        else:
            self.near_misses = self._search_chunk(range(1, limit + 1), **params)
        return self.near_misses

    # ---- analysis ------------------------------------------------------
    def _calculate_entropy(self, counts: List[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * math.log2(p) for p in probs)

    def _detect_forbidden_bands(self, prime_counts: Dict[int, int], threshold_factor: float = 0.3) -> List[int]:
        if not prime_counts or len(prime_counts) < 3:
            return []
        max_prime = max(prime_counts.keys())
        mean_freq = np.mean(list(prime_counts.values()))
        threshold = threshold_factor * mean_freq
        forbidden: List[int] = []
        for p in range(2, min(max_prime + 1, 50)):
            if self._is_prime(p) and prime_counts.get(p, 0) < threshold:
                forbidden.append(p)
        return forbidden

    def _detect_resonance_peaks(self, weighted_counts: Dict[int, float], regular_counts: Dict[int, int]) -> List[int]:
        peaks: List[int] = []
        for prime in weighted_counts:
            if prime in regular_counts:
                ratio = weighted_counts[prime] / regular_counts[prime]
                if ratio > 1.5:
                    peaks.append(prime)
        return sorted(peaks)

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

    def analyze_spectrum(self) -> Dict:
        if not self.near_misses:
            return {}
        primes = [nm.smallest_prime for nm in self.near_misses]
        deltas = [nm.delta for nm in self.near_misses]
        weights = [nm.lambda4_weight for nm in self.near_misses]
        prime_counts = collections.Counter()
        weighted_prime_counts: Dict[int, float] = collections.defaultdict(float)
        for nm in self.near_misses:
            prime_counts[nm.smallest_prime] += 1
            weighted_prime_counts[nm.smallest_prime] += nm.lambda4_weight
        analysis = {
            "total_samples": len(self.near_misses),
            "exponents": (self.near_misses[0].x, self.near_misses[0].y, self.near_misses[0].z),
            "prime_distribution": dict(prime_counts),
            "weighted_prime_distribution": dict(weighted_prime_counts),
            "coprime_fraction": sum(1 for nm in self.near_misses if nm.is_coprime) / len(self.near_misses),
            "lambda4_boost_factor": float(np.mean(weights)),
            "mean_prime": float(np.mean(primes)),
            "median_prime": float(np.median(primes)),
            "std_prime": float(np.std(primes)),
            "prime_entropy": self._calculate_entropy(list(prime_counts.values())),
            "mean_log_delta": float(np.mean([nm.log_delta for nm in self.near_misses])),
            "delta_range": (min(deltas), max(deltas)),
            "forbidden_bands": self._detect_forbidden_bands(prime_counts),
            "resonance_peaks": self._detect_resonance_peaks(weighted_prime_counts, prime_counts),
        }
        return analysis


__all__ = [
    "NearMiss",
    "OptimizedPrimeCache",
    "ProductionDeltaAnalyzer",
]

