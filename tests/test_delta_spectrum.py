import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.delta_spectrum import OptimizedPrimeCache, DeltaSpectrumAnalyzer


def test_smallest_prime_factor_cache():
    cache = OptimizedPrimeCache()
    assert cache.smallest_prime_factor(7) == 7
    assert cache.smallest_prime_factor(8) == 2
    assert cache.smallest_prime_factor(0) == 0
    # value should now be cached
    assert 8 in cache.cache


def test_entropy_and_prime_detection():
    analyzer = DeltaSpectrumAnalyzer()
    entropy = analyzer._calculate_entropy([3, 1])
    assert entropy > 0
    assert analyzer._is_prime(7)
    assert not analyzer._is_prime(9)


def test_search_near_misses_basic():
    analyzer = DeltaSpectrumAnalyzer()
    misses = analyzer.search_near_misses(limit=5, coprime_only=True, max_delta=1000)
    assert misses
    # verify that all deltas are positive
    assert all(m.delta > 0 for m in misses)
