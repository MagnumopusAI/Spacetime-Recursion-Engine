import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.delta_spectrum import OptimizedPrimeCache, ProductionDeltaAnalyzer, _is_prime


def test_smallest_prime_factor():
    cache = OptimizedPrimeCache()
    assert cache.smallest_prime_factor(15) == 3
    assert cache.smallest_prime_factor(2) == 2
    assert cache.smallest_prime_factor(1) == 1


def test_is_prime_basic():
    assert _is_prime(7)
    assert not _is_prime(9)


def test_search_near_misses_basic():
    analyzer = ProductionDeltaAnalyzer()
    results = analyzer.search_near_misses(limit=5, max_delta=100)
    assert results
    analysis = analyzer.analyze_spectrum()
    assert analysis['total_samples'] == len(results)
