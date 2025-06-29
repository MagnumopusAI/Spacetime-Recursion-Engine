import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.delta_spectrum import OptimizedPrimeCache, ProductionDeltaAnalyzer


def test_prime_cache_basic():
    cache = OptimizedPrimeCache()
    assert cache.smallest_prime_factor(15) == 3
    # second call should hit the cache
    assert cache.smallest_prime_factor(15) == 3


def test_near_miss_search_small():
    analyzer = ProductionDeltaAnalyzer()
    misses = analyzer.search_near_misses(limit=10, progress=False, parallel=False)
    assert len(misses) > 0
    analysis = analyzer.analyze_spectrum()
    assert analysis["total_samples"] == len(misses)
    assert analysis["mean_prime"] >= 2

