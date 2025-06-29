"""
Unified test suite for Δ-Spectrum analysis module.

This test suite combines comprehensive coverage from both branches,
ensuring the unified module works correctly across all use cases.
"""

import sys
import tempfile
import json
from pathlib import Path
import pytest

# Setup path to find the module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.delta_spectrum import (
    OptimizedPrimeCache, 
    ProductionDeltaAnalyzer, 
    NearMiss,
    _lambda4_weight,
    _is_prime,
    quick_search,
    production_search,
    DEFAULTS
)


class TestOptimizedPrimeCache:
    """Test the prime factor caching system."""
    
    def test_smallest_prime_factor_basic(self):
        """Test basic prime factor computation."""
        cache = OptimizedPrimeCache()
        
        # Test known values
        assert cache.smallest_prime_factor(1) == 1
        assert cache.smallest_prime_factor(2) == 2
        assert cache.smallest_prime_factor(4) == 2
        assert cache.smallest_prime_factor(15) == 3
        assert cache.smallest_prime_factor(17) == 17  # prime
        assert cache.smallest_prime_factor(21) == 3
        assert cache.smallest_prime_factor(100) == 2
    
    def test_cache_functionality(self):
        """Test that caching works correctly."""
        cache = OptimizedPrimeCache()
        
        # First call - cache miss
        result1 = cache.smallest_prime_factor(15)
        miss_count_1 = cache.miss_count
        
        # Second call - should hit cache
        result2 = cache.smallest_prime_factor(15)
        hit_count_2 = cache.hit_count
        
        assert result1 == result2 == 3
        assert hit_count_2 > 0  # Should have registered a hit
        assert cache.cache_efficiency > 0
    
    def test_edge_cases(self):
        """Test edge cases for prime factor computation."""
        cache = OptimizedPrimeCache()
        
        assert cache.smallest_prime_factor(0) == 0
        assert cache.smallest_prime_factor(-15) == 3  # Absolute value
        assert cache.smallest_prime_factor(1) == 1
    
    def test_large_primes(self):
        """Test with larger prime numbers."""
        cache = OptimizedPrimeCache()
        
        # Test a larger prime
        assert cache.smallest_prime_factor(97) == 97
        # Test a larger composite
        assert cache.smallest_prime_factor(91) == 7  # 91 = 7 * 13


class TestHelperFunctions:
    """Test standalone helper functions."""
    
    def test_is_prime_basic(self):
        """Test primality testing function."""
        assert _is_prime(2)
        assert _is_prime(3)
        assert _is_prime(5)
        assert _is_prime(7)
        assert _is_prime(11)
        assert _is_prime(17)
        
        assert not _is_prime(1)
        assert not _is_prime(4)
        assert not _is_prime(6)
        assert not _is_prime(8)
        assert not _is_prime(9)
        assert not _is_prime(15)
    
    def test_is_prime_edge_cases(self):
        """Test edge cases for primality."""
        assert not _is_prime(0)
        assert not _is_prime(-5)
        assert _is_prime(2)  # Smallest prime
    
    def test_lambda4_weight_basic(self):
        """Test λ=4 weighting function."""
        # Empty invariants
        assert _lambda4_weight(3, 4, []) == 1.0
        
        # Invariants without matches
        invariants = [{"name": "test", "value": 42}]
        assert _lambda4_weight(3, 4, invariants) == 1.0
        
        # Invariants with matches
        invariants_with_match = [{"name": "test_3", "value": 42}]
        assert _lambda4_weight(3, 4, invariants_with_match) == 2.0


class TestNearMiss:
    """Test the NearMiss dataclass."""
    
    def test_near_miss_creation(self):
        """Test basic NearMiss creation and properties."""
        nm = NearMiss(
            A=3, B=4, C_approx=5,
            x=3, y=3, z=3,
            delta=100, smallest_prime=2, gcd_abc=1
        )
        
        assert nm.A == 3
        assert nm.B == 4
        assert nm.is_coprime == True
        assert nm.log_delta > 0
        assert nm.resonance_classification == "fundamental_mode"  # prime 2
    
    def test_resonance_classification(self):
        """Test resonance mode classification."""
        # Fundamental mode (prime 2)
        nm1 = NearMiss(3, 4, 5, 3, 3, 3, 100, 2, 1)
        assert nm1.resonance_classification == "fundamental_mode"
        
        # Harmonic mode (prime 3-5)
        nm2 = NearMiss(3, 4, 5, 3, 3, 3, 100, 3, 1)
        assert nm2.resonance_classification == "harmonic_mode"
        
        # Overtone mode (prime 6-20)
        nm3 = NearMiss(3, 4, 5, 3, 3, 3, 100, 13, 1)
        assert nm3.resonance_classification == "overtone_mode"
        
        # Noise mode (prime >20)
        nm4 = NearMiss(3, 4, 5, 3, 3, 3, 100, 23, 1)
        assert nm4.resonance_classification == "noise_mode"
    
    def test_coprime_detection(self):
        """Test coprime detection."""
        # Coprime case
        coprime_nm = NearMiss(3, 5, 7, 3, 3, 3, 100, 2, 1)
        assert coprime_nm.is_coprime == True
        
        # Non-coprime case  
        non_coprime_nm = NearMiss(6, 9, 12, 3, 3, 3, 100, 2, 3)
        assert non_coprime_nm.is_coprime == False


class TestProductionDeltaAnalyzer:
    """Test the main analyzer class."""
    
    def test_initialization_no_invariants(self):
        """Test analyzer initialization without invariants."""
        analyzer = ProductionDeltaAnalyzer()
        
        assert analyzer.invariants_path is None
        assert analyzer.lambda4_items == []
        assert analyzer.near_misses == []
        assert analyzer.prime_cache is not None
    
    def test_initialization_with_dummy_invariants(self):
        """Test analyzer with dummy invariants file."""
        dummy_invariants = [
            {"name": "test_λ=4", "description": "lambda=4 eigenmode"},
            {"name": "regular", "value": 42}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dummy_invariants, f)
            json_path = Path(f.name)
        
        try:
            analyzer = ProductionDeltaAnalyzer(json_path)
            assert len(analyzer.lambda4_items) == 1
            assert "λ=4" in analyzer.lambda4_items[0]["name"]
        finally:
            json_path.unlink()
    
    def test_search_near_misses_small_sequential(self):
        """Test small search in sequential mode."""
        analyzer = ProductionDeltaAnalyzer()
        
        # Small search to keep test fast
        results = analyzer.search_near_misses(
            x=3, y=3, z=3, 
            limit=8, 
            progress=False, 
            parallel=False
        )
        
        assert len(results) > 0
        assert all(isinstance(r, NearMiss) for r in results)
        assert analyzer.near_misses == results
        
        # Check that all results have correct exponents
        assert all(r.x == 3 and r.y == 3 and r.z == 3 for r in results)
    
    def test_search_near_misses_with_constraints(self):
        """Test search with various constraints."""
        analyzer = ProductionDeltaAnalyzer()
        
        # Test with max_delta constraint
        results_constrained = analyzer.search_near_misses(
            x=3, y=5, z=7,
            limit=10,
            max_delta=1000,
            progress=False
        )
        
        # All deltas should be within constraint
        assert all(r.delta <= 1000 for r in results_constrained)
    
    def test_analyze_spectrum_basic(self):
        """Test spectrum analysis functionality."""
        analyzer = ProductionDeltaAnalyzer()
        
        # Generate some data
        analyzer.search_near_misses(
            x=3, y=3, z=4,
            limit=12, 
            progress=False
        )
        
        analysis = analyzer.analyze_spectrum()
        
        # Check required fields
        required_fields = [
            'total_samples', 'exponents', 'prime_distribution',
            'coprime_fraction', 'mean_prime', 'prime_entropy'
        ]
        for field in required_fields:
            assert field in analysis
        
        # Check data consistency
        assert analysis['total_samples'] == len(analyzer.near_misses)
        assert analysis['total_samples'] > 0
        assert analysis['exponents'] == (3, 3, 4)
        assert 0.0 <= analysis['coprime_fraction'] <= 1.0
        assert analysis['mean_prime'] >= 2.0
    
    def test_analyze_spectrum_empty(self):
        """Test spectrum analysis with no data."""
        analyzer = ProductionDeltaAnalyzer()
        analysis = analyzer.analyze_spectrum()
        assert analysis == {}
    
    def test_forbidden_bands_detection(self):
        """Test forbidden band detection."""
        analyzer = ProductionDeltaAnalyzer()
        
        # Create analyzer with some data
        analyzer.search_near_misses(x=3, y=5, z=7, limit=15, progress=False)
        
        if analyzer.near_misses:  # Only test if we have data
            analysis = analyzer.analyze_spectrum()
            
            # Forbidden bands should be a list
            assert isinstance(analysis.get('forbidden_bands', []), list)
            
            # All forbidden bands should be prime numbers
            forbidden = analysis.get('forbidden_bands', [])
            assert all(_is_prime(p) for p in forbidden)
    
    @pytest.mark.slow
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential give same results."""
        # Skip if joblib not available
        pytest.importorskip("joblib")
        
        analyzer1 = ProductionDeltaAnalyzer()
        analyzer2 = ProductionDeltaAnalyzer()
        
        params = dict(x=3, y=5, z=7, limit=20, progress=False)
        
        # Sequential
        results1 = analyzer1.search_near_misses(parallel=False, **params)
        
        # Parallel
        results2 = analyzer2.search_near_misses(parallel=True, n_jobs=2, **params)
        
        # Should find same number of results
        assert len(results1) == len(results2)
        
        # Results should be equivalent (order may differ)
        set1 = {(r.A, r.B, r.C_approx, r.delta) for r in results1}
        set2 = {(r.A, r.B, r.C_approx, r.delta) for r in results2}
        assert set1 == set2


class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    def test_quick_search(self):
        """Test quick_search convenience function."""
        results = quick_search(x=3, y=3, z=4, limit=8)
        
        assert isinstance(results, dict)
        assert 'total_samples' in results
        assert results['total_samples'] > 0
    
    def test_production_search(self):
        """Test production_search convenience function."""
        analyzer = production_search(
            invariants_path=None,
            limit=10,
            parallel=False  # Keep test simple
        )
        
        assert isinstance(analyzer, ProductionDeltaAnalyzer)
        assert len(analyzer.near_misses) > 0


class TestExportFunctionality:
    """Test data export capabilities."""
    
    def test_export_basic(self):
        """Test basic export functionality."""
        analyzer = ProductionDeltaAnalyzer()
        analyzer.search_near_misses(x=3, y=3, z=4, limit=8, progress=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test export without parquet
            analyzer.export_results(output_dir, include_parquet=False)
            
            # Check required files exist
            assert (output_dir / "delta_spectrum_data.csv").exists()
            assert (output_dir / "spectrum_analysis.json").exists()
            assert (output_dir / "analysis_report.md").exists()
            
            # Check files have content
            csv_size = (output_dir / "delta_spectrum_data.csv").stat().st_size
            json_size = (output_dir / "spectrum_analysis.json").stat().st_size
            md_size = (output_dir / "analysis_report.md").stat().st_size
            
            assert csv_size > 0
            assert json_size > 0  
            assert md_size > 0


class TestDefaults:
    """Test configuration defaults."""
    
    def test_defaults_structure(self):
        """Test that DEFAULTS contains expected parameters."""
        required_keys = ['x', 'y', 'z', 'limit', 'coprime_only']
        for key in required_keys:
            assert key in DEFAULTS
        
        # Check reasonable values
        assert DEFAULTS['x'] >= 3
        assert DEFAULTS['y'] >= 3
        assert DEFAULTS['z'] >= 3
        assert DEFAULTS['limit'] > 0
        assert isinstance(DEFAULTS['coprime_only'], bool)


class TestRegressionProtection:
    """Regression tests to catch changes in core behavior."""
    
    def test_no_exact_solutions(self):
        """Regression: should never find exact solutions for x,y,z > 2."""
        analyzer = ProductionDeltaAnalyzer()
        analyzer.search_near_misses(x=3, y=5, z=7, limit=15, progress=False)
        
        # Beal's conjecture: no exact solutions should exist
        exact_solutions = [nm for nm in analyzer.near_misses if nm.delta == 0]
        assert len(exact_solutions) == 0, "Found unexpected exact solution!"
    
    def test_coprime_constraint_respected(self):
        """Regression: coprime_only=True should only return coprime bases."""
        analyzer = ProductionDeltaAnalyzer()
        analyzer.search_near_misses(
            x=3, y=3, z=4, 
            limit=10, 
            coprime_only=True, 
            progress=False
        )
        
        # Check that A,B are coprime for all results
        for nm in analyzer.near_misses:
            assert math.gcd(nm.A, nm.B) == 1, f"Non-coprime pair found: ({nm.A}, {nm.B})"
    
    def test_delta_positive(self):
        """Regression: all deltas should be positive."""
        analyzer = ProductionDeltaAnalyzer()
        analyzer.search_near_misses(x=3, y=5, z=7, limit=12, progress=False)
        
        for nm in analyzer.near_misses:
            assert nm.delta > 0, f"Non-positive delta found: {nm.delta}"


# Performance markers for optional slow tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])