import pytest
import numpy as np
import torch
from torch import float32, float64, complex64, complex128, int32, int64
from cryolike.util.precision import (
    PrecisionLevel, PrecisionConfig, PrecisionContext,
    set_precision, get_precision, get_float_dtype, 
    get_complex_dtype, get_int_dtype, get_epsilon
)

class TestPrecisionLevel:
    """Test PrecisionLevel enum."""
    
    def test_enum_values(self):
        """Test that enum has correct values."""
        assert PrecisionLevel.SINGLE.value == 'single'
        assert PrecisionLevel.DOUBLE.value == 'double'
    
    def test_from_str_with_string_single(self):
        """Test from_str with 'single' string."""
        result = PrecisionLevel.from_str('single')
        assert result == PrecisionLevel.SINGLE
    
    def test_from_str_with_string_double(self):
        """Test from_str with 'double' string."""
        result = PrecisionLevel.from_str('double')
        assert result == PrecisionLevel.DOUBLE
    
    def test_from_str_with_uppercase(self):
        """Test from_str is case insensitive."""
        assert PrecisionLevel.from_str('SINGLE') == PrecisionLevel.SINGLE
        assert PrecisionLevel.from_str('DOUBLE') == PrecisionLevel.DOUBLE
        assert PrecisionLevel.from_str('Single') == PrecisionLevel.SINGLE
    
    def test_from_str_with_whitespace(self):
        """Test from_str handles whitespace."""
        assert PrecisionLevel.from_str('  single  ') == PrecisionLevel.SINGLE
        assert PrecisionLevel.from_str('double ') == PrecisionLevel.DOUBLE
    
    def test_from_str_with_precision_level(self):
        """Test from_str with PrecisionLevel instance returns itself."""
        level = PrecisionLevel.SINGLE
        result = PrecisionLevel.from_str(level)
        assert result is level
    
    def test_from_str_with_invalid_string(self):
        """Test from_str raises ValueError for invalid string."""
        with pytest.raises(ValueError, match="Unsupported precision value"):
            PrecisionLevel.from_str('invalid')
    
    def test_from_str_with_invalid_type(self):
        """Test from_str raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Input must be a string"):
            PrecisionLevel.from_str(123)
    
    def test_from_str_with_empty_string(self):
        """Test from_str raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Unsupported precision value"):
            PrecisionLevel.from_str('')


class TestPrecisionConfig:
    """Test PrecisionConfig dataclass."""
    
    def test_from_level_single(self):
        """Test from_level creates correct config for SINGLE."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        assert config.float_dtype == float32
        assert config.complex_dtype == complex64
        assert config.int_dtype == int32
        assert config.epsilon == 1.0e-6
    
    def test_from_level_double(self):
        """Test from_level creates correct config for DOUBLE."""
        config = PrecisionConfig.from_level(PrecisionLevel.DOUBLE)
        assert config.float_dtype == float64
        assert config.complex_dtype == complex128
        assert config.int_dtype == int64
        assert config.epsilon == 1.0e-14
    
    def test_config_is_frozen(self):
        """Test that PrecisionConfig is immutable."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            config.epsilon = 1.0e-3
    
    def test_get_epsilon_returns_requested_when_larger(self):
        """Test get_epsilon returns requested value when it's larger than machine precision."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        result = config.get_epsilon(1.0e-3)
        assert result == 1.0e-3
    
    def test_get_epsilon_returns_machine_precision_when_requested_smaller(self):
        """Test get_epsilon returns machine precision when requested is smaller."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        result = config.get_epsilon(1.0e-10)
        assert result == 1.0e-6
    
    def test_get_epsilon_double_precision(self):
        """Test get_epsilon with double precision."""
        config = PrecisionConfig.from_level(PrecisionLevel.DOUBLE)
        # Requested larger than machine precision
        assert config.get_epsilon(1.0e-6) == 1.0e-6
        # Requested smaller than machine precision
        assert config.get_epsilon(1.0e-14) == 1.0e-14
    
    def test_get_epsilon_exact_match(self):
        """Test get_epsilon when requested equals machine precision."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        result = config.get_epsilon(1.0e-6)
        assert result == 1.0e-6


class TestPrecisionContext:
    """Test PrecisionContext global manager."""
    
    def setup_method(self):
        """Reset to default before each test."""
        PrecisionContext.set_default(PrecisionLevel.SINGLE)
    
    def test_default_precision_is_single(self):
        """Test that default precision is SINGLE."""
        assert PrecisionContext.get_default() == PrecisionLevel.SINGLE
    
    def test_set_default_with_enum(self):
        """Test set_default with PrecisionLevel enum."""
        PrecisionContext.set_default(PrecisionLevel.DOUBLE)
        assert PrecisionContext.get_default() == PrecisionLevel.DOUBLE
    
    def test_set_default_with_string(self):
        """Test set_default with string."""
        PrecisionContext.set_default('double')
        assert PrecisionContext.get_default() == PrecisionLevel.DOUBLE
    
    def test_get_config_with_none_uses_default(self):
        """Test get_config with None uses default precision."""
        PrecisionContext.set_default(PrecisionLevel.DOUBLE)
        config = PrecisionContext.get_config(None)
        assert config.float_dtype == float64
    
    def test_get_config_with_precision_level(self):
        """Test get_config with explicit PrecisionLevel."""
        config = PrecisionContext.get_config(PrecisionLevel.DOUBLE)
        assert config.float_dtype == float64
    
    def test_get_config_with_string(self):
        """Test get_config with string."""
        config = PrecisionContext.get_config('double')
        assert config.float_dtype == float64
    
    def test_get_config_returns_cached_instance(self):
        """Test that get_config returns cached instances."""
        config1 = PrecisionContext.get_config(PrecisionLevel.SINGLE)
        config2 = PrecisionContext.get_config(PrecisionLevel.SINGLE)
        assert config1 is config2
    
    def test_get_dtypes_with_none_uses_default(self):
        """Test get_dtypes with None uses default precision."""
        PrecisionContext.set_default(PrecisionLevel.DOUBLE)
        float_dtype = PrecisionContext.get_float_dtype(None)
        complex_dtype = PrecisionContext.get_complex_dtype(None)
        int_dtype = PrecisionContext.get_int_dtype(None)
        assert float_dtype == float64
        assert complex_dtype == complex128
        assert int_dtype == int64
    
    def test_get_dtypes_with_precision_level(self):
        """Test get_dtypes with explicit PrecisionLevel."""
        float_dtype= PrecisionContext.get_float_dtype(PrecisionLevel.SINGLE)
        complex_dtype = PrecisionContext.get_complex_dtype(PrecisionLevel.SINGLE)
        int_dtype = PrecisionContext.get_int_dtype(PrecisionLevel.SINGLE)
        assert float_dtype == float32
        assert complex_dtype == complex64
        assert int_dtype == int32
    
    def test_get_dtypes_with_string(self):
        """Test get_dtypes with string."""
        float_dtype = PrecisionContext.get_float_dtype('double')
        complex_dtype = PrecisionContext.get_complex_dtype('double')
        int_dtype = PrecisionContext.get_int_dtype('double')
        assert float_dtype == float64
        assert complex_dtype == complex128
        assert int_dtype == int64
    
    def test_get_epsilon_with_none_uses_default(self):
        """Test get_epsilon with None uses default precision."""
        PrecisionContext.set_default(PrecisionLevel.DOUBLE)
        eps = PrecisionContext.get_epsilon(1.0e-14, None)
        assert eps == 1.0e-14
    
    def test_get_epsilon_with_precision_level(self):
        """Test get_epsilon with explicit PrecisionLevel."""
        eps = PrecisionContext.get_epsilon(1.0e-10, PrecisionLevel.SINGLE)
        assert eps == 1.0e-6
    
    def test_get_epsilon_with_string(self):
        """Test get_epsilon with string."""
        eps = PrecisionContext.get_epsilon(1.0e-10, 'single')
        assert eps == 1.0e-6
    
    def test_get_epsilon_default_requested_zero(self):
        """Test get_epsilon with default requested value of 0."""
        eps = PrecisionContext.get_epsilon()
        assert eps == 1.0e-6  # Default is SINGLE
    
    def test_cache_contains_all_levels(self):
        """Test that cache is pre-populated with all precision levels."""
        assert PrecisionLevel.SINGLE in PrecisionContext._config_cache
        assert PrecisionLevel.DOUBLE in PrecisionContext._config_cache


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Reset to default before each test."""
        set_precision(PrecisionLevel.SINGLE)
    
    def test_set_precision_with_string(self):
        """Test set_precision convenience function with string."""
        set_precision('double')
        assert get_precision() == PrecisionLevel.DOUBLE
    
    def test_set_precision_with_enum(self):
        """Test set_precision convenience function with enum."""
        set_precision(PrecisionLevel.DOUBLE)
        assert get_precision() == PrecisionLevel.DOUBLE
    
    def test_get_precision(self):
        """Test get_precision convenience function."""
        set_precision('double')
        assert get_precision() == PrecisionLevel.DOUBLE
    
    def test_get_dtypes_with_default(self):
        """Test get_dtypes convenience function uses default."""
        set_precision('double')
        # float_dtype, complex_dtype, int_dtype = get_dtypes()
        float_dtype = get_float_dtype()
        complex_dtype = get_complex_dtype()
        int_dtype = get_int_dtype()
        assert float_dtype == float64
        assert complex_dtype == complex128
        assert int_dtype == int64
    
    def test_get_dtypes_with_override(self):
        """Test get_dtypes convenience function with override."""
        set_precision('double')
        float_dtype = get_float_dtype('single')
        complex_dtype = get_complex_dtype('single')
        int_dtype = get_int_dtype('single')
        assert float_dtype == float32
        assert complex_dtype == complex64
        assert int_dtype == int32
    
    def test_get_epsilon_with_default(self):
        """Test get_epsilon convenience function uses default."""
        set_precision('single')
        eps = get_epsilon(1.0e-10)
        assert eps == 1.0e-6
    
    def test_get_epsilon_with_override(self):
        """Test get_epsilon convenience function with override."""
        set_precision('single')
        eps = get_epsilon(1.0e-14, 'double')
        assert eps == 1.0e-14
    
    def test_get_epsilon_default_value(self):
        """Test get_epsilon with default requested value."""
        set_precision('double')
        eps = get_epsilon()
        assert eps == 1.0e-14


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def setup_method(self):
        """Reset to default before each test."""
        set_precision(PrecisionLevel.SINGLE)
    
    def test_workflow_set_and_use_precision(self):
        """Test complete workflow of setting and using precision."""
        # Start with single precision
        assert get_precision() == PrecisionLevel.SINGLE
        
        # Switch to double precision
        set_precision('double')
        
        # Get dtypes for tensor creation
        float_dtype = get_float_dtype()
        tensor = torch.zeros(10, dtype=float_dtype)
        assert tensor.dtype == float64
        
        # Get appropriate epsilon
        eps = get_epsilon(1.0e-10)
        assert eps == 1.0e-10  # Can use requested value with double precision
    
    def test_temporary_precision_override(self):
        """Test using different precision for specific operation."""
        set_precision('single')
        
        # Default operation uses single precision
        float_dtype = get_float_dtype()
        assert float_dtype == float32
        
        # Specific operation uses double precision
        override_float_dtype = get_float_dtype('double')
        assert override_float_dtype == float64
        
        # Default is unchanged
        assert get_precision() == PrecisionLevel.SINGLE
    
    def test_epsilon_clamping_behavior(self):
        """Test epsilon clamping across different precisions."""
        # Single precision can't handle very small epsilon
        set_precision('single')
        eps_single = get_epsilon(1.0e-14)
        assert eps_single == 1.0e-6
        
        # Double precision can handle smaller epsilon
        set_precision('double')
        eps_double = get_epsilon(1.0e-14)
        assert eps_double == 1.0e-14


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_epsilon_request(self):
        """Test epsilon request larger than 1.0."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        eps = config.get_epsilon(10.0)
        assert eps == 10.0
    
    def test_zero_epsilon_request(self):
        """Test epsilon request of exactly 0."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        eps = config.get_epsilon(0.0)
        assert eps == 1.0e-6
    
    def test_negative_epsilon_request(self):
        """Test epsilon request with negative value returns machine precision."""
        config = PrecisionConfig.from_level(PrecisionLevel.SINGLE)
        eps = config.get_epsilon(-1.0e-6)
        assert eps == 1.0e-6
    
    def test_multiple_precision_switches(self):
        """Test switching precision multiple times."""
        set_precision('single')
        assert get_precision() == PrecisionLevel.SINGLE
        
        set_precision('double')
        assert get_precision() == PrecisionLevel.DOUBLE
        
        set_precision('single')
        assert get_precision() == PrecisionLevel.SINGLE
        
        set_precision('double')
        assert get_precision() == PrecisionLevel.DOUBLE
    
    def test_case_variations_in_from_str(self):
        """Test various case combinations in from_str."""
        assert PrecisionLevel.from_str('SINGLE') == PrecisionLevel.SINGLE
        assert PrecisionLevel.from_str('Single') == PrecisionLevel.SINGLE
        assert PrecisionLevel.from_str('sInGlE') == PrecisionLevel.SINGLE
        assert PrecisionLevel.from_str('DOUBLE') == PrecisionLevel.DOUBLE
        assert PrecisionLevel.from_str('Double') == PrecisionLevel.DOUBLE
        assert PrecisionLevel.from_str('DoUbLe') == PrecisionLevel.DOUBLE