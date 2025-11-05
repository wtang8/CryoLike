from dataclasses import dataclass
from enum import Enum
from typing import Optional
from torch import dtype, float32, float64, complex64, complex128, int32, int64


class PrecisionLevel(Enum):
    """Precision levels for computation."""
    SINGLE = 'single'
    DOUBLE = 'double'
    
    @classmethod
    def from_str(cls, label: str) -> 'PrecisionLevel':
        if isinstance(label, PrecisionLevel):
            return label
        if not isinstance(label, str):
            raise ValueError('Input must be a string.')
        
        label = label.lower().strip()
        if label == 'single':
            return PrecisionLevel.SINGLE
        if label == 'double':
            return PrecisionLevel.DOUBLE
        
        raise ValueError(f"Unsupported precision value: '{label}'. Use 'single' or 'double'.")


@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable precision configuration."""
    float_dtype: dtype
    complex_dtype: dtype
    int_dtype: dtype
    epsilon: float
    
    @classmethod
    def from_level(cls, level: PrecisionLevel) -> 'PrecisionConfig':
        if level == PrecisionLevel.SINGLE:
            return cls(float32, complex64, int32, 1.0e-6)
        else:  # DOUBLE
            return cls(float64, complex128, int64, 1.0e-14)
    
    def get_epsilon(self, requested: float) -> float:
        """Return requested epsilon or machine precision, whichever is larger."""
        return max(requested, self.epsilon)


class PrecisionContext:
    """Global precision configuration manager."""
    _default_precision: PrecisionLevel = PrecisionLevel.SINGLE
    _config_cache: dict[PrecisionLevel, PrecisionConfig] = {
        level: PrecisionConfig.from_level(level) for level in PrecisionLevel
    }
    
    @classmethod
    def set_default(cls, precision: PrecisionLevel | str):
        """Set the default precision for all computations."""
        if isinstance(precision, str):
            precision = PrecisionLevel.from_str(precision)
        cls._default_precision = precision
    
    @classmethod
    def get_default(cls) -> PrecisionLevel:
        """Get the current default precision level."""
        return cls._default_precision
    
    @classmethod
    def get_config(cls, precision: Optional[PrecisionLevel | str] = None) -> PrecisionConfig:
        """Get precision configuration, using default if not specified."""
        if precision is None:
            precision = cls._default_precision
        elif isinstance(precision, str):
            precision = PrecisionLevel.from_str(precision)
        return cls._config_cache[precision]
    
    @classmethod
    def get_float_dtype(cls, precision: Optional[PrecisionLevel | str] = None) -> dtype:
        """Get float dtype for the specified or default precision."""
        config = cls.get_config(precision)
        return config.float_dtype
    
    @classmethod
    def get_complex_dtype(cls, precision: Optional[PrecisionLevel | str] = None) -> dtype:
        """Get complex dtype for the specified or default precision."""
        config = cls.get_config(precision)
        return config.complex_dtype
    
    @classmethod
    def get_int_dtype(cls, precision: Optional[PrecisionLevel | str] = None) -> dtype:
        """Get int dtype for the specified or default precision."""
        config = cls.get_config(precision)
        return config.int_dtype
    
    @classmethod
    def get_epsilon(cls, requested: float = 0.0, precision: Optional[PrecisionLevel | str] = None) -> float:
        """Get epsilon value for the specified or default precision."""
        config = cls.get_config(precision)
        return config.get_epsilon(requested)


# Convenience functions for easy access
def set_precision(precision: PrecisionLevel | str):
    """Set the default precision for all computations in the package."""
    PrecisionContext.set_default(precision)


def get_precision() -> PrecisionLevel:
    """Get the current default precision level."""
    return PrecisionContext.get_default()


def get_float_dtype(precision: Optional[PrecisionLevel | str] = None) -> dtype:
    """Get float dtypes for the specified or default precision."""
    return PrecisionContext.get_float_dtype(precision)

def get_complex_dtype(precision: Optional[PrecisionLevel | str] = None) -> dtype:
    """Get complex dtypes for the specified or default precision."""
    return PrecisionContext.get_complex_dtype(precision)

def get_int_dtype(precision: Optional[PrecisionLevel | str] = None) -> dtype:
    """Get int dtypes for the specified or default precision."""
    return PrecisionContext.get_int_dtype(precision)

def get_epsilon(requested: float = 0.0, precision: Optional[PrecisionLevel | str] = None) -> float:
    """Get epsilon value for the specified or default precision."""
    return PrecisionContext.get_epsilon(requested, precision)