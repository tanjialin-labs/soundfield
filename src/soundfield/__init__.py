"""Audio and microphone-array signal processing."""

from .enhancement.automatic_gain import (
    AutomaticGainControl,
    GainControlConfig,
    GainCurveConfig,
    GainSmoothingConfig,
)
from .transform.short_time_fourier import FourierConfig, ShortTimeFourierTransform

__all__ = [
    "AutomaticGainControl",
    "FourierConfig",
    "GainControlConfig",
    "GainCurveConfig",
    "GainSmoothingConfig",
    "ShortTimeFourierTransform",
]
