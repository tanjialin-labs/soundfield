"""Audio and microphone-array signal processing."""

from .enhancement.automatic_gain import (
    AutomaticGainControl,
    GainControlConfig,
    GainCurveConfig,
    GainSmoothingConfig,
)
from .transform.short_time_fourier import FourierConfig, ShortTimeFourierTransform
from .transform.streaming_fourier import FourierAnalyzer, FourierSynthesizer

__all__ = [
    "AutomaticGainControl",
    "FourierConfig",
    "FourierAnalyzer",
    "FourierSynthesizer",
    "GainControlConfig",
    "GainCurveConfig",
    "GainSmoothingConfig",
    "ShortTimeFourierTransform",
]
