"""Complete audio-processing operations."""

from .gain_control import apply_automatic_gain
from .array_enhancement import enhance_microphone_array
from .source_separation import separate_sources, separate_stationary_pair
from .spatial_enhancement import enhance_with_spatial_mask, reduce_reverberation, suppress_correlated_noise
from .spatial_selection import classify_nearfield_audio, route_audio_channels
from .neural_enhancement import enhance_with_neural_masks
from .conference_capture import ConferenceCaptureResult, capture_conference
from .speech_tracking import track_speech_directions

__all__ = ["apply_automatic_gain", "enhance_microphone_array", "separate_sources", "separate_stationary_pair",
           "enhance_with_spatial_mask", "reduce_reverberation", "suppress_correlated_noise",
           "classify_nearfield_audio", "route_audio_channels", "enhance_with_neural_masks",
           "ConferenceCaptureResult", "capture_conference", "track_speech_directions"]

from .beam_separation import BeamSeparationResult, separate_beams
from .farfield_capture import FarfieldCaptureResult, capture_farfield, estimate_aligned_masks, render_farfield_spectra
from .directional_capture import DirectionTrackingResult, DirectedCaptureResult, localize_speech, capture_directed_region
from .nearfield_capture import NearfieldCaptureResult, capture_nearfield
from .neural_routing import NeuralRoutingResult, separate_and_route, select_wakeword_events
from .reverberation_evaluation import ReverberationEvaluation, evaluate_reverberation

__all__ += ['BeamSeparationResult', 'separate_beams', 'FarfieldCaptureResult', 'capture_farfield', 'estimate_aligned_masks', 'render_farfield_spectra', 'DirectionTrackingResult', 'DirectedCaptureResult', 'localize_speech', 'capture_directed_region', 'NearfieldCaptureResult', 'capture_nearfield', 'NeuralRoutingResult', 'separate_and_route', 'select_wakeword_events', 'ReverberationEvaluation', 'evaluate_reverberation']
