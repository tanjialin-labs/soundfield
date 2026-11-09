"""Covariance-guided residual refinement using reference and echo components."""

from dataclasses import dataclass

import numpy as np

from ..contracts import positive_integer, finite_scalar, spectral_frame
from .partitioned_filter import EchoSpectrumResult


@dataclass(frozen=True)
class EchoRefinementConfig:
    frequency_bin_count: int = 513
    reference_component_count: int = 3
    iterations: int = 1
    distribution: str = 'gaussian'
    covariance_smoothing: float = 0.98
    update_interval_frames: int = 1
    initial_update_frames: int = 0
    use_prior_weights: bool = False

    def __post_init__(self):
        for name in ('frequency_bin_count', 'iterations', 'update_interval_frames'):
            positive_integer(getattr(self, name), name)
        if self.reference_component_count < 0 or self.initial_update_frames < 0:
            raise ValueError('component and initial frame counts must not be negative')
        finite_scalar(self.covariance_smoothing, 'covariance_smoothing', minimum=0, maximum=1)
        if not isinstance(self.use_prior_weights, bool):
            raise TypeError('use_prior_weights must be boolean')
        if self.distribution not in ('gaussian', 'laplace'):
            raise ValueError('distribution must be gaussian or laplace')

    @classmethod
    def predictive(cls, *, frequency_bin_count=513, reference_component_count=3):
        """Apply the preceding estimate and refit every third active frame."""
        return cls(frequency_bin_count=frequency_bin_count, reference_component_count=reference_component_count,
                   update_interval_frames=3, use_prior_weights=True)


class CovarianceEchoRefiner:
    def __init__(self, config=EchoRefinementConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.active_frame_count = 0
        self._reset_matrices()

    def _reset_matrices(self):
        bins, components = self.config.frequency_bin_count, self.config.reference_component_count+2
        self.covariance = np.tile(np.eye(components, dtype=complex)*.001, (bins, 1, 1))
        self.demixing_weights = np.ones((bins, components), complex)
        if self.config.use_prior_weights:
            self.demixing_weights[:, 1:] = 0

    def process_frame(self, residual_spectrum, echo_spectrum, reference_history, *, is_active=True, silent_frames=0):
        config = self.config
        residual = spectral_frame(residual_spectrum, config.frequency_bin_count, 1)[:, 0]
        echo = spectral_frame(echo_spectrum, config.frequency_bin_count, 1)[:, 0]
        history = np.asarray(reference_history, dtype=complex)
        if history.shape != (config.frequency_bin_count, config.reference_component_count) or not np.isfinite(history).all():
            raise ValueError('reference_history has an invalid shape or value')
        if not is_active or silent_frames:
            if not config.use_prior_weights:
                self._reset_matrices()
            return EchoSpectrumResult(residual[:, None].copy(), np.zeros((len(residual), 1), complex))
        self.active_frame_count += 1
        components = np.column_stack((residual, history, echo))
        outer = components[:, :, None]*components.conj()[:, None, :]
        for iteration in range(config.iterations):
            separated = np.sum(components*self.demixing_weights.conj(), axis=1)
            if config.use_prior_weights and iteration == 0:
                prior = separated
            variance = (max(float(np.mean(np.abs(separated)**2)), 1e-4) if config.distribution == 'gaussian'
                        else 2*max(float(np.sqrt(np.sum(np.abs(separated)**2))), 1e-4))
            self.covariance = (config.covariance_smoothing*self.covariance+(1-config.covariance_smoothing)
                               *(outer/variance+np.eye(components.shape[1])*1e-4))
            if (self.active_frame_count < config.initial_update_frames
                    or self.active_frame_count % config.update_interval_frames == 0):
                submatrix = self.covariance[:, 1:, 1:]
                index = np.arange(submatrix.shape[1])
                regularized = .99*submatrix
                regularized[:, index, index] += .01*submatrix[:, index, index]
                if config.use_prior_weights:
                    regularized[:, index, index] += 1e-6
                else:
                    regularized += 1e-6
                self.demixing_weights[:, 0] = 1
                self.demixing_weights[:, 1:] = -np.linalg.solve(regularized, self.covariance[:, 1:, 0, None])[..., 0]
        separated = prior if config.use_prior_weights else np.sum(components*self.demixing_weights.conj(), axis=1)
        output = np.where(np.abs(separated)**2 > np.abs(residual)**2, residual, separated)
        estimated = residual-(output if config.use_prior_weights else separated)
        return EchoSpectrumResult(output[:, None], estimated[:, None])
