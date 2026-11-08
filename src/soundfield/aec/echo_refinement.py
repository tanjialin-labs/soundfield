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

    def __post_init__(self):
        for name in ('frequency_bin_count', 'iterations', 'update_interval_frames'):
            positive_integer(getattr(self, name), name)
        if self.reference_component_count < 0 or self.initial_update_frames < 0:
            raise ValueError('component and initial frame counts must not be negative')
        finite_scalar(self.covariance_smoothing, 'covariance_smoothing', minimum=0, maximum=1)
        if self.distribution not in ('gaussian', 'laplace'):
            raise ValueError('distribution must be gaussian or laplace')


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

    def process_frame(self, residual_spectrum, echo_spectrum, reference_history, *, is_active=True, silent_frames=0):
        config = self.config
        residual = spectral_frame(residual_spectrum, config.frequency_bin_count, 1)[:, 0]
        echo = spectral_frame(echo_spectrum, config.frequency_bin_count, 1)[:, 0]
        history = np.asarray(reference_history, dtype=complex)
        if history.shape != (config.frequency_bin_count, config.reference_component_count) or not np.isfinite(history).all():
            raise ValueError('reference_history has an invalid shape or value')
        if not is_active or silent_frames:
            self._reset_matrices()
            return EchoSpectrumResult(residual[:, None].copy(), np.zeros((len(residual), 1), complex))
        self.active_frame_count += 1
        components = np.column_stack((residual, history, echo))
        outer = components[:, :, None]*components.conj()[:, None, :]
        for _ in range(config.iterations):
            separated = np.sum(components*self.demixing_weights.conj(), axis=1)
            variance = (max(float(np.mean(np.abs(separated)**2)), 1e-4) if config.distribution == 'gaussian'
                        else 2*max(float(np.sqrt(np.sum(np.abs(separated)**2))), 1e-4))
            self.covariance = (config.covariance_smoothing*self.covariance+(1-config.covariance_smoothing)
                               *(outer/variance+np.eye(components.shape[1])*1e-4))
            submatrix = self.covariance[:, 1:, 1:]
            diagonal = np.zeros_like(submatrix)
            index = np.arange(submatrix.shape[1])
            diagonal[:, index, index] = submatrix[:, index, index]
            regularized = .99*submatrix+.01*diagonal+1e-6
            if (self.active_frame_count < config.initial_update_frames
                    or self.active_frame_count % config.update_interval_frames == 0):
                self.demixing_weights[:, 0] = 1
                self.demixing_weights[:, 1:] = -np.linalg.solve(regularized, self.covariance[:, 1:, 0, None])[..., 0]
        separated = np.sum(components*self.demixing_weights.conj(), axis=1)
        estimated = residual-separated
        output = np.where(np.abs(separated)**2 > np.abs(residual)**2, residual, separated)
        return EchoSpectrumResult(output[:, None], estimated[:, None])
