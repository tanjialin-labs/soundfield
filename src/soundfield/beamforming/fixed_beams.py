"""Frequency-domain fixed beams and blocking projections."""

import numpy as np

from ..contracts import spectral_array, spectral_frame


class FixedBeamBank:
    """Project microphone frames using explicitly supplied complex coefficients.

    Fixed weights have shape (bins, microphones, beams). Blocking coefficients
    have shape (bins, microphones, references, beams). Coefficients multiply
    the input directly: conjugation, if required, belongs in the coefficients.
    """

    def __init__(self, fixed_beam_weights, blocking_matrix):
        weights = spectral_array(fixed_beam_weights, 3)
        blocking = spectral_array(blocking_matrix, 4)
        if (min(weights.shape) < 1 or min(blocking.shape) < 1
                or blocking.shape[:2] != weights.shape[:2]
                or blocking.shape[3] != weights.shape[2]):
            raise ValueError("fixed beams and blocking matrix dimensions must agree")
        self._fixed_beam_weights = weights.copy()
        self._blocking_matrix = blocking.copy()
        self._fixed_beam_weights.flags.writeable = False
        self._blocking_matrix.flags.writeable = False

    @property
    def fixed_beam_weights(self):
        return self._fixed_beam_weights.copy()

    @property
    def blocking_matrix(self):
        return self._blocking_matrix.copy()

    @property
    def frequency_bin_count(self):
        return self._fixed_beam_weights.shape[0]

    @property
    def microphone_count(self):
        return self._fixed_beam_weights.shape[1]

    @property
    def beam_count(self):
        return self._fixed_beam_weights.shape[2]

    @property
    def reference_count(self):
        return self._blocking_matrix.shape[2]

    def project(self, spectrum_frame):
        """Return fixed beams (bins, beams) and references (bins, refs, beams)."""
        frame = spectral_frame(spectrum_frame, self.frequency_bin_count, self.microphone_count)
        beams = np.einsum("fm,fmb->fb", frame, self._fixed_beam_weights)
        references = np.einsum("fm,fmrb->frb", frame, self._blocking_matrix)
        return beams, references

    @property
    def active_references(self):
        return np.sum(np.abs(self._blocking_matrix), axis=1) > 1e-15
