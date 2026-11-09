"""Broadband direction candidates from phase agreement across microphone pairs."""
from dataclasses import dataclass
import numpy as np
from ..contracts import finite_scalar, positive_integer, real_array, spectral_array
from .geometry import plane_wave_responses


@dataclass(frozen=True)
class PhaseDirectionResult:
    azimuths_degrees: tuple
    vote_fractions: tuple
    scores: np.ndarray


class PhaseDirectionScanner:
    def __init__(self, geometry, frequencies_hz, scan_angles_degrees):
        self.angles = real_array(scan_angles_degrees,'scan_angles_degrees',1).copy()
        frequencies = real_array(frequencies_hz,'frequencies_hz',1)
        if not len(frequencies) or len(self.angles)<3 or len(np.unique(self.angles % 360)) != len(self.angles):
            raise ValueError('direction and frequency grids must contain distinct usable coordinates')
        response = plane_wave_responses(geometry,frequencies,self.angles)
        self._left,self._right = np.triu_indices(geometry.microphone_count,1)
        self._pairs = (response[:,self._left]*response[:,self._right].conj()).conj()
        self._channels = geometry.microphone_count

    def estimate(self, spectrum, *, maximum_sources=2, minimum_separation_degrees=25., minimum_vote_fraction=.08):
        """Score a short (bins, frames, channels) interval and return separated local maxima."""
        positive_integer(maximum_sources,'maximum_sources')
        finite_scalar(minimum_separation_degrees,'minimum_separation_degrees',minimum=0,maximum=180)
        finite_scalar(minimum_vote_fraction,'minimum_vote_fraction',minimum=0,maximum=1)
        x = spectral_array(spectrum)
        if x.shape[0]!=len(self._pairs) or x.shape[2]!=self._channels:
            raise ValueError('spectrum must match the configured frequency grid and array')
        power = np.mean(abs(x)**2,axis=2)
        votes = np.zeros(len(self.angles))
        if power.size and power.max() > 0:
            cross = x[:,:,self._left]*x[:,:,self._right].conj()
            unit = np.divide(cross,abs(cross),out=np.zeros_like(cross),where=abs(cross)>0)
            agreement = np.einsum('ftp,fpd->ftd',unit,self._pairs).real/len(self._left)
            peak = agreement.max(axis=2)
            weights = np.sqrt(power)*np.maximum(peak,0)**2
            weights[power < power.max()*1e-4] = 0.
            votes = np.bincount(agreement.argmax(axis=2).ravel(),weights=weights.ravel(),minlength=len(self.angles))
        total = votes.sum()
        delta = abs((self.angles[:,None]-self.angles[None,:]+180)%360-180)
        smooth = np.exp(-.5*(delta/6.)**2) @ votes
        eligible = np.array([smooth[i] >= smooth[(delta[i] > 0) & (delta[i] <= min(10,minimum_separation_degrees))].max(initial=0.)
                             for i in range(len(votes))])
        angles, fractions = [],[]
        for _ in range(maximum_sources):
            if not total or not eligible.any():
                break
            index = np.argmax(np.where(eligible,smooth,-np.inf))
            fraction = votes[delta[index] <= 10].sum()/total
            if fraction < minimum_vote_fraction:
                break
            angles.append(float(self.angles[index]))
            fractions.append(float(fraction))
            eligible &= delta[index] > minimum_separation_degrees
        return PhaseDirectionResult(tuple(angles),tuple(fractions),smooth)
