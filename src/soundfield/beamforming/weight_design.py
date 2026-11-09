"""Steering responses, constrained beam design and signal-subspace blocking."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from ..contracts import finite_scalar, positive_integer, real_array
from .fixed_beams import FixedBeamBank


def array_steering_vectors(microphone_positions_meters, angles_degrees, frequencies_hz, *, sound_speed_meters_per_second=340.0):
    """Return (bins, microphones, directions), using linear angles or polar/azimuth pairs."""
    positions = np.asarray(microphone_positions_meters, dtype=float)
    frequencies = real_array(frequencies_hz, 'frequencies_hz', 1)
    angles = np.asarray(angles_degrees, dtype=float)
    finite_scalar(sound_speed_meters_per_second, 'sound_speed_meters_per_second', minimum=1)
    if not np.isfinite(positions).all() or not np.isfinite(angles).all():
        raise ValueError('positions and angles must be finite')
    if positions.ndim == 1 or positions.ndim == 2 and positions.shape[1] == 1:
        if angles.ndim != 1:
            raise ValueError('linear arrays require one angle per direction')
        delays = positions.reshape(-1, 1)*np.cos(np.deg2rad(angles))[None, :]
    elif positions.ndim == 2 and positions.shape[1] == 3:
        if angles.ndim != 2 or angles.shape[1] != 2:
            raise ValueError('spatial arrays require (polar, azimuth) angle pairs')
        polar, azimuth = np.deg2rad(angles).T
        direction = -np.stack((np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth), np.cos(polar)))
        delays = positions @ direction
    else:
        raise ValueError('microphone positions must have one or three coordinates')
    if delays.shape[0] < 1 or delays.shape[1] < 1 or len(frequencies) < 1:
        raise ValueError('steering grids must not be empty')
    return np.exp(-2j*np.pi*frequencies[:, None, None]*delays[None]/sound_speed_meters_per_second)


@dataclass(frozen=True)
class BeamOptimizationResult:
    fixed_beam_weights: np.ndarray
    objective_value: float
    solver_status: str
    maximum_constraint_violation: float


def _optimizer():
    try:
        import cvxpy
    except ImportError as exception:
        raise ImportError('Constrained beam design requires soundfield[design]') from exception
    return cvxpy


def _finish_problem(problem, weights, constraints, solver):
    problem.solve(solver=solver)
    if problem.status not in ('optimal', 'optimal_inaccurate') or weights.value is None:
        raise ValueError('beam constraints have no available solution: '+str(problem.status))
    violation = max((float(np.max(constraint.violation(), initial=0)) for constraint in constraints), default=0.)
    if not np.isfinite(weights.value).all() or violation > 2e-4:
        raise ArithmeticError('beam solution does not meet the requested constraints')
    return BeamOptimizationResult(np.asarray(weights.value).T.copy(), float(problem.value), problem.status, violation)


def design_linear_beam(microphone_positions_meters, passband_degrees, *, fft_length_samples=512,
                       sample_rate_hz=16000, pass_amplitude_maximum=None, pass_real_minimum=None,
                       coefficient_norm_maximum=4.0, frequency_indices=None, solver='CLARABEL'):
    """Minimize summed stopband magnitude with real passband and coefficient-norm bounds."""
    positive_integer(fft_length_samples, 'fft_length_samples')
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    positions = real_array(microphone_positions_meters, 'microphone_positions_meters', 1)
    if len(positions) < 2 or fft_length_samples % 2:
        raise ValueError('at least two microphones and an even FFT length are required')
    start, stop = passband_degrees
    if not 0 <= start <= stop <= 180:
        raise ValueError('passband must lie inside [0, 180] degrees')
    bins = fft_length_samples//2+1
    frequencies = np.arange(bins)*sample_rate_hz/fft_length_samples
    angles = np.arange(0, 181, 5)
    passing = np.arange(int(np.floor(start/5)), int(np.floor(stop/5))+1)
    stopping = np.setdiff1d(np.arange(len(angles)), passing)
    if not len(stopping):
        raise ValueError('at least one stopband direction is required')
    indices = np.arange(bins) if frequency_indices is None else np.asarray(frequency_indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or np.any(indices < 0) or np.any(indices >= bins):
        raise ValueError('frequency_indices must identify a nonempty FFT subset')
    maximum = (1.15 if len(positions) == 2 else 1.05) if pass_amplitude_maximum is None else pass_amplitude_maximum
    minimum = (.85 if len(positions) == 2 else .9) if pass_real_minimum is None else pass_real_minimum
    for value, name in ((maximum, 'pass_amplitude_maximum'), (minimum, 'pass_real_minimum'),
                        (coefficient_norm_maximum, 'coefficient_norm_maximum')):
        finite_scalar(value, name, minimum=0)
    if minimum > maximum:
        raise ValueError('real passband minimum cannot exceed its amplitude maximum')
    steering = array_steering_vectors(positions, angles, frequencies[indices])
    cp = _optimizer(); weights = cp.Variable((len(positions), len(indices)), complex=True)
    passing_response = cp.vstack([cp.sum(cp.multiply(weights, steering[:, :, index].T), axis=0) for index in passing])
    stop_response = cp.vstack([cp.sum(cp.multiply(weights, steering[:, :, index].T), axis=0) for index in stopping])
    constraints = [cp.abs(passing_response) <= maximum, cp.real(passing_response) >= minimum,
                   cp.norm(weights, axis=0) <= coefficient_norm_maximum]
    problem = cp.Problem(cp.Minimize(cp.sum(cp.abs(stop_response))/stop_response.size), constraints)
    result = _finish_problem(problem, weights, constraints, solver)
    complete = np.zeros((bins, len(positions)), complex)
    complete[indices] = result.fixed_beam_weights
    return BeamOptimizationResult(complete, result.objective_value, result.solver_status, result.maximum_constraint_violation)


def design_circular_beam(radius_meters, microphone_count, center_degrees, *, half_width_degrees=30.,
                         upper_pass_frequency_hz=2000., stop_amplitude_maximum=.7, pass_deviation_maximum=.5,
                         high_frequency_norm_maximum=1., fft_length_samples=512, sample_rate_hz=16000,
                         solver='CLARABEL'):
    """Minimize the global stopband peak while constraining look response across frequency."""
    positive_integer(microphone_count, 'microphone_count')
    positive_integer(fft_length_samples, 'fft_length_samples')
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    finite_scalar(radius_meters, 'radius_meters', minimum=np.finfo(float).eps)
    if microphone_count not in (4, 6) or fft_length_samples % 2:
        raise ValueError('circular design requires four or six microphones and an even FFT length')
    finite_scalar(center_degrees, 'center_degrees')
    finite_scalar(half_width_degrees, 'half_width_degrees', minimum=0, maximum=85)
    if center_degrees % 5 or half_width_degrees % 5:
        raise ValueError('beam angles must lie on the five-degree steering grid')
    for value, name in ((upper_pass_frequency_hz, 'upper_pass_frequency_hz'),
                        (stop_amplitude_maximum, 'stop_amplitude_maximum'),
                        (pass_deviation_maximum, 'pass_deviation_maximum'),
                        (high_frequency_norm_maximum, 'high_frequency_norm_maximum')):
        finite_scalar(value, name, minimum=np.finfo(float).eps)
    if not 500 <= upper_pass_frequency_hz <= sample_rate_hz/2:
        raise ValueError('upper pass frequency must lie between 500 Hz and Nyquist')
    phase = np.arange(microphone_count)*2*np.pi/microphone_count
    positions = radius_meters*np.stack((np.cos(phase), np.sin(phase), np.zeros_like(phase)), axis=1)
    bins = fft_length_samples//2+1
    frequencies = np.fft.rfftfreq(fft_length_samples, 1/sample_rate_hz)
    steering = array_steering_vectors(positions, np.column_stack((np.full(72, 90), np.arange(72)*5)), frequencies)
    def indices(width):
        return np.arange(int((center_degrees-width)/5), int((center_degrees+width)/5)+1) % 72
    passing = indices(half_width_degrees)
    stopping = np.setdiff1d(np.arange(72), indices(half_width_degrees+20))
    distant = np.setdiff1d(np.arange(72), indices(half_width_degrees+60))
    if not len(stopping) or not len(distant):
        raise ValueError('the angular grid must contain both stopband regions')
    def band(low, high):
        return np.arange(int(np.floor(low*fft_length_samples/sample_rate_hz)),
                         min(bins-1, int(np.floor(high*fft_length_samples/sample_rate_hz)))+1)
    active = band(50, min(4000 if microphone_count == 4 else 5000, sample_rate_hz/2))
    outside = np.setdiff1d(np.arange(bins), active)
    pass_frequencies = band(500, upper_pass_frequency_hz)
    wide_frequencies = band(upper_pass_frequency_hz if microphone_count == 4 else 3200, sample_rate_hz/2)
    bounded_frequencies = band(2800, sample_rate_hz/2)
    cp = _optimizer(); weights = cp.Variable((microphone_count, bins), complex=True)
    responses = cp.vstack([cp.sum(cp.multiply(weights, steering[:, :, direction].T), axis=0) for direction in range(72)])
    look = responses[passing[(len(passing)-1)//2]]
    constraints = [cp.norm(look-1, 2) <= .1, cp.abs(responses[distant][:, pass_frequencies]) <= stop_amplitude_maximum,
                   cp.abs(responses[passing][:, wide_frequencies]-1) <= pass_deviation_maximum,
                   cp.norm(weights[:, active], axis=0) <= 3.,
                   cp.norm(weights[:, bounded_frequencies], axis=0) <= high_frequency_norm_maximum]
    if len(outside):
        constraints.append(cp.abs(responses[stopping][:, outside]) <= 1.1)
    problem = cp.Problem(cp.Minimize(cp.max(cp.abs(responses[stopping][:, active]))), constraints)
    return _finish_problem(problem, weights, constraints, solver)


def design_subspace_blocking(steering_vectors, *, retained_energy_fraction=.9, reference_count=None,
                             reverse_basis=False, scale=1.):
    """Return conjugated trailing singular vectors as (bins, microphones, references)."""
    steering = np.asarray(steering_vectors, dtype=complex)
    if steering.ndim != 3 or min(steering.shape) < 1 or not np.isfinite(steering).all():
        raise ValueError('steering_vectors must be finite (bins, microphones, directions)')
    finite_scalar(retained_energy_fraction, 'retained_energy_fraction', minimum=0, maximum=1)
    finite_scalar(scale, 'scale', minimum=0)
    microphones = steering.shape[1]
    count = microphones if reference_count is None else reference_count
    positive_integer(count, 'reference_count')
    if count > microphones:
        raise ValueError('reference count cannot exceed microphone count')
    result = np.zeros((len(steering), microphones, count), complex)
    for frequency, matrix in enumerate(steering):
        basis, singular, _ = svd(matrix, full_matrices=True, lapack_driver='gesvd')
        energy = singular**2
        if energy.sum() == 0:
            raise ValueError('steering directions must contain nonzero energy')
        rank = min(microphones, int(np.searchsorted(np.cumsum(energy)/energy.sum(), retained_energy_fraction, side='right'))+1)
        complement = basis[:, rank:]
        if reverse_basis:
            complement = complement[:, ::-1]
        active = min(count, complement.shape[1])
        result[frequency, :, :active] = complement[:, :active].conj()*scale
    return result


def design_linear_blocking(microphone_positions_meters, passband_degrees, *, fft_length_samples=512,
                           sample_rate_hz=16000, retained_energy_fraction=None):
    positions = real_array(microphone_positions_meters, 'microphone_positions_meters', 1)
    positive_integer(fft_length_samples, 'fft_length_samples')
    positive_integer(sample_rate_hz, 'sample_rate_hz')
    start, stop = passband_degrees
    if len(positions) < 2 or fft_length_samples % 2 or not 0 <= start <= stop <= 180:
        raise ValueError('linear blocking needs at least two microphones, an even FFT and a valid angular band')
    angles = np.arange(np.floor(start/5), np.floor(stop/5)+1)*5
    frequencies = np.arange(fft_length_samples//2+1)*sample_rate_hz/fft_length_samples
    fraction = (.85 if len(positions) == 2 else .9) if retained_energy_fraction is None else retained_energy_fraction
    return design_subspace_blocking(array_steering_vectors(positions, angles, frequencies), retained_energy_fraction=fraction)


def design_circular_blocking(radius_meters, microphone_count, *, half_width_degrees=30., beam_count=None,
                             fft_length_samples=512, sample_rate_hz=16000):
    """Build reversed subspace complements with layout-specific high-band references."""
    count = microphone_count if beam_count is None else beam_count
    for value, name in ((count, 'beam_count'), (fft_length_samples, 'fft_length_samples'), (sample_rate_hz, 'sample_rate_hz')):
        positive_integer(value, name)
    finite_scalar(radius_meters, 'radius_meters', minimum=np.finfo(float).eps)
    finite_scalar(half_width_degrees, 'half_width_degrees', minimum=5, maximum=175)
    if microphone_count not in (4, 6) or fft_length_samples % 2:
        raise ValueError('circular blocking requires four or six microphones and an even FFT length')
    phase = np.arange(microphone_count)*2*np.pi/microphone_count
    positions = radius_meters*np.stack((np.cos(phase), np.sin(phase), np.zeros_like(phase)), axis=1)
    bins = fft_length_samples//2+1
    frequencies = np.fft.rfftfreq(fft_length_samples, 1/sample_rate_hz)
    result = np.zeros((bins, microphone_count, microphone_count-2, count), complex)
    for beam in range(count):
        center = beam*360/count
        azimuths = np.arange(np.ceil((center-half_width_degrees)/5), np.floor((center+half_width_degrees)/5))*5 % 360
        angles = np.array([(polar, azimuth) for polar in (90, 110) for azimuth in azimuths])
        steering = array_steering_vectors(positions, angles, frequencies)
        for frequency, matrix in enumerate(steering):
            basis, singular, _ = svd(matrix, full_matrices=True, lapack_driver='gesvd')
            fraction = np.cumsum(singular**2)/np.sum(singular**2)
            rank = min(microphone_count, int(np.searchsorted(fraction, .9, side='right'))+1)
            blocked = np.zeros((microphone_count, microphone_count), complex)
            blocked[:, :microphone_count-rank] = basis[:, rank:][:, ::-1].conj()
            if microphone_count == 4 and frequency > 160:
                blocked[:, 1] = basis[:, 2].conj()
            if microphone_count == 6:
                blocked[:, 1] += blocked[:, 4]
            result[frequency, :, :, beam] = blocked[:, :microphone_count-2]/2
    return result


def combine_beam_banks(banks):
    """Concatenate compatible beam resources along their beam axes."""
    banks = tuple(banks)
    if not banks:
        raise ValueError('at least one beam bank is required')
    dimensions = {(bank.frequency_bin_count, bank.microphone_count, bank.reference_count) for bank in banks}
    if len(dimensions) != 1:
        raise ValueError('beam banks must agree on frequency, microphone and reference counts')
    return FixedBeamBank(np.concatenate([bank.fixed_beam_weights for bank in banks], axis=2),
                         np.concatenate([bank.blocking_matrix for bank in banks], axis=3))
