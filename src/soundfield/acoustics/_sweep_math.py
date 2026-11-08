"""Numerical building blocks for synchronized acoustic measurements."""

import math
import numpy as np
from scipy import signal
from scipy.fft import next_fast_len, rfft, irfft, rfftfreq

EPS = 1e-15

def _normalize_peak(x: np.ndarray, peak: float=0.3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    current = np.max(np.abs(x), initial=0.0)
    return x * (peak / current) if current > EPS else x

def _fade_edges(x: np.ndarray, sample_rate_hz: int, ms: float=30.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    n = min(len(x) // 2, int(sample_rate_hz * ms / 1000))
    if n:
        w = np.sin(np.linspace(0, np.pi / 2, n, endpoint=True)) ** 2
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x

def _bandpass(x: np.ndarray, sample_rate_hz: int, lo: float, hi: float, order: int=4) -> np.ndarray:
    hi = min(hi, sample_rate_hz / 2 * 0.95)
    lo = max(lo, 1.0)
    if lo >= hi:
        raise ValueError(f'Invalid band-pass: {lo} .. {hi} for fs={sample_rate_hz}')
    sos = signal.butter(order, [lo, hi], btype='bandpass', fs=sample_rate_hz, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)

def _make_sync_marker(sample_rate_hz: int, seconds: float, seed: int, peak: float=0.25) -> np.ndarray:
    n = max(1024, int(round(seconds * sample_rate_hz)))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z = _bandpass(z, sample_rate_hz, 800, min(6700, sample_rate_hz * 0.41), order=3)
    z *= signal.windows.tukey(n, alpha=0.13)
    z = z - np.mean(z)
    return _normalize_peak(z, peak)

def _make_sweep(sample_rate_hz: int, lower_frequency_hz: float, upper_frequency_hz: float, seconds: float, peak: float=0.35) -> np.ndarray:
    if lower_frequency_hz <= 0 or upper_frequency_hz <= lower_frequency_hz or upper_frequency_hz >= sample_rate_hz * 0.46:
        raise ValueError('ESS band must satisfy 0 < f0 < f1 < 0.46*fs')
    n = int(round(sample_rate_hz * seconds))
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    log_ratio = math.log(upper_frequency_hz / lower_frequency_hz)
    phase = 2 * np.pi * lower_frequency_hz * seconds / log_ratio * np.expm1(t * log_ratio / seconds)
    return _fade_edges(peak * np.sin(phase), sample_rate_hz, ms=45.0)

def _make_probe(sample_rate_hz: int, seconds: float, peak: float=0.27) -> np.ndarray:
    n = int(round(sample_rate_hz * seconds))
    rng = np.random.default_rng(20261104)
    noise = rng.standard_normal(n)
    noise = _bandpass(noise, sample_rate_hz, 160, min(7600, sample_rate_hz * 0.42), order=5)
    t = np.arange(n) / sample_rate_hz
    env = 0.18 + 0.82 * np.maximum(0.0, np.sin(2 * np.pi * (2.4 * t + 0.24 * np.sin(0.59 * t))))
    env *= 0.8 + 0.2 * np.cos(2 * np.pi * 0.37 * t)
    return _fade_edges(_normalize_peak(noise * env, peak), sample_rate_hz, ms=70.0)

def _segment_of(manifest: dict, kind: str) -> dict:
    ss = [s for s in manifest['segments'] if s['type'] == kind]
    if len(ss) != 1:
        raise ValueError(f'Expected exactly one segment {kind}, found {len(ss)}')
    return ss[0]

def _padded_segment(x: np.ndarray, start: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    left = max(0, start)
    right = min(len(x), start + n)
    if right > left:
        out[left - start:right - start] = x[left:right]
    return out

def _match_marker(y: np.ndarray, reference: np.ndarray, lo: int=0, hi: int | None=None) -> tuple[int, float]:
    m = len(reference)
    if len(y) < m:
        raise ValueError('Recording shorter than marker')
    corr = signal.fftconvolve(y, reference[::-1], mode='valid')
    sq = y * y
    cs = np.concatenate(([0.0], np.cumsum(sq)))
    local_norm = np.sqrt(np.maximum(cs[m:] - cs[:-m], EPS) * np.sum(reference ** 2))
    ncc = np.abs(corr) / np.maximum(local_norm, EPS)
    a = max(0, int(lo))
    b = min(len(ncc), int(hi) if hi is not None else len(ncc))
    if a >= b:
        raise ValueError('No possible location for marker')
    i = a + int(np.argmax(ncc[a:b]))
    return (i, float(ncc[i]))

def _align_channels(recording: np.ndarray, sample_rate_hz: int, ref_audio: np.ndarray, manifest: dict, maximum_clock_drift_ppm: float, minimum_marker_correlation: float) -> tuple[np.ndarray, dict]:
    start_seg = _segment_of(manifest, 'start_marker')
    end_seg = _segment_of(manifest, 'end_marker')
    start_ref = ref_audio[start_seg['start']:start_seg['end']]
    end_ref = ref_audio[end_seg['start']:end_seg['end']]
    n_ref = end_seg['start'] - start_seg['start']
    cand = [_match_marker(recording[:, c], start_ref) for c in range(recording.shape[1])]
    best_channel = int(np.argmax([x[1] for x in cand]))
    start_pos, start_corr = cand[best_channel]
    if start_corr < minimum_marker_correlation:
        raise ValueError(f'start marker NCC={start_corr:.3f} < {minimum_marker_correlation:.3f}; likely missing marker/AGC/noise')
    search_radius = max(round(n_ref * maximum_clock_drift_ppm / 1000000), int(0.025 * sample_rate_hz))
    expected_end = start_pos + n_ref
    end_pos, end_corr = _match_marker(recording[:, best_channel], end_ref, max(0, expected_end - search_radius), expected_end + search_radius + 1)
    if end_corr < minimum_marker_correlation:
        raise ValueError(f'end marker NCC={end_corr:.3f} < {minimum_marker_correlation:.3f}; recording truncated?')
    scale = (end_pos - start_pos) / n_ref
    ppm = (scale - 1) * 1000000.0
    if abs(ppm) > maximum_clock_drift_ppm:
        raise ValueError(f'clock skew {ppm:.1f} ppm > limit {maximum_clock_drift_ppm:.0f}')
    idx_play = np.arange(len(ref_audio), dtype=np.float64)
    sample_rec = start_pos + (idx_play - start_seg['start']) * scale
    aligned = np.stack([np.interp(sample_rec, np.arange(len(recording)), recording[:, c], left=0.0, right=0.0) for c in range(recording.shape[1])], axis=1)
    alignment = {'marker_reference_channel_1based': best_channel + 1, 'start_marker_recording_sample': int(start_pos), 'end_marker_recording_sample': int(end_pos), 'start_marker_ncc': round(start_corr, 4), 'end_marker_ncc': round(end_corr, 4), 'clock_recording_samples_per_playback_sample': scale, 'clock_drift_ppm': ppm, 'start_anchor_semantics': 'Playback start marker, estimated from filtered mic signal; cancels unknown common system/propagation delay'}
    return (aligned, alignment)

def _deconvolve_sweep(y: np.ndarray, sweep: np.ndarray, start: int, sample_rate_hz: int, preroll_samples: int, response_length_samples: int, regularization: float, guard_samples: int) -> np.ndarray:
    out_length = len(sweep) + preroll_samples + response_length_samples + guard_samples
    ys = _padded_segment(y, start - preroll_samples, out_length)
    xs = np.zeros(out_length, dtype=np.float64)
    xs[:len(sweep)] = sweep
    fft_length = next_fast_len(out_length)
    X = rfft(xs, fft_length)
    Y = rfft(ys, fft_length)
    power = np.abs(X) ** 2
    lamb = max(regularization * float(power.max()), EPS)
    h = irfft(Y * np.conj(X) / (power + lamb), fft_length)
    return h[:preroll_samples + response_length_samples + guard_samples].copy()

def _arrival_peak(impulse_responses: np.ndarray, preroll_samples: int, sample_rate_hz: int) -> int:
    low = max(0, preroll_samples - round(0.06 * sample_rate_hz))
    hi = min(impulse_responses.shape[-1], preroll_samples + round(0.12 * sample_rate_hz))
    energy = np.median(np.sum(impulse_responses ** 2, axis=1), axis=0)
    if hi <= low or np.max(energy[low:hi], initial=0) <= EPS:
        raise ValueError('No direct sound detected around marker-based zero-lag; check sweep or playback')
    return int(np.argmax(energy[low:hi]) + low)

def _repeat_consistency(impulse_responses: np.ndarray, sample_rate_hz: int) -> dict:
    if len(impulse_responses) < 2:
        return {'repeat_min_corr': None, 'repeat_median_corr': None, 'repeat_median_spectral_mad_db': None}
    pair_corr = []
    spec_differences = []
    fft_length = next_fast_len(len(impulse_responses[0]))
    freqs = rfftfreq(fft_length, 1 / sample_rate_hz)
    band = (freqs >= 200) & (freqs <= min(7500, sample_rate_hz * 0.43))
    spectra = [20 * np.log10(np.abs(rfft(h, fft_length)) + 1e-10) for h in impulse_responses]
    for a in range(len(impulse_responses)):
        for b in range(a + 1, len(impulse_responses)):
            x = impulse_responses[a]
            y = impulse_responses[b]
            dot = float(np.dot(x, y))
            corr = dot / (np.linalg.norm(x) * np.linalg.norm(y) + EPS)
            pair_corr.append(corr)
            spec_differences.append(float(np.median(np.abs(spectra[a][band] - spectra[b][band]))))
    return {'repeat_min_corr': float(min(pair_corr)), 'repeat_median_corr': float(np.median(pair_corr)), 'repeat_median_spectral_mad_db': float(np.median(spec_differences))}

def _combine_repeats(impulse_responses: np.ndarray) -> tuple[np.ndarray, str, int]:
    if len(impulse_responses) == 1:
        return (impulse_responses[0].copy(), 'single', 0)
    normed = impulse_responses / (np.linalg.norm(impulse_responses, axis=1, keepdims=True) + EPS)
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0.0)
    medoid = int(np.argmax(sim.sum(axis=1)))
    if float(np.min(sim[np.triu_indices(len(impulse_responses), k=1)])) > 0.95:
        return (np.mean(impulse_responses, axis=0), 'mean_repeat', medoid)
    return (impulse_responses[medoid].copy(), 'medoid_repeat', medoid)

def _probe_accuracy(pred: np.ndarray, actual: np.ndarray, sample_rate_hz: int, lower_frequency_hz: float, upper_frequency_hz: float) -> dict:
    lo = max(100.0, lower_frequency_hz * 1.1)
    hi = min(upper_frequency_hz * 0.92, sample_rate_hz * 0.42)
    p = _bandpass(pred, sample_rate_hz, lo, hi)
    a = _bandpass(actual, sample_rate_hz, lo, hi)
    n = min(len(p), len(a))
    p = p[:n]
    a = a[:n]
    ap = float(a @ p)
    energy_a = float(a @ a)
    energy_p = float(p @ p)
    resid = a - p
    nmse = math.sqrt(float(resid @ resid) / (energy_a + EPS))
    corr = ap / (math.sqrt(energy_a * energy_p) + EPS)
    gain = ap / (energy_p + EPS)
    gain_db = 20 * math.log10(abs(gain) + EPS)
    maxlag = min(round(0.02 * sample_rate_hz), n // 5)
    mid = signal.correlate(a, p, mode='full', method='fft')
    shifts = signal.correlation_lags(len(a), len(p), mode='full')
    ok = np.abs(shifts) <= maxlag
    lag = int(shifts[ok][np.argmax(np.abs(mid[ok]))])
    return {'probe_band_corr_no_refit': float(corr), 'probe_band_nmse_no_refit': float(nmse), 'probe_error_energy_db': float(20 * math.log10(nmse + EPS)), 'probe_estimated_gain_db': float(gain_db), 'probe_estimated_lag_samples': int(lag)}

def _measurement_quality(markers: dict, clipping: float, repeats: dict, probe: dict) -> tuple[str, list[str]]:
    warnings = []
    severe = []
    if clipping > 0.0001:
        severe.append('possible_clipping')
    if min(markers['start_marker_ncc'], markers['end_marker_ncc']) < 0.22:
        warnings.append('weak_markers')
    if repeats['repeat_min_corr'] is not None:
        if repeats['repeat_min_corr'] < 0.65:
            severe.append('unstable_repeats')
        elif repeats['repeat_min_corr'] < 0.85:
            warnings.append('repeat_correlation_below_0.85')
    if probe['probe_band_corr_no_refit'] < 0.6 or probe['probe_band_nmse_no_refit'] > 1.0:
        severe.append('independent_probe_mismatch')
    elif probe['probe_band_corr_no_refit'] < 0.85 or probe['probe_band_nmse_no_refit'] > 0.6:
        warnings.append('probe_partly_mismatched')
    if abs(probe['probe_estimated_lag_samples']) > 15:
        warnings.append('probe_timing_shift_gt_15_samples')
    return ('reject' if severe else 'review' if warnings else 'pass', severe + warnings)

def _response_features(impulse_response: np.ndarray, sample_rate_hz: int) -> tuple[np.ndarray, dict]:
    fft_length = next_fast_len(max(8 * len(impulse_response), 4 * sample_rate_hz))
    spectrum = np.abs(rfft(impulse_response, fft_length)) ** 2
    freq = rfftfreq(fft_length, 1 / sample_rate_hz)
    centers = np.geomspace(150, min(7500, sample_rate_hz * 0.42), 32)
    mags = []
    for cf in centers:
        inds = (freq > cf * 2 ** (-1 / 12)) & (freq < cf * 2 ** (1 / 12))
        mags.append(10 * np.log10(np.mean(spectrum[inds]) + 1e-15))
    mags = np.array(mags)
    shape_db = np.clip(mags - np.mean(mags), -35, 35)
    head = round(0.05 * sample_rate_hz)
    Ehead = float(np.sum(impulse_response[:head] ** 2))
    Etail = float(np.sum(impulse_response[head:] ** 2))
    c50 = 10 * np.log10((Ehead + EPS) / (Etail + EPS))
    rms_db = 10 * np.log10(float(impulse_response @ impulse_response) + EPS)
    return (shape_db, {'clarity_c50_db': float(c50), 'rir_energy_db': float(rms_db)})
