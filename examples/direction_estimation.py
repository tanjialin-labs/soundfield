"""Estimate azimuths for a linear or circular array and save event results as JSON."""

import argparse
import json
import math
from pathlib import Path

from soundfield import ShortTimeFourierTransform
import numpy as np
from soundfield.localization.phase_scan import PhaseDirectionScanner
from soundfield.io import read_audio
from soundfield.localization import DirectionalMatchingConfig, DirectionalMatchingEstimator, SubspaceDirectionConfig, SubspaceDirectionEstimator, circular_array, linear_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("subspace", "matching", "phase_scan"), default="subspace")
    parser.add_argument("--array", choices=("linear", "circular"), default="linear")
    parser.add_argument("--radius-meters", type=float, default=.035)
    parser.add_argument("--spacing-meters", type=float, default=.035)
    parser.add_argument("--block-frames", type=int, default=40)
    parser.add_argument("--frequency-voting", choices=("raw", "normalized", "coherent"), default="raw",
                        help="combine subspace scores across frequency")
    parser.add_argument("--band-hz", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--maximum-sources",type=int,default=2,help="Maximum candidates for phase scanning")
    arguments = parser.parse_args()
    if arguments.method != "subspace" and arguments.frequency_voting != "raw":
        parser.error("frequency voting requires the subspace method")
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    circular = arguments.array == "circular"
    geometry = (circular_array(audio.shape[1], arguments.radius_meters) if circular
                else linear_array(audio.shape[1], arguments.spacing_meters))
    band = (48, 151)
    if arguments.band_hz is not None:
        low, high = arguments.band_hz
        if not 0 <= low < high <= rate / 2:
            parser.error("band must lie between zero and Nyquist")
        band = (math.ceil(low * 512 / rate), math.floor(high * 512 / rate) + 1)
        if band[0] >= band[1]:
            parser.error("band must include an FFT bin")
    if arguments.method == "phase_scan":
        if arguments.block_frames < 1:
            parser.error("block-frames must be positive")
        spectrum = ShortTimeFourierTransform().analyze(audio)[slice(*band)]
        scanner = PhaseDirectionScanner(geometry,np.arange(*band)*rate/512,np.arange(0,360 if circular else 181,5))
        estimates = []
        for start in range(0,spectrum.shape[1],arguments.block_frames):
            result = scanner.estimate(spectrum[:,start:start+arguments.block_frames],maximum_sources=arguments.maximum_sources)
            estimates.append(dict(start_frame=start, azimuths_degrees=result.azimuths_degrees,vote_fractions=result.vote_fractions))
        arguments.output.write_text(json.dumps(estimates,indent=2,allow_nan=False)+"\n")
        return
    if arguments.method == "subspace":
        grid = dict(beam_count=4, directions_per_beam=18, beam_width_degrees=90, grid_step_degrees=5,
                    azimuth_offset_degrees=0, search_half_width_steps=9, circular_search=True) if circular else {}
        engine = SubspaceDirectionEstimator(SubspaceDirectionConfig(geometry=geometry, sample_rate_hz=rate,
                    block_frames=arguments.block_frames, search_all_directions=True, analysis_band=band,
                    frequency_voting=arguments.frequency_voting, **grid))
    else:
        grid = dict(scan_angles_degrees=tuple(range(0,360,5)),
                    beam_intervals_degrees=((315,45),(45,135),(135,225),(225,315)),
                    circular_search=True, center_beam_constraint=False) if circular else {}
        engine = DirectionalMatchingEstimator(DirectionalMatchingConfig(geometry=geometry, sample_rate_hz=rate,
                    block_frames=arguments.block_frames, event_delay_frames=0, periodic_estimation=True, analysis_band=band, **grid))
    spectrum = ShortTimeFourierTransform().analyze(audio)
    estimates = []
    for frame in range(spectrum.shape[1]):
        if arguments.method == "subspace":
            result = engine.estimate(spectrum[:, frame], wake_confidence=[1] * engine.config.beam_count if (frame + 1) % arguments.block_frames == 0 else None)
        else:
            result = engine.estimate(spectrum[:, frame])
        if result is not None:
            estimates.append(dict(frame_index=frame, time_seconds=frame * 256 / rate, azimuth_degrees=result.azimuth_degrees))
    arguments.output.write_text(json.dumps(estimates, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
