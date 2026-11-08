"""Estimate azimuths for a linear array and save event results as JSON."""

import argparse
import json
from pathlib import Path

from soundfield import ShortTimeFourierTransform
from soundfield.io import read_audio
from soundfield.localization import DirectionalMatchingConfig, DirectionalMatchingEstimator, SubspaceDirectionConfig, SubspaceDirectionEstimator, linear_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("subspace", "matching"), default="subspace")
    parser.add_argument("--spacing-meters", type=float, default=.035)
    parser.add_argument("--block-frames", type=int, default=40)
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    geometry = linear_array(audio.shape[1], arguments.spacing_meters)
    if arguments.method == "subspace":
        engine = SubspaceDirectionEstimator(SubspaceDirectionConfig(geometry=geometry, sample_rate_hz=rate,
                    block_frames=arguments.block_frames, search_all_directions=True))
    else:
        engine = DirectionalMatchingEstimator(DirectionalMatchingConfig(geometry=geometry, sample_rate_hz=rate,
                    block_frames=arguments.block_frames, event_delay_frames=0, periodic_estimation=True))
    spectrum = ShortTimeFourierTransform().analyze(audio)
    estimates = []
    for frame in range(spectrum.shape[1]):
        if arguments.method == "subspace":
            result = engine.estimate(spectrum[:, frame], wake_confidence=[1, 1, 1] if (frame + 1) % arguments.block_frames == 0 else None)
        else:
            result = engine.estimate(spectrum[:, frame])
        if result is not None:
            estimates.append(dict(frame_index=frame, time_seconds=frame * 256 / rate, azimuth_degrees=result.azimuth_degrees))
    arguments.output.write_text(json.dumps(estimates, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
