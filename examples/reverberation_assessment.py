"""Measure low- and high-band direct-to-reverberant ratios."""

import argparse
import json
from pathlib import Path

from soundfield.io import read_audio
from soundfield.metrics import BinauralReverberationConfig, BinauralReverberationEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--microphones", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--spacing-meters", type=float, default=.03)
    parser.add_argument("--azimuth-degrees", type=float, default=90)
    parser.add_argument("--microphone-model",choices=("binaural","free_field"),default="binaural")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if rate != 16000:
        parser.error("this example uses 16 kHz frequency intervals")
    config = BinauralReverberationConfig(microphone_indices=tuple(arguments.microphones), microphone_spacing_meters=arguments.spacing_meters,
                                         azimuth_degrees=arguments.azimuth_degrees,microphone_model=arguments.microphone_model)
    result = BinauralReverberationEstimator(config).evaluate(audio)
    arguments.output.write_text(json.dumps(dict(low_band_db=result.low_band_db, high_band_db=result.high_band_db), indent=2) + "\n")


if __name__ == "__main__":
    main()
