"""Cancel a supplied interference reference from mono target audio."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.adaptive import NlmsConfig, NormalizedLmsCanceller
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.resolve() in (arguments.target.resolve(), arguments.reference.resolve()):
        parser.error("output must differ from both inputs")
    rate, target = read_audio(arguments.target)
    reference_rate, reference = read_audio(arguments.reference)
    if reference_rate != rate or len(reference) != len(target) or target.shape[1] != 1:
        parser.error("target must be mono; input rates and sample counts must agree")
    transform = ShortTimeFourierTransform()
    target_spectrum, reference_spectrum = transform.analyze(target), transform.analyze(reference)
    engine = NormalizedLmsCanceller(NlmsConfig(sample_rate_hz=rate, reference_count=reference.shape[1]))
    output = np.empty_like(target_spectrum)
    strength = np.ones((transform.config.frequency_bin_count, 1))
    for frame in range(output.shape[1]):
        output[:, frame] = engine.process_frame(target_spectrum[:, frame], reference_spectrum[:, frame], strength)
    write_audio(arguments.output, transform.synthesize(output, length_samples=len(target)), rate)


if __name__ == "__main__":
    main()
