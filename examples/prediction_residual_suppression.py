"""Filter aligned residual and predicted-component audio."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.dereverb import PredictionResidualFilter
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("residual", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.resolve() in (arguments.residual.resolve(), arguments.prediction.resolve()):
        parser.error("output must differ from both inputs")
    rate, residual = read_audio(arguments.residual)
    prediction_rate, prediction = read_audio(arguments.prediction)
    if rate != prediction_rate or residual.shape != prediction.shape:
        parser.error("input rates, sample counts, and channel counts must agree")
    transform = ShortTimeFourierTransform()
    residual_spectrum, prediction_spectrum = transform.analyze(residual), transform.analyze(prediction)
    engine = PredictionResidualFilter(257, residual.shape[1])
    output = np.empty_like(residual_spectrum)
    for frame in range(output.shape[1]):
        output[:, frame] = engine.process_frame(residual_spectrum[:, frame], prediction_spectrum[:, frame])
    write_audio(arguments.output, transform.synthesize(output, length_samples=len(residual)), rate)


if __name__ == "__main__":
    main()
