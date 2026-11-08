"""Select one beam from an aligned wake-word recording."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.io import read_audio, write_audio
from soundfield.selection import WakeWordBeamSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--active-beams", type=int, nargs="*")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if rate != 16000:
        parser.error("this example uses 16 kHz frequency intervals")
    activity = np.ones(audio.shape[1])
    if arguments.active_beams is not None:
        if any(index < 0 or index >= audio.shape[1] for index in arguments.active_beams):
            parser.error("active beam indices must be within the input channels")
        activity[:] = 0
        activity[arguments.active_beams] = 1
    spectrum = ShortTimeFourierTransform().analyze(audio)
    selected = WakeWordBeamSelector().select(activity, spectrum)
    write_audio(arguments.output, audio[:, selected:selected + 1], rate)


if __name__ == "__main__":
    main()
