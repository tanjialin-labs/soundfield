"""Measure each beam's windowed speech-to-noise ratio at a chosen frame."""

import argparse
import json
from pathlib import Path

from soundfield import ShortTimeFourierTransform
from soundfield.io import read_audio
from soundfield.metrics import WindowedSnrConfig, WindowedSnrEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--event-frame", type=int)
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if rate != 16000:
        parser.error("this example uses 16 kHz frequency intervals")
    spectrum = ShortTimeFourierTransform().analyze(audio)
    event = spectrum.shape[1] - 1 if arguments.event_frame is None else arguments.event_frame
    if event < 0 or event >= spectrum.shape[1]:
        parser.error("event-frame must be within the audio")
    engine = WindowedSnrEstimator(WindowedSnrConfig(channel_count=audio.shape[1]))
    for frame in range(event):
        engine.evaluate(spectrum[:, frame])
    ratios = [engine.evaluate(spectrum[:, event], wake_active=True, beam_index=channel) for channel in range(audio.shape[1])]
    arguments.output.write_text(json.dumps(dict(frame_index=event, linear_ratios=ratios), indent=2) + "\n")


if __name__ == "__main__":
    main()
