"""Enhance audio with a noise reference or per-channel separation gains."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.enhancement import NoiseReferencePostfilter, SeparationGuidedPostfilter, SeparationPostfilterConfig
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--noise-reference", type=Path)
    mode.add_argument("--separation-gains", type=Path)
    arguments = parser.parse_args()
    resource = arguments.noise_reference or arguments.separation_gains
    if arguments.output.resolve() in (arguments.input.resolve(), resource.resolve()):
        parser.error("output must differ from input resources")
    rate, audio = read_audio(arguments.input)
    transform = ShortTimeFourierTransform()
    spectrum = transform.analyze(audio)
    output = np.empty_like(spectrum)
    if arguments.noise_reference:
        reference_rate, reference = read_audio(resource)
        if rate != reference_rate or reference.shape != audio.shape or audio.shape[1] != 1:
            parser.error("noise-reference mode requires aligned mono audio")
        control = transform.analyze(reference)
        engine = NoiseReferencePostfilter()
    else:
        control = np.load(resource, allow_pickle=False)
        if control.shape != spectrum.shape:
            parser.error("separation gains must have shape (bins, frames, channels)")
        engine = SeparationGuidedPostfilter(SeparationPostfilterConfig(channel_count=audio.shape[1]))
    for frame in range(spectrum.shape[1]):
        output[:, frame] = engine.process_frame(spectrum[:, frame], control[:, frame])
    write_audio(arguments.output, transform.synthesize(output, length_samples=len(audio)), rate)


if __name__ == "__main__":
    main()
