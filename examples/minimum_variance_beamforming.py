"""Beamform aligned mixture, speech-estimate, and noise-estimate recordings."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.beamforming import MvdrBeamformer, MvdrConfig
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("speech", type=Path)
    parser.add_argument("noise", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-microphone", type=int, help="preserve speech on this zero-based channel")
    arguments = parser.parse_args()
    paths = (arguments.input, arguments.speech, arguments.noise)
    if arguments.output.resolve() in [path.resolve() for path in paths]:
        parser.error("output must differ from all input resources")
    rate, audio = read_audio(arguments.input)
    speech_rate, speech = read_audio(arguments.speech)
    noise_rate, noise = read_audio(arguments.noise)
    if speech_rate != rate or noise_rate != rate or speech.shape != audio.shape or noise.shape != audio.shape:
        parser.error("input rates, sample counts, and channel counts must agree")
    transform = ShortTimeFourierTransform()
    spectrum, speech_spectrum, noise_spectrum = [transform.analyze(value) for value in (audio, speech, noise)]
    engine = MvdrBeamformer(MvdrConfig(microphone_count=audio.shape[1], reference_microphone=arguments.reference_microphone))
    output = np.empty((*spectrum.shape[:2], 1), dtype=np.complex128)
    for frame in range(spectrum.shape[1]):
        output[:, frame] = engine.process_frame(spectrum[:, frame], speech_spectrum[:, frame], noise_spectrum[:, frame])
    write_audio(arguments.output, transform.synthesize(output, length_samples=len(audio)), rate)


if __name__ == "__main__":
    main()
