"""Beamform audio using supplied speech gains or online separation evidence."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import ShortTimeFourierTransform
from soundfield.beamforming import MaskGuidedBeamformer, MaskGuidedConfig, MvdrConfig
from soundfield.contracts import unit_interval_array
from soundfield.io import read_audio, write_audio
from soundfield.separation import DemixingConfig, OnlineSourceSeparator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--speech-gains", type=Path)
    source.add_argument("--estimate-gains", action="store_true")
    arguments = parser.parse_args()
    resources = [arguments.input] + ([arguments.speech_gains] if arguments.speech_gains else [])
    if arguments.output.resolve() in [path.resolve() for path in resources]:
        parser.error("output must differ from all input resources")
    rate, audio = read_audio(arguments.input)
    transform = ShortTimeFourierTransform()
    spectrum = transform.analyze(audio)
    beamformer = MaskGuidedBeamformer(MaskGuidedConfig(MvdrConfig(microphone_count=audio.shape[1])))
    separator = None
    if arguments.estimate_gains:
        separator = OnlineSourceSeparator(DemixingConfig(frequency_bin_count=257, channel_count=audio.shape[1]))
    else:
        gains = unit_interval_array(np.load(arguments.speech_gains, allow_pickle=False), "speech_gains", (*spectrum.shape[:2], 1))
    output = np.empty((*spectrum.shape[:2], 1), dtype=np.complex128)
    for frame in range(spectrum.shape[1]):
        if separator is not None:
            separator.process_frame(spectrum[:, frame])
            gain = np.max(separator.separation_gain, axis=1, keepdims=True)**2
        else:
            gain = gains[:, frame]
        output[:, frame] = beamformer.process_frame(spectrum[:, frame], gain)
    write_audio(arguments.output, transform.synthesize(output, length_samples=len(audio)), rate)


if __name__ == "__main__":
    main()
