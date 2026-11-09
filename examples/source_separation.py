"""Separate multichannel audio using online source-dependent spectral masks."""

import argparse
from pathlib import Path

from soundfield.enhancement import SeparationPostfilterConfig
from soundfield.io import read_audio, write_audio
from soundfield.pipelines import separate_sources, separate_stationary_pair
from soundfield.separation import DemixingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output-mode", choices=("channel_mask", "reference_mask", "mixture_mask", "postfilter"), default=None)
    parser.add_argument("--distribution", choices=("gaussian", "laplace"), default="gaussian")
    parser.add_argument("--stationary-pair", action="store_true", help="fit two sources using the complete two-microphone recording")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    channels = audio.shape[1]
    if arguments.stationary_pair:
        if channels != 2 or arguments.output_mode is not None or arguments.distribution != "gaussian":
            parser.error("stationary pair separation uses two microphones, Gaussian weights and source-image rendering")
        write_audio(arguments.output, separate_stationary_pair(audio), rate)
        return
    postfilter = SeparationPostfilterConfig(frequency_bin_count=2049, channel_count=channels,
                                           speech_start_bin=64, speech_stop_bin=897) if arguments.output_mode == "postfilter" else None
    config = DemixingConfig(channel_count=channels, source_distribution=arguments.distribution,
                            output_mode=arguments.output_mode or "reference_mask", postfilter=postfilter)
    write_audio(arguments.output, separate_sources(audio, config), rate)


if __name__ == "__main__":
    main()
