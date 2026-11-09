"""Enhance every audio channel using cross-channel noise correlation."""

import argparse
from pathlib import Path

from soundfield.enhancement import CorrelationSuppressionConfig
from soundfield.io import read_audio, write_audio
from soundfield.pipelines import suppress_correlated_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--channel-model", choices=("interference_reference", "coherent_speech"),
                        default="interference_reference")
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.input.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    config = CorrelationSuppressionConfig(channel_count=audio.shape[1], channel_model=arguments.channel_model)
    write_audio(arguments.output, suppress_correlated_noise(audio, config), rate)


if __name__ == "__main__":
    main()
