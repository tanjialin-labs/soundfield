"""Reduce predictable reverberation with optional residual suppression."""

import argparse
from dataclasses import replace
from pathlib import Path

from soundfield.dereverb import ResidualSuppressionConfig, WpeConfig
from soundfield.io import read_audio, write_audio
from soundfield.pipelines import reduce_reverberation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low-latency", action="store_true", help="16 kHz WPE with tap-block covariance and early speech protection")
    parser.add_argument("--postfilter", action="store_true")
    parser.add_argument("--coherence-bypass", action="store_true", help="Protect coherent multichannel speech from unnecessary attenuation")
    parser.add_argument("--precision", choices=("single","double"), default="double")
    parser.add_argument("--current-frame", action="store_true")
    parser.add_argument("--compensate-delay", action="store_true")
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if arguments.low_latency and rate != 16000:
        parser.error("the low-latency profile requires 16 kHz input")
    factory = WpeConfig.low_latency if arguments.low_latency else WpeConfig
    config = factory(channel_count=audio.shape[1])
    config = replace(config, precision=arguments.precision, coherence_bypass=arguments.coherence_bypass, output_delay_frames=0 if arguments.current_frame else config.output_delay_frames,
                     postfilter=ResidualSuppressionConfig() if arguments.postfilter else None)
    output = reduce_reverberation(audio, config, compensate_delay=arguments.compensate_delay)
    write_audio(arguments.output, output, rate)


if __name__ == "__main__":
    main()
