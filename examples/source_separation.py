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
    parser.add_argument('--speaker-model',type=Path)
    parser.add_argument('--enrolment',type=Path,nargs=2,metavar=('FIRST','SECOND'))
    arguments = parser.parse_args()
    if bool(arguments.enrolment) != bool(arguments.speaker_model):
        parser.error('speaker routing requires the model and both enrolments')
    if any(p.resolve()==arguments.output.resolve() for p in ([arguments.speaker_model,*arguments.enrolment] if arguments.enrolment else [])):
        parser.error('output must differ from speaker resources')
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    channels = audio.shape[1]
    if arguments.stationary_pair:
        if channels != 2 or arguments.output_mode is not None or arguments.distribution != "gaussian":
            parser.error("stationary pair separation uses two microphones, Gaussian weights and source-image rendering")
        output = separate_stationary_pair(audio)
    else:
        postfilter = SeparationPostfilterConfig(frequency_bin_count=2049, channel_count=channels,
                speech_start_bin=64, speech_stop_bin=897) if arguments.output_mode == "postfilter" else None
        config = DemixingConfig(channel_count=channels, source_distribution=arguments.distribution,
                                output_mode=arguments.output_mode or "reference_mask", postfilter=postfilter)
        output = separate_sources(audio,config)
    if arguments.enrolment:
        from soundfield.separation.speaker_identity import EnrolledPairRouter
        recordings = [read_audio(p) for p in arguments.enrolment]
        if any(r != rate for r,_ in recordings):
            parser.error('enrolment and input sample rates must agree')
        router = EnrolledPairRouter(arguments.speaker_model,[x for _,x in recordings],rate)
        for start in range(0,len(output),2*rate):
            block = output[start:start+2*rate]
            block[:] = router.process_block(block)
    write_audio(arguments.output, output, rate)


if __name__ == "__main__":
    main()
