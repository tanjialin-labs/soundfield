"""Process paired recordings in chunks and optionally save sample timing."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from soundfield.aec import (CommunicationPostfilterConfig, EchoCancellationConfig, EchoCancellationStream,
                            EchoFilterConfig, EchoRefinementConfig, InputConditioningConfig, nonlinear_playback_reference)
from soundfield.aec.reference_alignment import ReferenceAlignmentConfig
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone')
    parser.add_argument('reference')
    parser.add_argument('output')
    parser.add_argument('--communication-output')
    parser.add_argument('--timeline')
    parser.add_argument('--chunk-samples', type=int, default=320)
    parser.add_argument('--fast-tracking', action='store_true', help='track changing echo paths with bounded updates')
    parser.add_argument('--preserve-nearend', action='store_true',
                        help='pass full-band communication audio through while playback is inactive')
    parser.add_argument('--speaker-transition-amplitude', type=float,
                        help='expand mono playback to model loudspeaker saturation at this normalized amplitude')
    parser.add_argument('--refine-echo', action='store_true',
                        help='refine residual echo using playback history and previously fitted weights')
    parser.add_argument('--align-reference', action='store_true', help='Track playback leading microphone capture')
    parser.add_argument('--reverberant-playback', action='store_true', help='Use aligned nonlinear playback and a room-residual profile for mono playback at 16 kHz')
    parser.add_argument('--maximum-drift-ppm', type=float, help='Track relative capture clock drift during alignment')
    args = parser.parse_args()
    if args.maximum_drift_ppm and not (args.align_reference or args.reverberant_playback):
        parser.error('clock drift tracking requires --align-reference')
    rate, microphone = read_audio(args.microphone)
    reference_rate, reference = read_audio(args.reference)
    if rate != reference_rate or len(microphone) != len(reference) or args.chunk_samples < 1:
        parser.error('paired recordings need equal sample rates and lengths; chunk size must be positive')
    transition = args.speaker_transition_amplitude
    if args.reverberant_playback:
        if rate!=16000 or reference.shape[1]!=1:
            parser.error('the reverberant playback profile requires mono playback at 16 kHz')
        transition = .12 if transition is None else transition
    if transition is not None:
        reference = nonlinear_playback_reference(reference, transition_amplitude=transition)
    communication = CommunicationPostfilterConfig(
        inactive_gain_policy='preserve_nearend' if args.preserve_nearend else 'wiener')
    if args.reverberant_playback:
        config = EchoCancellationConfig.reverberant_playback(microphone.shape[1],reference.shape[1])
        if args.communication_output:
            config = replace(config, communication_filter=communication)
    else:
        config = EchoCancellationConfig(input=InputConditioningConfig(microphone_count=microphone.shape[1],reference_count=reference.shape[1]),
                                        linear_filter=EchoFilterConfig(reference_count=reference.shape[1]), track_path_changes=args.fast_tracking,
                                        communication_filter=communication)
    if args.refine_echo:
        config = replace(config, refinement=EchoRefinementConfig.predictive())
    drift = args.maximum_drift_ppm if args.maximum_drift_ppm is not None else (500. if args.reverberant_playback else 0.)
    alignment = ReferenceAlignmentConfig(maximum_drift_ppm=drift) if args.align_reference or args.reverberant_playback else None
    stream = EchoCancellationStream(config, sample_rate_hz=rate, reference_alignment=alignment)
    chunks = []
    for start in range(0, len(microphone), args.chunk_samples):
        chunks.extend(stream.process_block(microphone[start:start+args.chunk_samples], reference[start:start+args.chunk_samples], start_sample=start))
    chunks.extend(stream.flush())
    def assemble(name):
        return np.concatenate([getattr(chunk, name) for chunk in chunks]) if chunks else np.empty_like(microphone)
    write_audio(args.output, assemble('audio'), rate)
    if args.communication_output:
        write_audio(args.communication_output, assemble('communication_audio'), rate)
    if args.timeline:
        Path(args.timeline).write_text(json.dumps(dict(sample_rate_hz=rate, latency_samples=stream.latency_samples,
            startup_samples=stream.startup_samples, chunks=[dict(start_sample=chunk.start_sample,
            stop_sample=chunk.stop_sample, available_after_sample=chunk.available_after_sample,
            released_on_flush=chunk.released_on_flush) for chunk in chunks]), indent=2)+'\n')


if __name__ == '__main__':
    main()
