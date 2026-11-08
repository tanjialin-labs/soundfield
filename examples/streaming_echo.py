"""Process paired recordings in chunks and optionally save sample timing."""

import argparse
import json
from pathlib import Path

import numpy as np

from soundfield.aec import EchoCancellationConfig, EchoCancellationStream, EchoFilterConfig, InputConditioningConfig
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone')
    parser.add_argument('reference')
    parser.add_argument('output')
    parser.add_argument('--communication-output')
    parser.add_argument('--timeline')
    parser.add_argument('--chunk-samples', type=int, default=320)
    args = parser.parse_args()
    rate, microphone = read_audio(args.microphone)
    reference_rate, reference = read_audio(args.reference)
    if rate != reference_rate or len(microphone) != len(reference) or args.chunk_samples < 1:
        parser.error('paired recordings need equal sample rates and lengths; chunk size must be positive')
    config = EchoCancellationConfig(input=InputConditioningConfig(microphone_count=microphone.shape[1], reference_count=reference.shape[1]),
                                    linear_filter=EchoFilterConfig(reference_count=reference.shape[1]))
    stream = EchoCancellationStream(config, sample_rate_hz=rate)
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
