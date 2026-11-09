"""Cancel multichannel playback echo with coupled noise suppression."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from soundfield.aec import CoupledEchoCanceller, CoupledEchoConfig
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--noise-window-ms', type=float, help='Duration of the noise-minimum tracking window')
    parser.add_argument('--require-echo-for-double-talk', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() in (args.microphone.resolve(), args.reference.resolve()):
        parser.error('output must differ from audio inputs')
    rate, microphone = read_audio(args.microphone)
    reference_rate, reference = read_audio(args.reference)
    if rate != 16000 or reference_rate != rate or len(microphone) != len(reference):
        parser.error('audio must be aligned at 16000 Hz')
    config = CoupledEchoConfig()
    if args.require_echo_for_double_talk:
        config = replace(config, speech_detection=replace(config.speech_detection,require_echo_for_double_talk=True))
    if args.noise_window_ms is not None:
        if not np.isfinite(args.noise_window_ms) or args.noise_window_ms <= 0:
            parser.error('noise window must be positive and finite')
        frames = max(1, round(args.noise_window_ms * rate / (1000 * config.postfilter.hop_length_samples)))
        config = replace(config, postfilter=replace(config.postfilter, minimum_window_frames=frames))
    config = replace(config, input=replace(config.input, microphone_count=microphone.shape[1], reference_count=reference.shape[1]),
                     linear_filter=replace(config.linear_filter, reference_count=reference.shape[1]))
    engine = CoupledEchoCanceller(config)
    hop, length = engine.latency_samples, len(microphone)
    pad = (-length) % hop
    microphone = np.pad(microphone, ((0, pad), (0, 0)))
    reference = np.pad(reference, ((0, pad), (0, 0)))
    blocks = [engine.process_block(microphone[start:start+hop], reference[start:start+hop]).audio
              for start in range(0, len(microphone), hop)]
    blocks.append(engine.flush())
    output = np.concatenate(blocks)[hop:hop+length]
    write_audio(args.output, output, rate)


if __name__ == '__main__':
    main()
