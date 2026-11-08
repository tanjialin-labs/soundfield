"""Save microphone, playback-reference and estimated-echo spectra."""

import argparse
from pathlib import Path

import numpy as np

from soundfield.aec import PlaybackEchoConfig, PlaybackSpectrumAnalyzer, PlaybackSpectrumConfig
from soundfield.io import read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--hop-samples', type=int, default=512)
    parser.add_argument('--partitions', type=int, default=8)
    args = parser.parse_args()
    if args.output.resolve() in (args.microphone.resolve(), args.reference.resolve()):
        parser.error('output must differ from audio inputs')
    rate, microphone = read_audio(args.microphone)
    reference_rate, reference = read_audio(args.reference)
    if rate != 16000 or reference_rate != rate or len(microphone) != len(reference) or microphone.shape[1] != 1:
        parser.error('inputs must be aligned at 16000 Hz with one microphone channel')
    linear = PlaybackEchoConfig(hop_length_samples=args.hop_samples, partition_count=args.partitions,
                                reference_count=reference.shape[1])
    engine = PlaybackSpectrumAnalyzer(PlaybackSpectrumConfig(linear_filter=linear))
    hop = linear.hop_length_samples
    count = (len(microphone)+hop-1)//hop
    microphone = np.pad(microphone, ((0, count*hop-len(microphone)), (0, 0)))
    reference = np.pad(reference, ((0, count*hop-len(reference)), (0, 0)))
    outputs = [engine.process_block(microphone[start:start+hop], reference[start:start+hop])
               for start in range(0, len(microphone), hop)]
    spectra = {}
    for name, channels in (('residual_spectrum', 1), ('echo_spectrum', 1), ('reference_spectrum', linear.reference_count)):
        spectra[name] = np.stack([getattr(result, name) for result in outputs], axis=1) if outputs else np.empty((hop+1, 0, channels), complex)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, sample_rate_hz=rate, hop_length_samples=hop, **spectra)


if __name__ == '__main__':
    main()
