"""Reconstruct audio through arbitrarily sized Fourier input blocks."""

import argparse
from pathlib import Path

import numpy as np

from soundfield import FourierAnalyzer, FourierSynthesizer
from soundfield.io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--block-samples', type=int, default=1000)
    args = parser.parse_args()
    if args.block_samples <= 0 or args.input.resolve() == args.output.resolve():
        parser.error('block size must be positive and output must differ from input')
    rate, audio = read_audio(args.input)
    analyzer, synthesizer = FourierAnalyzer(), FourierSynthesizer()
    output = []
    for start in range(0, len(audio), args.block_samples):
        output.append(synthesizer.process_block(analyzer.process_block(audio[start:start+args.block_samples])))
    if not len(audio):
        analyzer.process_block(audio)
    output.append(synthesizer.process_block(analyzer.flush()))
    output.append(synthesizer.flush())
    write_audio(args.output, np.concatenate(output)[:len(audio)], rate)


if __name__ == '__main__':
    main()
