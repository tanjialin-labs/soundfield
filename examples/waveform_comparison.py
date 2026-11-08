"""Compare aligned recordings without changing their gain or sample positions."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from soundfield.io import read_audio
from soundfield.metrics import compare_waveforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('processed')
    parser.add_argument('output')
    parser.add_argument('--remove-mean', action='store_true')
    args = parser.parse_args()
    rate, reference = read_audio(args.reference)
    processed_rate, processed = read_audio(args.processed)
    if rate != processed_rate:
        parser.error('sample rates must agree')
    result = compare_waveforms(reference, processed, remove_mean=args.remove_mean)
    def encode(value):
        if np.isnan(value):
            return 'undefined'
        if np.isinf(value):
            return '+infinity' if value > 0 else '-infinity'
        return float(value)
    Path(args.output).write_text(json.dumps({field.name: [encode(value) for value in getattr(result, field.name)]
                                           for field in fields(result)}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
