"""Suppress echo in an explicit archive of residual, echo and reference spectra."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.aec import SpectralEchoPostfilter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('output must differ from input')
    with np.load(args.input,allow_pickle=False) as archive:
        residual=archive['residual_spectrum'];echo=archive['echo_spectrum'];reference=archive['reference_spectrum']
    if residual.ndim!=3 or residual.shape!=echo.shape or reference.shape[:2]!=residual.shape[:2] or residual.shape[0]!=257:
        parser.error('spectra must have shape (257, frames, channels) with matching frame counts')
    output=np.zeros_like(residual,dtype=complex)
    processors=[SpectralEchoPostfilter() for _ in range(residual.shape[2])]
    for frame in range(residual.shape[1]):
        for channel,processor in enumerate(processors):
            output[:,frame,channel:channel+1]=processor.process_frame(residual[:,frame,channel:channel+1],echo[:,frame,channel:channel+1],reference[:,frame])
    np.savez_compressed(args.output,spectrum=output)


if __name__=='__main__':main()
