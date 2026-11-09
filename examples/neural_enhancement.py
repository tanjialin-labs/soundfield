"""Enhance 16 kHz audio with a memory network and optional spatial filtering."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.io import read_audio, write_audio
from soundfield.neural import MemoryNetwork, SpectralMaskEstimator
from soundfield.pipelines.neural_enhancement import enhance_with_neural_masks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--network',type=Path,required=True)
    parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--beamform',action='store_true')
    parser.add_argument('--reference-microphone',type=int,help='reference channel for spatial output')
    parser.add_argument('--no-gain-control',action='store_true',help='output the enhanced signal before level control')
    args=parser.parse_args()
    if args.reference_microphone is not None and not args.beamform:
        parser.error('--reference-microphone requires --beamform')
    if args.output.resolve() in {p.resolve() for p in (args.input,args.network,args.normalization)}:
        parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('the configured feature bank requires 16000 Hz audio')
    model=MemoryNetwork.load_text(args.network)
    normalization=np.loadtxt(args.normalization)
    if normalization.ndim!=2 or normalization.shape[0]!=2:
        parser.error('normalization must contain two rows: offset and scale')
    estimator=SpectralMaskEstimator(model,*normalization,channel_count=audio.shape[1])
    write_audio(args.output,enhance_with_neural_masks(audio,estimator,beamform=args.beamform,
        reference_microphone=args.reference_microphone,apply_gain_control=not args.no_gain_control),rate)


if __name__=='__main__':main()
