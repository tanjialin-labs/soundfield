"""Apply a temporal activity gate to audio using externally supplied spatial evidence."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.io import read_audio,write_audio
from soundfield.enhancement.spatial_activity import SpatialActivityGate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--hop-samples',type=int,default=256)
    args=parser.parse_args()
    if args.output.resolve() in (args.input.resolve(),args.evidence.resolve()):parser.error('output must differ from inputs')
    if args.hop_samples<1:parser.error('hop-samples must be positive')
    rate,audio=read_audio(args.input);evidence=np.load(args.evidence,allow_pickle=False)
    if evidence.ndim!=1 or len(evidence)!=(len(audio)+args.hop_samples-1)//args.hop_samples:
        parser.error('evidence must contain one count per audio hop')
    gate=SpatialActivityGate();output=audio.copy()
    for frame,value in enumerate(evidence):
        output[frame*args.hop_samples:(frame+1)*args.hop_samples]*=gate.process_frame(value)
    write_audio(args.output,output,rate)


if __name__=='__main__':main()
