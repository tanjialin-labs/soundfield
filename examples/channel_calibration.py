"""Estimate channel sensitivity corrections from a calibration recording."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.io import read_audio,write_audio
from soundfield.acoustics.channel_calibration import calibrate_channel_levels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--parameters',type=Path,required=True)
    parser.add_argument('--segment-starts',nargs='+',type=int)
    parser.add_argument('--segment-length',type=int)
    args=parser.parse_args()
    paths=[p.resolve() for p in (args.input,args.output,args.parameters)]
    if len(set(paths))!=3:parser.error('input, output and parameter paths must differ')
    rate,audio=read_audio(args.input)
    result=calibrate_channel_levels(audio,rate,segment_start_samples=args.segment_starts,segment_length_samples=args.segment_length)
    write_audio(args.output,audio*result.amplitude_gains,rate)
    np.savez_compressed(args.parameters,**vars(result),sample_rate_hz=rate)


if __name__=='__main__':main()
