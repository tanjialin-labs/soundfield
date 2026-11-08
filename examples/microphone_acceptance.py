"""Inspect microphone energy against explicit digital-level limits."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.io import read_audio
from soundfield.acoustics.channel_calibration import inspect_microphone_energy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--frequency-bins',nargs=2,type=int,default=(1,128))
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--level-limits-db',nargs=2,type=float,default=(-80.,-10.))
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('output must differ from input')
    rate,audio=read_audio(args.input)
    result=inspect_microphone_energy(audio,start_frame_index=args.start_frame,frequency_interval=tuple(args.frequency_bins),
        lower_level_db=args.level_limits_db[0],upper_level_db=args.level_limits_db[1])
    np.savez_compressed(args.output,**vars(result),sample_rate_hz=rate)


if __name__=='__main__':main()
