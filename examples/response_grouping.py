"""Group measured impulse responses by spectral shape and clarity."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.io import read_audio
from soundfield.acoustics.response_groups import group_impulse_responses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('responses',nargs='+',type=Path)
    parser.add_argument('--threshold',type=float,default=3.)
    args=parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in args.responses}:parser.error('output must differ from input resources')
    responses=[];rates=[];names=[]
    for path in args.responses:
        rate,audio=read_audio(path);rates.append(rate)
        for channel in range(audio.shape[1]):responses.append(audio[:,channel]);names.append(f'{path.name}:{channel}')
    if len(set(rates))!=1:parser.error('all responses must share the sample rate')
    result=group_impulse_responses(responses,rates[0],distance_threshold=args.threshold)
    np.savez_compressed(args.output,**vars(result),response_names=np.asarray(names),sample_rate_hz=rates[0])


if __name__=='__main__':main()
