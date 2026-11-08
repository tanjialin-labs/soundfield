"""Save per-channel speech probabilities for 160-sample audio hops."""

import argparse
import json
from pathlib import Path
import numpy as np

from soundfield.io import read_audio
from soundfield.neural import SpeechActivityDetector, SpeechActivityModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve() in (args.input.resolve(),args.model.resolve()):
        parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('speech activity requires 16000 Hz audio')
    processor=SpeechActivityDetector(SpeechActivityModel.load(args.model),audio.shape[1])
    results=[]
    for start in range(0,len(audio),160):
        block=audio[start:start+160]
        probability=processor.process_block(np.pad(block,((0,160-len(block)),(0,0))))
        results.append(dict(start_sample=start,valid_samples=len(block),speech_probability=probability.tolist()))
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
