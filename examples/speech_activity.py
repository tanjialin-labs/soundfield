"""Save per-channel speech probabilities for 160-sample audio hops."""

import argparse
import json
from pathlib import Path
import numpy as np

from soundfield.neural.speech_activity import SpeechActivityGate
from soundfield.io import read_audio
from soundfield.neural import SpeechActivityDetector, SpeechActivityModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument("--decisions", action="store_true", help="Include hysteretic activity decisions")
    args=parser.parse_args()
    if args.output.resolve() in (args.input.resolve(),args.model.resolve()):
        parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('speech activity requires 16000 Hz audio')
    processor=SpeechActivityDetector(SpeechActivityModel.load(args.model),audio.shape[1])
    results=[]
    gate=SpeechActivityGate(audio.shape[1]) if args.decisions else None
    for start in range(0,len(audio)+processor.latency_samples,160):
        block=audio[start:start+160]
        probability=processor.process_block(np.pad(block,((0,160-len(block)),(0,0))))
        target=start-processor.latency_samples
        if 0 <= target < len(audio):
            row=dict(start_sample=target,valid_samples=min(160,len(audio)-target),speech_probability=probability.tolist())
            if gate is not None:
                row['active']=gate.process_frame(probability).tolist()
            results.append(row)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
