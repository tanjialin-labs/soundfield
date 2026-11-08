"""Run iterative echo adaptation on one microphone and one reference."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.io import read_audio,write_audio
from soundfield.aec import IterativeEchoCanceller,EchoAudioSynthesizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone',type=Path);parser.add_argument('reference',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.resolve() in (args.microphone.resolve(),args.reference.resolve()):parser.error('output must differ from inputs')
    rate,microphone=read_audio(args.microphone);other_rate,reference=read_audio(args.reference)
    if rate!=16000 or other_rate!=rate or microphone.shape!=reference.shape or microphone.shape[1]!=1:
        parser.error('equal-length mono recordings at 16000 Hz are required')
    processor=IterativeEchoCanceller();synthesizer=EchoAudioSynthesizer(256,enable_deemphasis=False)
    length=len(microphone);padded=((length+255)//256)*256+256
    microphone=np.pad(microphone,((0,padded-length),(0,0)));reference=np.pad(reference,((0,padded-length),(0,0)))
    output=[]
    for start in range(0,padded,256):
        output.append(synthesizer.process_frame(processor.process_block(microphone[start:start+256],reference[start:start+256])))
    write_audio(args.output,np.concatenate(output)[256:256+length],rate)


if __name__=='__main__':main()
