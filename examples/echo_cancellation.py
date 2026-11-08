"""Cancel acoustic echo with explicit microphone and reference recordings."""

import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.aec import (InputConditioningConfig,EchoFilterConfig,ResidualEchoConfig,EchoCancellationConfig,cancel_echo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--method',choices=('constrained','spectral'),default='constrained')
    parser.add_argument('--hop-samples',type=int,default=256)
    parser.add_argument('--partitions',type=int,default=8)
    args=parser.parse_args()
    if args.output.resolve() in (args.microphone.resolve(),args.reference.resolve()):
        parser.error('output must differ from input recordings')
    rate,microphone=read_audio(args.microphone)
    reference_rate,reference=read_audio(args.reference)
    if rate!=16000 or reference_rate!=rate:parser.error('both recordings must use 16000 Hz')
    hop=args.hop_samples
    config=EchoCancellationConfig(
        input=InputConditioningConfig(hop,microphone.shape[1],reference.shape[1]),
        linear_filter=EchoFilterConfig(hop,reference.shape[1],args.partitions,method=args.method),
        residual_filter=ResidualEchoConfig(frequency_bin_count=hop+1,energy_intervals=((0,hop+1),)),
        communication_filter=None)
    output,_=cancel_echo(microphone,reference,config)
    write_audio(args.output,output,rate)


if __name__=='__main__':main()
