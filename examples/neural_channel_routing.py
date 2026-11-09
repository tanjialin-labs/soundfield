"""Separate source channels and route speech using neural evidence."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.io import read_audio,write_audio
from soundfield.neural import MemoryNetwork,SpectralMaskEstimator
from soundfield.separation import DemixingConfig
from soundfield.pipelines.neural_routing import separate_and_route


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--network',type=Path,required=True);parser.add_argument('--normalization',type=Path,required=True)
    args=parser.parse_args()
    inputs={p.resolve() for p in (args.input,args.network,args.normalization)}
    if args.output.resolve() in inputs:
        parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('the configured feature bank requires 16000 Hz')
    normalization=np.loadtxt(args.normalization)
    estimator=SpectralMaskEstimator(MemoryNetwork.load_text(args.network),*normalization,channel_count=audio.shape[1])
    result=separate_and_route(audio,estimator,DemixingConfig(frequency_bin_count=513,channel_count=audio.shape[1]))
    write_audio(args.output,result.audio,rate)


if __name__=='__main__':main()
