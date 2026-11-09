"""Track speech directions using aligned spectra, candidate scores and a memory network."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.neural import MemoryNetwork,SpectralMaskEstimator
from soundfield.pipelines.speech_tracking import track_speech_directions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--network',type=Path,required=True);parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--search-region',type=int,nargs=2,action='append',metavar=('START','STOP'))
    parser.add_argument('--circular',action='store_true')
    args=parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.input,args.network,args.normalization)}:
        parser.error('output must differ from input resources')
    with np.load(args.input,allow_pickle=False) as resource:
        spectrum=resource['spectrum'];scores=resource['direction_scores'];selected=resource['selected_channels']
    model=MemoryNetwork.load_text(args.network);normalization=np.loadtxt(args.normalization)
    estimator=SpectralMaskEstimator(model,*normalization,channel_count=spectrum.shape[-1])
    indices=track_speech_directions(spectrum,scores,selected,estimator,search_regions=args.search_region,
                                    circular_search=args.circular)
    np.savez_compressed(args.output,direction_indices=indices)


if __name__=='__main__':main()
