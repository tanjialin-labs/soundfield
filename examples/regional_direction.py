"""Estimate a microphone region from pairwise phase and energy evidence."""

import argparse
import json
from pathlib import Path

from soundfield.io import read_audio
from soundfield.transform import FourierConfig,ShortTimeFourierTransform
from soundfield.localization.regional_correlation import RegionalCorrelationConfig,RegionalCorrelationEstimator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('output must differ from input')
    rate,audio=read_audio(args.input)
    if rate!=16000 or not len(audio):parser.error('a nonempty 16000 Hz recording is required')
    spectrum=ShortTimeFourierTransform(FourierConfig(1024,512)).analyze(audio)
    scores,index=RegionalCorrelationEstimator(RegionalCorrelationConfig(channel_count=audio.shape[1])).estimate(spectrum,spectrum)
    args.output.write_text(json.dumps(dict(channel_index=index,pair_scores=scores.tolist()),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
