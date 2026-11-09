"""Select and separate conference microphones using an explicit speech model."""

import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.neural import SpeechActivityModel
from soundfield.selection.conference_channels import ConferenceSelectionConfig
from soundfield.pipelines.conference_capture import capture_conference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-channels',type=int,default=3)
    parser.add_argument('--prefer-active-speakers',action='store_true')
    args=parser.parse_args()
    if args.output.resolve() in (args.input.resolve(),args.model.resolve()):
        parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('conference capture requires 16000 Hz audio')
    config=ConferenceSelectionConfig(channel_count=audio.shape[1],output_channel_count=args.output_channels,prefer_active_speakers=args.prefer_active_speakers)
    result=capture_conference(audio,SpeechActivityModel.load(args.model),selection_config=config)
    write_audio(args.output,result.audio,rate)


if __name__=='__main__':main()
