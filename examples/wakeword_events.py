"""Select beams for a sequence of timestamped wake-word events."""
import argparse,json
from pathlib import Path
import numpy as np
from soundfield.io import read_audio
from soundfield.pipelines.neural_routing import select_wakeword_events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('events',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.resolve() in {args.input.resolve(),args.events.resolve()}:parser.error('output must differ from input resources')
    rate,audio=read_audio(args.input)
    if rate!=16000:parser.error('this configuration requires 16000 Hz')
    events=json.loads(args.events.read_text())
    times=np.array([event['seconds'] for event in events]);activity=np.array([event['beam_activity'] for event in events])
    choices=select_wakeword_events(audio,rate,times,activity)
    args.output.write_text(json.dumps([dict(seconds=float(time),selected_channel=int(choice)) for time,choice in zip(times,choices)],indent=2)+'\n')


if __name__=='__main__':main()
