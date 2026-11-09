"""Route microphone or beam channels using temporal energy evidence."""

import argparse
from pathlib import Path

from soundfield.io import read_audio, write_audio
from soundfield.selection import ChannelRoutingConfig
from soundfield.pipelines import route_audio_channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compensate-delay", action="store_true")
    parser.add_argument("--transition-frames", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("output must differ from input")
    rate, audio = read_audio(arguments.input)
    if rate != 16000:
        parser.error("this example uses 16 kHz frequency intervals")
    write_audio(arguments.output, route_audio_channels(audio, ChannelRoutingConfig(channel_count=audio.shape[1],transition_frames=arguments.transition_frames), compensate_delay=arguments.compensate_delay), rate)


if __name__ == "__main__":
    main()
