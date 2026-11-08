"""Track speech direction using aligned beam masks and microphone audio."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.io import read_audio
from soundfield.transform import ShortTimeFourierTransform
from soundfield.localization import DirectionalMatchingConfig, SpeechDirectionConfig, linear_array
from soundfield.pipelines.directional_capture import localize_speech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--evidence', type=Path, required=True, help='NPZ with speech_masks and selected_channels')
    parser.add_argument('--spacing-meters', type=float, default=.03)
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.evidence.resolve()):
        parser.error('output must differ from input resources')
    rate, audio = read_audio(args.input)
    if rate != 16000:
        parser.error('this example requires 16000 Hz audio')
    spectrum = ShortTimeFourierTransform().analyze(audio)
    with np.load(args.evidence, allow_pickle=False) as resource:
        masks, selected = resource['speech_masks'], resource['selected_channels'][:, 0]
    if masks.shape[:2] != spectrum.shape[:2]:
        parser.error('evidence and audio must share the analysis frame grid')
    chosen = np.take_along_axis(masks, selected[None, :, None], axis=2)[:, :, 0]
    angles = (0, *range(30, 151, 5), 180)
    matching = DirectionalMatchingConfig(geometry=linear_array(audio.shape[1], args.spacing_meters), block_frames=1,
        event_delay_frames=0, analysis_band=(56, 170), scan_angles_degrees=angles,
        beam_intervals_degrees=((0,90),(60,120),(90,180)), smoothing_half_width_steps=3)
    tracking = SpeechDirectionConfig(smoothing=.98, silence_hold_frames=30, silence_reset_frames=70,
                                     speech_threshold=133., history_frames=2)
    result = localize_speech(spectrum, chosen, selected, matching, tracking_config=tracking,
                            search_regions=((0,15),(7,20),(13,27)))
    np.savez_compressed(args.output, **vars(result), sample_rate_hz=rate)


if __name__ == '__main__':
    main()
