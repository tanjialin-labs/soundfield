"""Capture speech using beam selection, spatial filtering or a fixed direction."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from soundfield.io import read_audio, write_audio
from soundfield.neural import MemoryNetwork, SpectralMaskEstimator
from soundfield.enhancement import GainControlConfig
from soundfield.beamforming import MaskGuidedConfig, MvdrConfig
from soundfield.pipelines.farfield_capture import capture_farfield
from soundfield.pipelines.source_separation import separate_sources
from soundfield.separation import DemixingConfig
from soundfield.io.gain_settings import load_gain_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--mode', choices=('beam_selection', 'spatial', 'fixed_direction', 'ambient'), default='beam_selection')
    parser.add_argument('--channels', type=int, nargs='+')
    parser.add_argument('--digital-gain', type=float)
    features = parser.add_mutually_exclusive_group()
    features.add_argument('--feature-audio', type=Path)
    features.add_argument('--separate-features', action='store_true')
    parser.add_argument('--evidence', type=Path, help='NPZ with frame_probability, bin_probability and statistical_gain')
    parser.add_argument('--stages', type=Path)
    parser.add_argument('--no-gain-control', action='store_true')
    parser.add_argument('--gain-settings', type=Path)
    parser.add_argument('--no-spatial-filter', action='store_true')
    parser.add_argument('--reference-microphone', type=int, help='Preserve the selected microphone in spatial mode')
    parser.add_argument('--per-channel-mask', action='store_true')
    args = parser.parse_args()
    inputs = {path.resolve() for path in (args.input, args.network, args.normalization, args.feature_audio, args.evidence, args.gain_settings) if path}
    outputs = [path.resolve() for path in (args.output, args.stages) if path]
    if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
        parser.error('outputs must be distinct from each other and from input resources')
    rate, samples = read_audio(args.input)
    if rate != 16000:
        parser.error('the configured feature bank requires 16000 Hz')
    if args.channels:
        if min(args.channels) < 0 or max(args.channels) >= samples.shape[1] or len(set(args.channels)) != len(args.channels):
            parser.error('channels must select distinct zero-based input channels')
        samples = samples[:, args.channels]
    spatial_config = None
    if args.reference_microphone is not None:
        if args.mode != 'spatial' or args.no_spatial_filter or not 0 <= args.reference_microphone < samples.shape[1]:
            parser.error('reference microphone must identify an input to the active spatial filter')
        spatial_config = MaskGuidedConfig(MvdrConfig(microphone_count=samples.shape[1], reference_microphone=args.reference_microphone))
    gain = args.digital_gain if args.digital_gain is not None else (2. if args.mode == 'spatial' else 4. if args.mode in ('fixed_direction', 'ambient') else 1.)
    feature_audio = None
    if args.feature_audio:
        feature_rate, feature_audio = read_audio(args.feature_audio)
        if feature_rate != rate:
            parser.error('feature audio sample rate must match input')
    elif args.separate_features:
        feature_audio = separate_sources(samples*gain, DemixingConfig(frequency_bin_count=513, channel_count=samples.shape[1], energy_stop_bin=257))
        feature_audio = np.round(feature_audio*32767)/32768
    model = MemoryNetwork.load_text(args.network)
    normalization = np.loadtxt(args.normalization)
    if normalization.ndim != 2 or normalization.shape[0] != 2:
        parser.error('normalization must contain offset and scale rows')
    estimator = SpectralMaskEstimator(model, *normalization, channel_count=(samples if feature_audio is None else feature_audio).shape[1])
    evidence = {}
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as resource:
            evidence = {name: resource[name] for name in ('frame_probability', 'bin_probability', 'statistical_gain')}
    if args.mode == 'fixed_direction' and not evidence:
        parser.error('fixed_direction requires probability and statistical-gain evidence')
    config = load_gain_settings(args.gain_settings) if args.gain_settings else replace(GainControlConfig(), enable_equalizer=args.mode != 'fixed_direction')
    if args.no_gain_control:
        config = None
    result = capture_farfield(samples, estimator, mode=args.mode, feature_audio=feature_audio, digital_gain=gain,
        gain_config=config, synthesis_scale=32768/32767 if args.mode in ('fixed_direction', 'ambient') else 1.,
        enable_spatial_filter=not args.no_spatial_filter, spatial_config=spatial_config, ambient_shared_mask=not args.per_channel_mask, **evidence)
    write_audio(args.output, result.audio, rate)
    if args.stages:
        np.savez_compressed(args.stages, **vars(result), sample_rate_hz=rate)


if __name__ == '__main__':
    main()
