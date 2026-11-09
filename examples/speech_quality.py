"""Measure signal quality with explicit spectral and perceptual conventions."""
import argparse,json
import numpy as np
from pathlib import Path
from soundfield.metrics import compare_waveforms, integrated_loudness, interpolated_peak
from soundfield.io import read_audio
from soundfield.metrics.signal_quality import log_spectral_distance,perceptual_speech_quality,speech_intelligibility
from soundfield.metrics.modulation_energy import modulation_energy_ratio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('processed',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--channel',type=int,default=0)
    parser.add_argument('--metrics',nargs='+',choices=('spectral_distance','perceptual','intelligibility','modulation','waveform','levels'),default=('waveform','levels','spectral_distance','perceptual','intelligibility','modulation'))
    parser.add_argument('--perceptual-mode',choices=('narrowband','wideband'),default='narrowband')
    parser.add_argument('--intelligibility-window',choices=('hann','hann_inset'),default='hann')
    parser.add_argument('--normalized-modulation',action='store_true')
    args=parser.parse_args()
    if args.output.resolve() in {args.reference.resolve(),args.processed.resolve()}:parser.error('output must differ from input resources')
    rate,clean=read_audio(args.reference);processed_rate,processed=read_audio(args.processed)
    if rate!=processed_rate:parser.error('sample rates must match')
    if not 0<=args.channel<min(clean.shape[1],processed.shape[1]):parser.error('channel must exist in both recordings')
    if len(clean) != len(processed):parser.error('recordings must share a sample timeline')
    programme = (clean,processed)
    clean,processed=clean[:,args.channel],processed[:,args.channel]
    result=dict(schema_version=2,sample_rate_hz=rate,channel=args.channel,samples=len(clean),
                alignment="sample_aligned",gain_normalization=False)
    for metric in args.metrics:
        if metric=='spectral_distance':result[metric]=log_spectral_distance(clean,processed,rate,magnitude_floor=1e-12)
        elif metric=='perceptual':result[metric]=vars(perceptual_speech_quality(clean,processed,rate,mode=args.perceptual_mode))
        elif metric=='intelligibility':result[metric]=dict(score=speech_intelligibility(clean,processed,rate,window=args.intelligibility_window),window=args.intelligibility_window)
        elif metric=='levels':
            result[metric]={}
            for name,signal in zip(('reference','processed'),programme):
                channels = list(range(signal.shape[1])) if signal.shape[1] <= 2 else [args.channel]
                available = signal[:,channels]
                result[metric][name]=dict(channels=channels,
                    integrated_lufs=integrated_loudness(available,rate) if rate >= 8000 else None,
                    interpolated_peak_linear=interpolated_peak(available,rate).tolist())
        elif metric=='waveform':
            result[metric]={}
            for name,values in vars(compare_waveforms(clean[:,None],processed[:,None])).items():
                value=float(values[0])
                result[metric][name]=(value if np.isfinite(value) else None if np.isnan(value)
                                      else "+infinity" if value>0 else "-infinity")
        else:result[metric]=dict(score=modulation_energy_ratio(processed,rate,normalized=args.normalized_modulation).ratio,normalized=args.normalized_modulation)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
