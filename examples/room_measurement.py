"""Generate synchronized sweeps or estimate channel impulse responses."""
import argparse
import json
from pathlib import Path
import numpy as np
from soundfield.io import read_audio,write_audio
from soundfield.io.measurement_resources import save_sweep_stimulus,load_sweep_stimulus
from soundfield.acoustics.sweep_measurement import SweepConfig,create_sweep_stimulus,measure_impulse_response,resample_impulse_response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    generate=commands.add_parser('generate')
    generate.add_argument('output_directory',type=Path)
    generate.add_argument('--sample-rate',type=int,default=48000)
    generate.add_argument('--band-hz',type=float,nargs=2,default=(100,8000))
    generate.add_argument('--sweep-seconds',type=float,default=8)
    generate.add_argument('--repeats',type=int,default=3)
    generate.add_argument('--probe',type=Path)
    measure=commands.add_parser('measure')
    measure.add_argument('stimulus',type=Path)
    measure.add_argument('recording',type=Path)
    measure.add_argument('output_directory',type=Path)
    measure.add_argument('--response-seconds',type=float,default=1.15)
    measure.add_argument('--export-sample-rate',type=int)
    args=parser.parse_args()
    output=args.output_directory
    output.mkdir(parents=True,exist_ok=True)
    if args.command=='generate':
        config=SweepConfig(sample_rate_hz=args.sample_rate,lower_frequency_hz=args.band_hz[0],upper_frequency_hz=args.band_hz[1],
            sweep_seconds=args.sweep_seconds,repeat_count=args.repeats)
        probe={}
        if args.probe:
            if args.probe.resolve() in ((output/'excitation.wav').resolve(),(output/'stimulus.json').resolve()):parser.error('probe must differ from output resources')
            rate,audio=read_audio(args.probe);probe=dict(probe_audio=audio,probe_sample_rate_hz=rate)
        save_sweep_stimulus(output/'stimulus.json',output/'excitation.wav',create_sweep_stimulus(config,**probe))
    else:
        stimulus=load_sweep_stimulus(args.stimulus)
        inputs={args.recording.resolve(),args.stimulus.resolve()}
        outputs={output/name for name in ('response.wav','measurement.npz','quality.json','response_resampled.wav')}
        if inputs & {p.resolve() for p in outputs}:parser.error('measurement outputs must differ from input resources')
        rate,audio=read_audio(args.recording)
        result=measure_impulse_response(audio,rate,stimulus,response_seconds=args.response_seconds)
        write_audio(output/'response.wav',result.impulse_response,result.sample_rate_hz,encoding='float64')
        np.savez_compressed(output/'measurement.npz',impulse_response=result.impulse_response,repeated_responses=result.repeated_responses,
            predicted_probe=result.predicted_probe,recorded_probe=result.recorded_probe,response_offset_samples=result.response_offset_samples,
            sample_rate_hz=result.sample_rate_hz)
        (output/'quality.json').write_text(json.dumps(dict(alignment=result.alignment,channels=result.channel_quality),indent=2,allow_nan=False)+'\n')
        if args.export_sample_rate:
            resampled=resample_impulse_response(result.impulse_response,result.sample_rate_hz,args.export_sample_rate)
            write_audio(output/'response_resampled.wav',resampled,args.export_sample_rate,encoding='float64')


if __name__=='__main__':main()
