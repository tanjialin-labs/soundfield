"""Evaluate a recording manifest with explicit geometry and reference channels."""
import argparse,json
from pathlib import Path
from soundfield.io import read_audio
from soundfield.metrics import BinauralReverberationConfig,BinauralReverberationEstimator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();records=json.loads(args.manifest.read_text())
    inputs={args.manifest.resolve()}
    for record in records:
        inputs.add((args.manifest.parent/record['audio_file']).resolve())
        if record.get('reference_file'):inputs.add((args.manifest.parent/record['reference_file']).resolve())
    if args.output.resolve() in inputs:parser.error('output must differ from all input resources')
    results=[]
    for record in records:
        rate,audio=read_audio(args.manifest.parent/record['audio_file'])
        config=BinauralReverberationConfig(sample_rate_hz=rate,reference_cancellation=None,**record.get('parameters',{}))
        reference=None
        if record.get('reference_file'):
            from dataclasses import replace
            from soundfield.metrics import ReferenceCancellationConfig
            reference_rate,reference=read_audio(args.manifest.parent/record['reference_file'])
            if reference_rate!=rate:parser.error('reference rate must match recording')
            config=replace(config,reference_cancellation=ReferenceCancellationConfig())
        result=BinauralReverberationEstimator(config).evaluate(audio,reference_audio=reference)
        results.append(dict(id=record['id'],low_band_db=result.low_band_db,high_band_db=result.high_band_db))
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
