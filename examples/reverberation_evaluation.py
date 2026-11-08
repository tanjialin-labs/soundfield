"""Evaluate dereverberation using an explicit clean source and room response."""
import argparse,json
from pathlib import Path
from soundfield.io import read_audio,write_audio
from soundfield.pipelines.reverberation_evaluation import evaluate_reverberation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('response',type=Path);parser.add_argument('output_directory',type=Path)
    parser.add_argument('--response-gain',type=float,default=4.)
    parser.add_argument('--metrics',nargs='+',default=('spectral_distance','coherent_diffuse','modulation','perceptual'))
    args=parser.parse_args()
    rate,clean=read_audio(args.reference);response_rate,response=read_audio(args.response)
    if rate!=response_rate:parser.error('source and impulse-response sample rates must match')
    output=args.output_directory;output.mkdir(parents=True,exist_ok=True)
    inputs={args.reference.resolve(),args.response.resolve()}
    if any((output/name).resolve() in inputs for name in ('reverberant.wav','enhanced.wav','scores.json')):parser.error('outputs must differ from inputs')
    result=evaluate_reverberation(clean,response,rate,response_gain=args.response_gain,metrics=args.metrics)
    write_audio(output/'reverberant.wav',result.reverberant_audio,rate,encoding='float64')
    write_audio(output/'enhanced.wav',result.enhanced_audio,rate,encoding='float64')
    (output/'scores.json').write_text(json.dumps(dict(delay_samples=result.delay_samples,input=result.input_scores,output=result.output_scores),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
