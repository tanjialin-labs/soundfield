"""Generate residual, echo and reference spectra for downstream spectral suppression."""

import argparse
from pathlib import Path
import numpy as np

from soundfield.io import read_audio
from soundfield.aec import AcousticEchoCanceller,EchoCancellationConfig,EchoFilterConfig,InputConditioningConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('microphone',type=Path);parser.add_argument('reference',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.resolve() in (args.microphone.resolve(),args.reference.resolve()):parser.error('output must differ from inputs')
    rate,microphone=read_audio(args.microphone);other_rate,reference=read_audio(args.reference)
    if rate!=16000 or other_rate!=rate or len(microphone)!=len(reference):
        parser.error('equal-length recordings at 16000 Hz are required')
    config=EchoCancellationConfig(input=InputConditioningConfig(256,microphone.shape[1],reference.shape[1],enable_preemphasis=False),
        linear_filter=EchoFilterConfig(256,reference.shape[1]),reference_activity=None,residual_filter=None,communication_filter=None)
    processor=AcousticEchoCanceller(config)
    frames=(len(microphone)+255)//256
    residual=np.zeros((257,frames,microphone.shape[1]),complex);echo=residual.copy()
    reference_spectrum=np.zeros((257,frames,reference.shape[1]),complex)
    for frame in range(frames):
        start=frame*256;length=min(256,len(microphone)-start)
        mic=np.pad(microphone[start:start+256],((0,256-length),(0,0)))
        ref=np.pad(reference[start:start+256],((0,256-length),(0,0)))
        result=processor.process_block(mic,ref)
        residual[:,frame],echo[:,frame]=result.residual_spectrum,result.echo_spectrum
        reference_spectrum[:,frame]=processor.reference_buffer.spectrum
    np.savez_compressed(args.output,residual_spectrum=residual,echo_spectrum=echo,reference_spectrum=reference_spectrum,
                        sample_count=len(microphone),hop_length_samples=256,sample_rate_hz=rate)


if __name__=='__main__':main()
