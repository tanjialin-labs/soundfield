"""Design fixed beams and blocking matrices for linear or circular arrays."""
import argparse
from pathlib import Path
import numpy as np
from soundfield.beamforming import FixedBeamBank
from soundfield.beamforming.weight_design import design_linear_beam,design_linear_blocking,design_circular_beam,design_circular_blocking
from soundfield.io import save_beam_bank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--geometry',choices=('linear','circular'),default='linear')
    parser.add_argument('--microphones',type=int,default=2)
    parser.add_argument('--spacing-meters',type=float,default=.03)
    parser.add_argument('--radius-meters',type=float,default=.035)
    parser.add_argument('--centers-degrees',type=float,nargs='+',default=None)
    parser.add_argument('--half-width-degrees',type=float,default=30.)
    parser.add_argument('--fft-length',type=int,default=512)
    parser.add_argument('--sample-rate',type=int,default=16000)
    args=parser.parse_args()
    options=dict(fft_length_samples=args.fft_length,sample_rate_hz=args.sample_rate)
    if args.geometry=='linear':
        positions=np.arange(args.microphones)*args.spacing_meters
        centers = args.centers_degrees if args.centers_degrees is not None else (30., 90., 150.)
        intervals=[(max(0,center-args.half_width_degrees),min(180,center+args.half_width_degrees)) for center in centers]
        fixed=np.stack([design_linear_beam(positions,interval,**options).fixed_beam_weights for interval in intervals],axis=2)
        blocking=np.stack([design_linear_blocking(positions,interval,**options) for interval in intervals],axis=3)
    else:
        centers = args.centers_degrees if args.centers_degrees is not None else np.arange(args.microphones)*360/args.microphones
        fixed=np.stack([design_circular_beam(args.radius_meters,args.microphones,center,half_width_degrees=args.half_width_degrees,**options).fixed_beam_weights for center in centers],axis=2)
        blocking=design_circular_blocking(args.radius_meters,args.microphones,half_width_degrees=args.half_width_degrees,centers_degrees=centers,**options)
    save_beam_bank(args.output,FixedBeamBank(fixed,blocking))


if __name__=='__main__':main()
