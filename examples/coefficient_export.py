"""Export a beam bank with explicit binary ordering and blocking reduction."""
import argparse
from pathlib import Path
from soundfield.io import load_beam_bank
from soundfield.io.coefficient_export import export_beam_coefficients,export_byte_array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('beam_bank',type=Path)
    parser.add_argument('fixed_output',type=Path)
    parser.add_argument('blocking_output',type=Path)
    parser.add_argument('--format',choices=('binary','text'),default='binary')
    parser.add_argument('--microphone-major',action='store_true')
    parser.add_argument('--blocking-order',choices=('frequency','microphone'))
    parser.add_argument('--blocking-reduction',choices=('all','first','mean'),default='all')
    parser.add_argument('--byte-header',type=Path)
    args=parser.parse_args()
    paths=[p.resolve() for p in (args.beam_bank,args.fixed_output,args.blocking_output,args.byte_header) if p]
    if len(set(paths))!=len(paths):parser.error('all resource paths must be distinct')
    bank=load_beam_bank(args.beam_bank)
    export_beam_coefficients(args.fixed_output,args.blocking_output,bank,format=args.format,
        frequency_major=not args.microphone_major,blocking_reduction=args.blocking_reduction,
        blocking_frequency_major=None if args.blocking_order is None else args.blocking_order=='frequency')
    if args.byte_header:export_byte_array(args.byte_header,args.fixed_output.read_bytes())


if __name__=='__main__':main()
