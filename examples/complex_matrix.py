"""Expand a small complex matrix into real arithmetic expressions."""
import argparse,json
from pathlib import Path
from soundfield.numerics.matrix_algebra import expand_complex_matrix, export_matrix_inverse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('--dimension',type=int,default=3)
    parser.add_argument('--general',action='store_true')
    parser.add_argument('--function-output',type=Path)
    args=parser.parse_args()
    if args.function_output and args.function_output.resolve()==args.output.resolve():parser.error('expression and function outputs must differ')
    result=expand_complex_matrix(args.dimension,hermitian=not args.general)
    args.output.write_text(json.dumps(dict(matrix=str(result.matrix),determinant=str(result.determinant),adjugate=str(result.adjugate),
        common_terms=[(str(symbol),str(expression)) for symbol,expression in result.common_terms],expressions=[str(value) for value in result.reduced_expressions]),indent=2)+'\n')
    if args.function_output:export_matrix_inverse(args.function_output,args.dimension,hermitian=not args.general)


if __name__=='__main__':main()
