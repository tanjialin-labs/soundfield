"""Save an annotated figure from a numeric NPY matrix."""

import argparse
from pathlib import Path
import numpy as np
from soundfield.plotting import plot_annotated_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error('output must differ from input')
    figure = plot_annotated_matrix(np.load(arguments.input, allow_pickle=False))
    figure.savefig(arguments.output, dpi=150)


if __name__ == '__main__':
    main()
