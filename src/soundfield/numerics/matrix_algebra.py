"""Numeric cofactors and symbolic expressions for small complex matrices."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import positive_integer


def adjugate_and_determinant(matrix):
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values) or not np.isfinite(values).all():
        raise ValueError('matrix must be finite, nonempty and square')
    count = len(values)
    adjugate = np.empty_like(values)
    for row in range(count):
        for column in range(count):
            minor = np.delete(np.delete(values, column, axis=0), row, axis=1)
            adjugate[row, column] = (-1)**(row+column)*np.linalg.det(minor)
    determinant = np.sum(values[0]*adjugate[:, 0])
    return adjugate, determinant


@dataclass(frozen=True)
class SymbolicMatrixExpansion:
    matrix: object
    determinant: object
    adjugate: object
    common_terms: tuple
    reduced_expressions: tuple


def expand_complex_matrix(dimension, *, hermitian=True):
    """Construct explicit real/imaginary symbols and eliminate common expressions."""
    positive_integer(dimension, 'dimension')
    if dimension > 6:
        raise ValueError('symbolic expansion is limited to matrices of order six')
    try:
        import sympy
    except ImportError as exception:
        raise ImportError('Symbolic expansion requires soundfield[design]') from exception
    matrix = sympy.zeros(dimension)
    for row in range(dimension):
        for column in range(row if hermitian else 0, dimension):
            real = sympy.Symbol(f'entry_{row}_{column}_real', real=True)
            imaginary = 0 if hermitian and row == column else sympy.Symbol(f'entry_{row}_{column}_imag', real=True)
            value = real+sympy.I*imaginary
            matrix[row, column] = value
            if hermitian:
                matrix[column, row] = sympy.conjugate(value)
    determinant = sympy.expand(matrix.det(method='berkowitz'))
    adjugate = matrix.adjugate(method='berkowitz').applyfunc(sympy.expand)
    expressions = [sympy.re(determinant).expand(complex=True), sympy.im(determinant).expand(complex=True)]
    for value in adjugate:
        expressions.extend((sympy.re(value).expand(complex=True), sympy.im(value).expand(complex=True)))
    common, reduced = sympy.cse(expressions, symbols=sympy.numbered_symbols('cofactor_'))
    return SymbolicMatrixExpansion(matrix, determinant, adjugate, tuple(common), tuple(reduced))


def export_matrix_inverse(path, dimension, *, hermitian=False, selected_columns=None):
    """Write a standalone NumPy function using real arithmetic and common terms."""
    expansion = expand_complex_matrix(dimension, hermitian=hermitian)
    import sympy
    columns = tuple(range(dimension)) if selected_columns is None else tuple(selected_columns)
    if not columns or len(set(columns)) != len(columns) or any(not isinstance(column, int) or not 0 <= column < dimension for column in columns):
        raise ValueError('selected_columns must identify distinct matrix columns')
    expressions = [sympy.re(expansion.determinant).expand(complex=True), sympy.im(expansion.determinant).expand(complex=True)]
    for row in range(dimension):
        for column in columns:
            value = expansion.adjugate[row, column]
            expressions.extend((sympy.re(value).expand(complex=True), sympy.im(value).expand(complex=True)))
    common, reduced = sympy.cse(expressions, symbols=sympy.numbered_symbols('cofactor_'))
    lines = ['"""Explicit complex matrix inversion."""', '', 'import numpy as np', '',
             'def inverse_complex_matrix(matrix):', '    values = np.asarray(matrix, dtype=complex)',
             f'    if values.shape != ({dimension}, {dimension}) or not np.isfinite(values).all():',
             "        raise ValueError('matrix has invalid dimensions or nonfinite entries')"]
    if hermitian:
        lines.extend(["    if not np.allclose(values, values.conj().T):", "        raise ValueError('matrix must be Hermitian')"])
    for symbol in sorted(expansion.matrix.free_symbols, key=str):
        _, row, column, component = str(symbol).split('_')
        lines.append(f'    {symbol} = values[{row}, {column}].'+('real' if component == 'real' else 'imag'))
    lines.extend(f'    {symbol} = {sympy.pycode(value)}' for symbol, value in common)
    lines.append(f'    determinant = complex({sympy.pycode(reduced[0])}, {sympy.pycode(reduced[1])})')
    lines.extend(["    if determinant == 0:", "        raise np.linalg.LinAlgError('matrix is singular')", f'    output = np.empty(({dimension}, {len(columns)}), complex)'])
    for index in range(dimension*len(columns)):
        row, column = divmod(index, len(columns))
        real, imaginary = reduced[2+2*index:4+2*index]
        lines.append(f'    output[{row}, {column}] = complex({sympy.pycode(real)}, {sympy.pycode(imaginary)}) / determinant')
    lines.append('    return output, determinant')
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')
