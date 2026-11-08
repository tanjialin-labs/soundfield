"""Interleaved coefficient export for numeric text and little-endian float arrays."""

from pathlib import Path
import re
import numpy as np


def interleave_complex_coefficients(values):
    """Interleave real/imaginary values, preserving the caller's axis order."""
    array = np.asarray(values, dtype=complex)
    if not np.isfinite(array).all():
        raise ValueError('coefficients must be finite')
    return np.stack((array.real, array.imag), axis=-1).reshape(-1)


def export_beam_coefficients(fixed_path, blocking_path, bank, *, format='binary',
                             frequency_major=True, blocking_frequency_major=None,
                             blocking_reduction='all'):
    """Export interleaved coefficients with explicit axis and reference choices.

    Frequency-major order is beam/bin/reference/microphone; otherwise it is
    beam/reference/microphone/bin. Fixed beams omit the reference axis.
    blocking_frequency_major can override the blocking stream independently.
    """
    if Path(fixed_path).resolve() == Path(blocking_path).resolve():
        raise ValueError('fixed and blocking outputs must have distinct paths')
    if blocking_reduction not in ('all', 'first', 'mean'):
        raise ValueError('blocking_reduction must be all, first or mean')
    blocking = bank.blocking_matrix
    if blocking_reduction == 'first':
        blocking = blocking[:, :, :1]
    elif blocking_reduction == 'mean':
        blocking = blocking.mean(axis=2, keepdims=True)
    fixed = bank.fixed_beam_weights.transpose(2, 0, 1) if frequency_major else bank.fixed_beam_weights.transpose(2, 1, 0)
    blocking_order = frequency_major if blocking_frequency_major is None else blocking_frequency_major
    blocking = blocking.transpose(3, 0, 2, 1) if blocking_order else blocking.transpose(3, 2, 1, 0)
    if format not in ('binary', 'text'):
        raise ValueError('format must be binary or text')
    for path, values in ((fixed_path, fixed), (blocking_path, blocking)):
        pairs = interleave_complex_coefficients(values)
        if format == 'binary':
            with open(path, 'wb') as stream:
                stream.write(pairs.astype('<f4').tobytes())
        else:
            np.savetxt(path, pairs.reshape(-1, 2*bank.microphone_count), fmt='%.16f')


def numeric_text_to_float_binary(input_path, output_path):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('output must differ from input')
    text = Path(input_path).read_text(encoding='utf-8').replace(',', ' ')
    values = np.array([float(item) for item in text.split()], dtype='<f4')
    if not np.isfinite(values).all():
        raise ValueError('text coefficients must be finite')
    with open(output_path, 'wb') as stream:
        stream.write(values.tobytes())


def export_byte_array(path, data, *, symbol='beam_coefficients'):
    """Write a C byte array and its byte count without invoking a shell."""
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol):
        raise ValueError('symbol must be a valid C identifier')
    values = bytes(data)
    rows = [', '.join(f'0x{value:02x}' for value in values[start:start+12]) for start in range(0, len(values), 12)]
    body = ',\n  '.join(rows) if rows else '0'
    Path(path).write_text(f'const unsigned char {symbol}[] = {{\n  {body}\n}};\nconst unsigned int {symbol}_length = {len(values)};\n', encoding='utf-8')
