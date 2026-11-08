"""Quantized feedforward memory networks with explicit per-stream state."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.special import expit

from ..contracts import real_array, positive_integer, nonnegative_integer
from .band_features import round_away_from_zero


@dataclass(frozen=True)
class MemoryLayer:
    input_weights: np.ndarray
    bias: np.ndarray
    projection_weights: np.ndarray
    temporal_weights: np.ndarray
    activation_limit: float | None = 2.0


@dataclass(frozen=True)
class DenseLayer:
    input_weights: np.ndarray
    bias: np.ndarray
    activation: str = 'sigmoid'


class MemoryNetwork:
    """Weights use a binary scale; features, biases and outputs use Q24."""

    def __init__(self, memory_layers, dense_layers, *, weight_fraction_bits=6):
        nonnegative_integer(weight_fraction_bits, 'weight_fraction_bits')
        self.weight_fraction_bits = weight_fraction_bits
        self.memory_layers = tuple(memory_layers)
        self.dense_layers = tuple(dense_layers)
        if not self.memory_layers and not self.dense_layers:
            raise ValueError('network must contain at least one layer')
        previous = None
        for layer in (*self.memory_layers, *self.dense_layers):
            weights = real_array(layer.input_weights, 'input_weights', 2)
            bias = real_array(layer.bias, 'bias', 1)
            if not all(weights.shape) or bias.shape != (weights.shape[0],):
                raise ValueError('layer weights and bias have incompatible shapes')
            if previous is not None and weights.shape[1] != previous:
                raise ValueError('adjacent layer dimensions do not agree')
            if isinstance(layer, MemoryLayer):
                projection = real_array(layer.projection_weights, 'projection_weights', 2)
                temporal = real_array(layer.temporal_weights, 'temporal_weights', 2)
                if (projection.shape[1] != weights.shape[0] or temporal.shape[1] != projection.shape[0]
                        or not all(temporal.shape) or not all(projection.shape)):
                    raise ValueError('memory projection and temporal dimensions do not agree')
                if layer.activation_limit is not None and layer.activation_limit <= 0:
                    raise ValueError('activation_limit must be positive')
                previous = projection.shape[0]
            else:
                if layer.activation not in ('sigmoid', 'tanh', 'relu', 'linear'):
                    raise ValueError('unsupported dense activation')
                previous = weights.shape[0]
        self.input_count = (*self.memory_layers, *self.dense_layers)[0].input_weights.shape[1]
        self.output_count = previous

    def create_state(self):
        return MemoryNetworkState(self)

    @classmethod
    def load_text(cls, path, *, activation_limit=2.0, weight_fraction_bits=6):
        """Read a text network containing Dfsmn and affine components."""
        text = Path(path).read_text(encoding='utf-8')
        components = re.split(r'(<Dfsmn>|<AffineTransformRelu>|<AffineTransform>)', text)
        if len(components) < 3:
            raise ValueError('no supported network components found')
        memories, dense = [], []
        quantize = lambda value: np.clip(round_away_from_zero(value * 2**weight_fraction_bits), -128, 127)
        for tag, body in zip(components[1::2], components[2::2]):
            arrays = []
            for block in re.findall(r'\[([^\]]*)\]', body, re.S):
                rows = [line.split() for line in block.strip().splitlines() if line.strip()]
                try:
                    arrays.append(np.array(rows, dtype=float))
                except ValueError as error:
                    raise ValueError('network matrix must have rectangular numeric rows') from error
            if tag == '<Dfsmn>':
                if dense or len(arrays) != 4:
                    raise ValueError('memory components require four matrices before affine layers')
                temporal, weights, bias, projection = arrays
                memories.append(MemoryLayer(quantize(weights), round_away_from_zero(bias.ravel()*2**24),
                                            quantize(projection), quantize(temporal), activation_limit))
            else:
                if len(arrays) != 2:
                    raise ValueError('affine components require weights and bias')
                dense.append(DenseLayer(quantize(arrays[0]), round_away_from_zero(arrays[1].ravel()*2**24),
                                        'relu' if tag == '<AffineTransformRelu>' else 'sigmoid'))
        return cls(memories, dense, weight_fraction_bits=weight_fraction_bits)


class MemoryNetworkState:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.histories = [np.zeros_like(layer.temporal_weights) for layer in self.model.memory_layers]
        self.layer_outputs = ()

    def process_frame(self, features):
        values = real_array(features, 'features', 1)
        if values.shape != (self.model.input_count,):
            raise ValueError('feature count does not match the network input')
        divisor = 2**self.model.weight_fraction_bits
        outputs = []
        for layer, history in zip(self.model.memory_layers, self.histories):
            hidden = np.maximum(layer.input_weights @ values / divisor + layer.bias, 0)
            if layer.activation_limit is not None:
                hidden = np.minimum(hidden, layer.activation_limit * 2**24)
            projected = layer.projection_weights @ hidden / divisor
            history[:-1] = history[1:]
            history[-1] = projected
            values = np.sum(history * layer.temporal_weights, axis=0) / divisor + projected
            outputs.append(values.copy())
        for layer in self.model.dense_layers:
            values = round_away_from_zero(layer.input_weights @ values / divisor) + layer.bias
            if layer.activation == 'sigmoid':
                values = round_away_from_zero(expit(values / 2**24) * 2**24)
            elif layer.activation == 'tanh':
                values = round_away_from_zero(np.tanh(values / 2**24) * 2**24)
            elif layer.activation == 'relu':
                values = np.maximum(values, 0)
            outputs.append(values.copy())
        self.layer_outputs = tuple(outputs)
        return values.copy()
