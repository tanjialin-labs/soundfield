"""Stateful complex spectral enhancement with a recurrent ONNX network."""
from pathlib import Path

import numpy as np

from ..contracts import audio_array, finite_scalar


class RecurrentSpectrumEnhancer:
    """One 16 kHz mono stream with explicit per-frame network memory."""

    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(model_path)), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = {v.name: v for v in self.session.get_inputs()}
        outputs = {v.name: v for v in self.session.get_outputs()}
        self._caches = ('conv_cache', 'tra_cache', 'inter_cache')
        shape = [1, 257, 1, 2]
        if (set(inputs) != {'mix', *self._caches} or inputs['mix'].shape != shape
                or set(outputs) != {'enh', *(name+'_out' for name in self._caches)} or outputs['enh'].shape != shape):
            raise ValueError('model must implement the 257-bin recurrent complex spectrum interface')
        if any(v.type != 'tensor(float)' for v in (*inputs.values(), *outputs.values())):
            raise ValueError('model inputs and outputs must use float32')
        self._shapes = {name: inputs[name].shape for name in self._caches}
        if any(outputs[name+'_out'].shape != dims or not all(isinstance(d,int) and d>0 for d in dims)
               for name,dims in self._shapes.items()):
            raise ValueError('recurrent inputs and outputs must have matching fixed sizes')
        self.reset()

    def reset(self):
        self._state = {name: np.zeros(dims, np.float32) for name,dims in self._shapes.items()}

    def process_frame(self, spectrum):
        x = np.asarray(spectrum)
        if x.shape != (257,) or not np.isfinite(x).all():
            raise ValueError('expected one finite 257-bin spectrum')
        mix = np.stack((x.real, x.imag), axis=-1).astype(np.float32).reshape(1,257,1,2)
        names = ['enh', *(name+'_out' for name in self._caches)]
        enhanced, *state = self.session.run(names, dict(self._state, mix=mix))
        self._state = dict(zip(self._caches, state))
        return enhanced[0,:,0,0] + 1j*enhanced[0,:,0,1]

    def enhance_audio(self, audio, sample_rate_hz, *, phase_preserving_floor=None):
        """Enhance a full mono recording using centered 512-point periodic square-root Hann windows."""
        x = audio_array(audio)
        if sample_rate_hz != 16000 or x.shape[1] != 1:
            raise ValueError('recurrent enhancement requires mono audio at 16000 Hz')
        if phase_preserving_floor is not None:
            finite_scalar(phase_preserving_floor, "phase_preserving_floor", minimum=0, maximum=1)
        self.reset()
        if not len(x):
            return x.copy()
        hop = 256
        window = np.sqrt(.5-.5*np.cos(2*np.pi*np.arange(512)/512))
        padded = np.pad(x[:,0], (hop, hop + (-len(x)) % hop))
        output = np.zeros_like(padded)
        for start in range(0,len(padded)-511,hop):
            frame = np.fft.rfft(padded[start:start+512]*window)
            enhanced = self.process_frame(frame)
            if phase_preserving_floor is not None:
                gain = np.clip(abs(enhanced) / np.maximum(abs(frame), 1e-12), phase_preserving_floor, 1.)
                enhanced = frame * gain
            output[start:start+512] += np.fft.irfft(enhanced)*window
        return output[hop:hop+len(x),None]
