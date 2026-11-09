"""Calibrated DNSMOS P.835 predictions for speech, background and overall quality."""
from dataclasses import dataclass
from math import gcd
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from ..contracts import audio_array, positive_integer


@dataclass(frozen=True)
class PredictedSpeechQuality:
    speech: float
    background: float
    overall: float
    windows: int


class SpeechBackgroundQualityEstimator:
    """Score normalized mono waveforms using a standard non-personalized P.835 model."""

    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(Path(model_path)),sess_options=options,providers=['CPUExecutionProvider'])
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != 'input_1' or inputs[0].type != 'tensor(float)':
            raise ValueError('expected a waveform-input DNSMOS P.835 model')

    def score(self, audio, sample_rate_hz):
        samples = audio_array(audio)
        positive_integer(sample_rate_hz,'sample_rate_hz')
        if not len(samples) or samples.shape[1]!=1:
            raise ValueError('quality prediction requires nonempty mono audio')
        x = samples[:,0]
        rate = 16000
        if sample_rate_hz != rate:
            divisor = gcd(sample_rate_hz,rate)
            x = resample_poly(x,rate//divisor,sample_rate_hz//divisor)
        width = 144160
        if len(x) < width:
            repeats = 1 << ((width-1)//len(x)).bit_length()
            x = np.tile(x,repeats)
        windows = int(len(x)//rate-9.01)+1
        coefficients = np.array([[-.08397278,1.22083953,.00524390],
                                 [-.13166888,1.60915514,-.39604546],
                                 [-.06766283,1.11546468,.04602535]])
        estimates = []
        for start in range(0,windows*rate,rate):
            block = x[start:start+width].astype(np.float32)[None,:]
            raw = self._session.run(None,{'input_1':block})[0]
            if raw.shape != (1,3) or not np.isfinite(raw).all():
                raise ValueError('expected three finite P.835 predictions')
            scores = raw[0].astype(float)
            estimates.append((coefficients[:,0]*scores+coefficients[:,1])*scores+coefficients[:,2])
        return PredictedSpeechQuality(*map(float,np.mean(estimates,axis=0)),len(estimates))
