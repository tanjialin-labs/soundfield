"""Assign two separated speech streams to enrolled voices over bounded audio intervals."""
import numpy as np
from ..contracts import audio_array, finite_scalar, positive_integer


class EnrolledPairRouter:
    def __init__(self, model_path, enrolment_audio, sample_rate_hz, *, switch_margin=.12):
        import sherpa_onnx
        positive_integer(sample_rate_hz,'sample_rate_hz')
        finite_scalar(switch_margin,'switch_margin',minimum=0,maximum=2)
        if sample_rate_hz != 16000 or len(enrolment_audio) != 2:
            raise ValueError('speaker routing requires two mono enrolments at 16 kHz')
        self.margin = switch_margin
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path),num_threads=1)
        if not config.validate():
            raise ValueError('invalid speaker embedding model')
        self._encoder = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        vectors = []
        for audio in enrolment_audio:
            samples = audio_array(audio)
            if samples.shape[1] != 1:
                raise ValueError('each enrolment recording must be mono')
            embedding = self._embedding(samples[:,0])
            if embedding is None:
                raise ValueError('enrolment needs sufficient audible speech')
            vectors.append(embedding)
        self._voices = np.stack(vectors)
        self.reset()

    def reset(self):
        self._order = np.arange(2)

    @property
    def source_order(self):
        return tuple(self._order.tolist())

    def _embedding(self, audio):
        if len(audio)<8000 or np.mean(audio**2)<1e-9:
            return None
        stream = self._encoder.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(audio,dtype=np.float32))
        stream.input_finished()
        if not self._encoder.is_ready(stream):
            return None
        embedding = np.asarray(self._encoder.compute(stream),dtype=float)
        norm = np.linalg.norm(embedding)
        return embedding/norm if np.isfinite(norm) and norm>0 else None

    def process_block(self, separated_audio):
        """Return channels in enrolment order, preserving assignments across quiet blocks."""
        block = audio_array(separated_audio)
        if block.shape[1] != 2:
            raise ValueError('expected a pair of separated channels')
        scores = np.zeros((2,2))
        for source in range(2):
            embedding = self._embedding(block[:,source])
            if embedding is not None:
                scores[:,source] = self._voices @ embedding
        difference = scores.trace() - scores[0,1] - scores[1,0]
        if abs(difference)>self.margin:
            self._order = np.array([0,1] if difference>0 else [1,0])
        return block[:,self._order].copy()
