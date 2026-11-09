# SoundField

English · [简体中文](README.zh-CN.md)

Release **1.0.0**

SoundField is a Python library for speech and microphone-array signal processing. Build a streaming echo canceller for a call, separate two speakers in a recording, focus capture on a direction, or measure a room with a swept sine. The NumPy and SciPy core runs on a CPU.

## Install

From the repository root, with Python 3.11 or later.

```sh
python -m pip install .
```

Install extras for the features you use. For example, `python -m pip install '.[inference]'` adds ONNX Runtime for recurrent enhancement. The `recognition` extra provides keyword search and speech verification, `identification` adds speaker identification, `design` adds beam-weight optimization, `evaluation` adds perceptual metrics, and `plotting` adds visualization.

## Analyze and reconstruct audio

This example generates a one-second tone. It needs no recording or model.

```python
import numpy as np
from soundfield import FourierConfig, ShortTimeFourierTransform

rate = 16_000
time = np.arange(rate) / rate
audio = (0.1 * np.sin(2 * np.pi * 440 * time))[:, None]

transform = ShortTimeFourierTransform(FourierConfig(
    frame_length_samples=512, hop_length_samples=256, window="sqrt_hann"
))
spectrum = transform.analyze(audio)
output = transform.synthesize(spectrum, length_samples=len(audio))
assert output.shape == audio.shape
```

Audio is shaped `(samples, channels)`, a spectrum is `(bins, frames, channels)`, and one frame is `(bins, channels)`. Mono retains its channel axis. Core calculations use `float64` waveforms and `complex128` spectra. For file input, `soundfield.io.read_audio` returns the sample rate and audio array.

## Choose a processing path

| Task | Inputs and behavior | Example |
| --- | --- | --- |
| Cancel acoustic echo | Microphones and synchronous playback. The streaming interface accepts timestamped blocks and returns the remaining samples on flush. | [Streaming echo cancellation](examples/streaming_echo.py) |
| Separate two speakers | The stationary-pair mode fits an entire two-microphone recording and produces two source images. An aligned 16 kHz playback reference enables echo cancellation before separation. | [Source separation](examples/source_separation.py) |
| Reduce reverberation | The 16 kHz low-latency WPE configuration uses a 512-point transform and five prediction taps. Its extra prediction delay is 16 ms. File processing can compensate this delay. | [Dereverberation](examples/dereverberation.py) |
| Focus on a direction | Supply array geometry, beam resources and a target direction. Fixed and adaptive capture, direction estimation and time-varying target selection have separate entry points. | [Directed capture](examples/directed_capture.py) · [Direction estimation](examples/direction_estimation.py) · [Beam design](examples/beam_design.py) |
| Enhance speech | Statistical and recurrent noise suppression, level control and per-channel speech activity. Model-based paths accept an explicit model file. | [Background suppression](examples/background_noise.py) · [Activity detection](examples/speech_activity.py) · [Automatic gain](examples/automatic_gain.py) |
| Detect keywords | A primary keyword search accepts a wider phonetic search and independent transcript confirmation. Events carry the sample index at which the decision is available. | [Keyword capture](examples/keyword_capture.py) |
| Measure and evaluate | Generate a sweep, estimate room responses, calibrate microphone channels and evaluate speech quality. | [Room measurement](examples/room_measurement.py) · [Channel calibration](examples/channel_calibration.py) · [Speech metrics](examples/speech_quality.py) |

Run the examples from the repository root. Each command describes its resources, units and options with `--help`.

```sh
python examples/streaming_echo.py --help
python examples/source_separation.py --help
python examples/room_measurement.py --help
```

For a moving target, supply a direction index for each frame. In two-microphone capture, `--selective-attenuation` can preserve frequency bands distinct from a persistent interferer when the spatial gate closes. The WPE example accepts `--protect-short-decays` with `--low-latency --compensate-delay` for complete 16 kHz two-microphone recordings.

For activity decisions after enhancement, `speech_activity.py --decisions --energy-range-db 31` applies per-channel energy evidence aligned with the detector output. For one recognition utterance, `background_noise.py --recognition-segment --recurrent-model MODEL` retains part of the microphone signal and bounds the level sent to the recognizer.

Keyword capture uses caller-supplied search vocabularies, transducer weights and a Mandarin CTC model. Its phrase map is a JSON object. `distractors` maps each non-trigger label in the contextual vocabulary to the trigger it resembles. `competing_text` maps a trigger to a list of competing transcriptions. Optional `support_text` fragments can corroborate a primary hit when a transcript is partial. The example uses English word verification for ASCII phrases and Mandarin character verification for the other phrases.

## State and resources

Create one stateful processor per stream. Use its documented frame or block size, call `flush` where provided to drain the tail, and `reset` before starting an unrelated recording. Whole-recording separation needs the entire input before returning output. Supply recordings, model weights, microphone geometry and beam resources explicitly. The [examples](examples/) cover far-field calls, near-field capture, conferencing, beamforming, calibration and measurement.

## License

SoundField is a personal project by Tan Jialin. Its original code and documentation, including release 1.0.0, are available under the [MIT License](LICENSE). Third-party components retain their own [notices and terms](licenses/). The [STOI notice](licenses/short-time-intelligibility.txt) specifies non-commercial use. External models and optional dependencies have their own licenses.
