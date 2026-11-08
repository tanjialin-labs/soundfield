# SoundField

SoundField 提供音频与麦克风阵列信号处理功能，包括时频分析、波束形成、源分离、去混响、方向估计、语音增强、声学回声消除、房间响应测量和信号质量评估。

要求 Python 3.11 或更新版本。

## 安装

```sh
python -m pip install .
```

权重优化、符号计算、语音质量指标与绘图可按需安装：

```sh
python -m pip install '.[design,evaluation,plotting]'
```

## 使用

```python
from soundfield import FourierConfig, ShortTimeFourierTransform
from soundfield.io import read_audio, write_audio

sample_rate_hz, audio = read_audio("input.wav")
transform = ShortTimeFourierTransform(FourierConfig(
    frame_length_samples=512,
    hop_length_samples=256,
    window="sqrt_hann",
))
spectrum = transform.analyze(audio)
reconstructed = transform.synthesize(spectrum, length_samples=len(audio))
write_audio("output.wav", reconstructed, sample_rate_hz)
```

音频数组采用 `(samples, channels)`，频谱采用 `(bins, frames, channels)`，单通道保留通道轴。配置、模型、阵列系数和输出路径均由调用方明确提供。

`examples/` 提供各功能的命令行入口：

```sh
python examples/audio_reconstruction.py input.wav output.wav
python examples/echo_cancellation.py microphone.wav reference.wav cleaned.wav
python examples/room_measurement.py generate stimulus
```
