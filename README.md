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

`FourierAnalyzer` 和 `FourierSynthesizer` 支持任意长度音频块及频谱帧批次。使用 `process_block` 连续输入，结束时调用 `flush`，新流开始前调用 `reset`。合成尾部包含补零范围，可按实际输入长度裁剪。

```sh
python examples/streaming_reconstruction.py input.wav output.wav --block-samples 1000
```

16 kHz 语音可使用低延迟 WPE 配置。它采用 512 点分析、256 样本帧移和五个预测抽头，每个抽头保留麦克风之间的相关性。预测器额外输出延迟为一帧（16 ms）。整段文件可以补偿这段延迟并保留处理尾部。

```sh
python examples/dereverberation.py input.wav output.wav --low-latency --compensate-delay
```

固定双麦录音中的两个说话人可用整段拟合模式分离。该模式从完整录音估计每个频点的解混矩阵，返回映射到参考麦克风的两路输出。逐帧输入使用 `OnlineSourceSeparator`。

```sh
python examples/source_separation.py mixture.wav separated.wav --stationary-pair
```

语音增益可按活动帧的 RMS 调节。默认目标为 −20 dBFS，最大提升为 24 dB，重建采样峰值上限为 −1 dBFS。频带语音掩码由调用方提供，低置信度与持续停顿会限制正增益。每个通道独立调节。

```sh
python examples/automatic_gain.py input.wav output.wav --speech-gains gains.npy --speech-level
```

## 功能接口

| 模块 | 主要功能 | 示例入口 |
| --- | --- | --- |
| `transform` | 整段与流式短时傅里叶变换 | `audio_reconstruction.py`、`streaming_reconstruction.py` |
| `beamforming`、`adaptive` | 固定波束、阻塞矩阵、NLMS、GSC、MVDR 与权重设计 | `array_enhancement.py`、`beam_design.py` |
| `enhancement` | 自动增益、软拐点动态范围控制、空间门控与噪声抑制 | `automatic_gain.py`、`correlation_suppression.py` |
| `separation`、`dereverb` | 在线源分离与加权预测去混响 | `source_separation.py`、`dereverberation.py` |
| `localization`、`selection` | 子空间与导向匹配定位、近场分类、通道与波束选择 | `direction_estimation.py`、`channel_routing.py` |
| `neural` | 频带特征、记忆网络、频谱增益与语音活动 | `neural_enhancement.py`、`speech_activity.py` |
| `aec` | 线性与迭代回声消除、耦合降噪、播放模式控制、远端单讲判别 | `echo_cancellation.py`、`coupled_echo_cancellation.py`、`playback_spectrum_capture.py` |
| `pipelines` | 远场、近场、会议与外部方向引导的组合处理 | `farfield_capture.py`、`nearfield_capture.py`、`conference_capture.py` |
| `acoustics`、`metrics` | 扫频测量、通道校准、响应分组与信号质量指标 | `room_measurement.py`、`channel_calibration.py`、`speech_quality.py` |
| `io`、`numerics`、`plotting` | 资源读写、系数导出、矩阵计算与显示 | `coefficient_export.py`、`complex_matrix.py`、`matrix_plot.py` |

各示例使用 `--help` 查看必需资源和参数。网络模型、归一化参数、阵列系数、录音与脉冲响应由调用方提供，包内不附带录音或模型。固定波束权重与阻塞矩阵可用 `beam_design.py` 生成；扫频刺激可用 `room_measurement.py generate` 生成。

有状态处理器按帧或按块更新，每条流使用独立实例；`reset` 保留配置并清空历史。块长度、前瞻和输出延迟以对应配置为准。数值计算使用 `float64` 和 `complex128`；音频写出时由编码参数决定量化方式。

导向匹配提供单个主方向估计，声源数必须小于麦克风数。神经接口校验模型层类型与特征尺寸。动态范围控制使用通道联动的软拐点压缩、攻击与释放时间及峰值上限。滑窗信噪比返回线性比值，PESQ 结果分别提供 raw MOS 与 MOS-LQO，扫频测量保持通道之间的相对时差。
