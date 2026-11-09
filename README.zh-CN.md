# SoundField

[English](README.md) · 简体中文

正式版本 **1.0.0**

SoundField 是用于语音和麦克风阵列信号处理的 Python 库。它可以消除通话回声、分离双人录音、按方向拾音，也可以用扫频信号测量房间响应。核心算法依赖 NumPy 和 SciPy，可在 CPU 上运行。

## 安装

使用 Python 3.11 或更高版本，在仓库根目录运行

```sh
python -m pip install .
```

按功能安装可选依赖。例如，`python -m pip install '.[inference]'` 会为循环网络增强安装 ONNX Runtime。`recognition` 用于关键词搜索和识别复核，`identification` 用于说话人身份识别，`design` 用于波束权重优化，`evaluation` 用于感知指标，`plotting` 用于绘图。

## 分析与重建音频

以下示例自行生成一秒钟的正弦波，可直接运行。

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

音频数组形状为 `(samples, channels)`，完整频谱为 `(bins, frames, channels)`，单帧为 `(bins, channels)`。单通道保留通道轴。核心计算采用 `float64` 波形和 `complex128` 频谱。读取文件时，`soundfield.io.read_audio` 返回采样率和音频数组。

## 选择处理方式

| 任务 | 输入与行为 | 示例 |
| --- | --- | --- |
| 消除回声 | 输入麦克风信号和同步播放参考。流式接口接收带样本时间戳的块，调用 flush 输出剩余采样。 | [流式回声消除](examples/streaming_echo.py) |
| 分离两位说话人 | 整段模式对完整双麦录音进行拟合，输出两路声像。传入对齐的 16 kHz 播放参考可先消除回声再分离。 | [声源分离](examples/source_separation.py) |
| 降低混响 | 16 kHz 低延迟 WPE 配置使用 512 点变换和五个预测抽头，预测额外延迟为 16 ms。文件处理可补偿这段延迟。 | [去混响](examples/dereverberation.py) |
| 定向拾音 | 调用方传入阵列几何、波束资源和目标方向，可分别使用固定或自适应拾音、方位估计和随时间变化的目标选择。 | [定向拾音](examples/directed_capture.py) · [方位估计](examples/direction_estimation.py) · [波束设计](examples/beam_design.py) |
| 增强语音 | 可使用统计或循环网络降噪、语音增益控制和逐通道语音活动检测。模型路径由调用方提供。 | [背景降噪](examples/background_noise.py) · [活动检测](examples/speech_activity.py) · [自动增益](examples/automatic_gain.py) |
| 关键词检测 | 结合主搜索、扩展发音词表和独立转写复核，事件标记为判决可用时的样本位置。 | [关键词拾取](examples/keyword_capture.py) |
| 测量与评估 | 可生成扫频信号、估计房间响应、校准麦克风通道并评估语音质量。 | [房间测量](examples/room_measurement.py) · [通道校准](examples/channel_calibration.py) · [语音指标](examples/speech_quality.py) |

在仓库根目录运行示例。各命令通过 `--help` 提供资源、单位和参数说明。

```sh
python examples/streaming_echo.py --help
python examples/source_separation.py --help
python examples/room_measurement.py --help
```

跟随移动目标时，应为每帧提供目标方向索引。双麦拾音可启用 `--selective-attenuation`，在空间门关闭时保留具有不同干扰相位的频带。对于完整的 16 kHz 双麦录音，WPE 示例支持将 `--protect-short-decays` 与 `--low-latency --compensate-delay` 一起使用。

对增强后的输入，`speech_activity.py --decisions --energy-range-db 31` 可将逐通道能量证据与检测判决对齐。对单个识别片段，`background_noise.py --recognition-segment --recurrent-model MODEL` 可保留部分麦克风信号并限制送入识别器的电平。

关键词示例由调用方提供发音词表、搜索网络权重与普通话 CTC 模型。短语配置使用 JSON。`distractors` 把扩展词表内的干扰标签映射到容易混淆的目标词，`competing_text` 把目标词映射到一组竞争转写。可选的 `support_text` 提供部分转写中能支持主命中的识别片段。示例对 ASCII 短语使用英语词级复核，对其余短语使用普通话字符级复核。

## 状态与资源

每条流使用独立的处理器，按接口要求输入帧或音频块。提供 `flush` 的接口用它输出尾部，开始下一条独立录音前调用 `reset`。整段分离需要收齐输入后才返回音频。录音、模型权重、麦克风几何和波束资源均由调用方显式提供。[示例目录](examples/) 还包含远场通话、近场拾音、会议、波束形成、校准和声学测量入口。

## 许可

SoundField 是 Tan Jialin 创建和维护的个人项目。项目自有代码和文档（含 1.0.0 版）采用 [MIT 许可证](LICENSE)。第三方组件适用[各自的声明和条款](licenses/)，其中 [STOI 的声明](licenses/short-time-intelligibility.txt) 限定为非商业使用。外部模型和可选依赖的许可另行适用。
