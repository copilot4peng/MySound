# dev_model 本地模型适配与验证

验证日期：2026-10-01。以下记录对应当前工作目录的模型文件，不是通用性能基准。

## 模型选择

在“配置 → 模型”中将扫描根目录设为项目内的 `dev_model`，扫描后点击具体模型并保存。扫描不会替用户更改正在使用的模型。

| 本地目录 | 界面模型类型 | 选择方式 |
| --- | --- | --- |
| `Whisper-large-v3-turbo` | OpenAI Whisper | 目录或 `large-v3-turbo.pt` |
| `Whisper-large-v3` | OpenAI Whisper | 目录或 `large-v3.pt` |
| `Qwen3-ASR-0.6B` | Qwen3-ASR | 完整目录 |
| `Qwen3-ASR-1.7B` | Qwen3-ASR | 完整目录，支持分片权重 |
| `netease-youdao` | Confucius4-R2T2 | 完整目录 |
| `sherpa-onnx-whisper-medium` | Sherpa-ONNX Whisper | 目录默认 INT8；明确选择 `medium-encoder.onnx` 或 `medium-decoder.onnx` 使用 FP32 |

ONNX 使用 `sherpa-onnx==1.13.8` 的 CPU 后端，`auto` 同样使用 CPU；显式 CUDA 会收到说明。encoder、decoder 必须是同一型号和同一精度，tokens 文件必须齐备。该后端内部最多按 29 秒处理，避免上游单段截断；字幕时间是这些输入片段的范围。

旧的 `Qwen3-ASR/Qwen3-ASR-1.7B.zip` 仍是压缩包，扫描会标为不可直接加载。已经解压到同级 `Qwen3-ASR-1.7B` 的目录可以使用，无需再次解压。

## Qwen3-ASR-0.6B 的配套文件

原目录只有 `config.json`、`tokenizer.json`、`model.safetensors`，缺少官方处理器配置。这次从 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B/files) 的 `3b885f72b1733a6a50dc17a597fb4135c3d656a0` 版本补入六个文件：

- `preprocessor_config.json`
- `tokenizer_config.json`
- `chat_template.json`
- `generation_config.json`
- `merges.txt`
- `vocab.json`

补入文件的大小与 SHA-256 均与官方文件清单一致；原有 `config.json`、`model.safetensors` 也通过官方 SHA-256 核验。原有文件未覆盖。程序日常转写仍完全离线，不会自动联网修补不完整模型。

## 真实推理检查

硬件为 Intel Core i7-9700K、约 23 GiB 系统内存。使用 Python 3.13.7，CPU 推理线程限制为 4。测试音频为模型自带的 `Whisper-large-v3-turbo/example/asr_example.wav`，16 kHz 单声道、5.55 秒。各模型在独立进程内依次加载与转写。

| 模型 | 加载耗时 | 转写耗时 | 进程峰值 RSS |
| --- | ---: | ---: | ---: |
| Whisper large-v3-turbo | 6.87 秒 | 6.87 秒 | 5.06 GiB |
| Qwen3-ASR-0.6B | 6.51 秒 | 1.72 秒 | 6.20 GiB |
| Qwen3-ASR-1.7B | 7.72 秒 | 5.02 秒 | 12.70 GiB |
| Sherpa Whisper medium INT8 | 4.12 秒 | 7.69 秒 | 2.12 GiB |
| Sherpa Whisper medium FP32 | 6.12 秒 | 9.18 秒 | 4.66 GiB |

RSS 包含模型加载峰值与 Python 依赖，不包含其他进程；此单条短音频的测量不能当成最低内存要求，也不能推断长音频、麦克风延迟或 GPU 性能。应用中的资源卡继续使用留有余量的规划范围。

Turbo 和两种 Qwen 均输出“欢迎大家来体验达摩院推出的语音识别模型”（标点略有不同）。当前 ONNX medium 文件在此样例输出“欢迎大家来体打院推出的音别模型”，存在漏字；FP32 也有同样现象。词表已逐项与官方 Whisper multilingual 词表核对，尾部填充 300 与默认值的对照也未改善，因此保留官方默认参数，没有通过替换文字掩盖识别结果。

另外，已实际运行 ONNX 的 FFmpeg 转码 → Silero-VAD → 后台转写 → 临时历史库保存 → SRT 导出，流程成功。本次未重新对旧 Whisper large-v3 / Confucius 权重运行推理，也未进行新模型的 CUDA 实测。

## 回归结果

- 159 项非桌面测试通过，覆盖路径识别、精度配对、模型切换、语言校验、配置兼容、资源建议及原有转写流程。
- 10 项 Ubuntu 桌面回归通过，覆盖八种主题、48 号字体、全部型号浏览、ONNX 扫描和选择、路径持久化、中英文切换、历史刷新及窗口缩放。
- 新增型号使 48 号字体下的长列表触发 X11 绘图分配错误，已改为下拉选择与左右翻页浏览资源卡；全部型号仍可访问。动态下拉销毁后的缩放回调也已清理。
- `main.py --check`、`pip check`、源码编译与 `build.sh` 语法检查通过。打包依赖已更新；本次未生成完整 PyInstaller 分发包。
