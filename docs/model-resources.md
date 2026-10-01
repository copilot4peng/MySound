# 模型资源说明与来源

核实日期：2026-10-01。`core/model_catalog.py` 区分官方参考数据、本程序的资源规划估算和实际文件大小；估算不作为加载限制，也不能保证运行成功或实时速度。

## 官方资料

[OpenAI Whisper 官方模型表](https://github.com/openai/whisper#available-models-and-languages) 给出的数据如下。显存单位保留原文的 GB；相对速度是在 A100 上转写英语、以 large 为 1 的参考，不是本机 CPU 或 GPU 测速。

| Whisper | 参数 | 官方约需显存 GB | 官方相对速度 |
| --- | ---: | ---: | ---: |
| tiny | 39 M | 1 | 10 |
| base | 74 M | 1 | 7 |
| small | 244 M | 2 | 4 |
| medium | 769 M | 5 | 2 |
| large | 1550 M | 10 | 1 |
| turbo | 809 M | 6 | 8 |

[Qwen3-ASR-0.6B 官方模型页](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 和 [Qwen3-ASR-1.7B 官方模型页](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 分别将检查点总参数显示为约 0.9B、2B；因此系列名称中的 0.6B/1.7B 不应直接用作完整检查点显存计算。页面没有给出本应用单任务 Transformers 推理的固定最低显存；官方高并发吞吐不能推导本机交互延迟。官方文档提到 96 GB RAM 的位置是在编译 FlashAttention 时限制构建并发，不是模型运行最低要求。

[Confucius4-R2T2 官方模型页](https://huggingface.co/netease-youdao/Confucius4-R2T2) 将检查点标为约 2B 参数。其流式低延迟结果依赖对应流式推理环境，本程序用 Transformers 重解码短句，不承诺相同延迟。未找到适用于本后端的官方固定最低显存，故目录中的官方显存字段为空。

[Sherpa-ONNX 官方 Whisper 导出说明](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html) 列出 medium 的 ONNX 模型，推理需要 encoder、decoder 和 tokens 表；INT8 与 FP32 是不同的编码器/解码器文件对。[medium 官方发布文件](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/tree/main) 同时包含这两套精度。MySound 将其作为独立 `whisper_onnx` 模型类型，当前适配器使用 CPU，不套用原生 Whisper 的 CUDA 显存或 A100 相对速度数据。

[ONNX Runtime 量化说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) 指出量化收益与硬件指令集有关，量化/反量化也有开销。INT8 文件更小不代表本机必然更快；medium 仍需实际测试短句延迟。上述来源没有给出适用于本程序的 medium 固定最低主存或本机速度保证。

## 本程序的估算

下表单位为 **GiB**（1024³ 字节），是假设单任务、常见短句/30–60 秒切片下的规划区间，**尚未在本机逐型号实测**。原生 PyTorch 后端按 CPU FP32、CUDA FP16/BF16 估算；Sherpa-ONNX 行单独注明 INT8 / FP32，并仅估算当前 CPU 后端。不同句长、生成 token 数、库版本和其他进程占用都会影响峰值。我们没有从参数量反推精确延迟，也没有将 vLLM 或高并发性能套用到当前程序。

| 模型 | CPU 模式 RAM 预算 | CUDA 模式 VRAM 预算 | CUDA 加载时主存预算 |
| --- | --- | --- | --- |
| Whisper tiny | 1–2 | 1–1.5 | 1–2 |
| Whisper base | 1.5–2.5 | 1–2 | 1.5–2.5 |
| Whisper small | 2.5–4 | 2–3 | 2.5–4 |
| Whisper medium | 6–8 | 3.5–5.5 | 6–8 |
| Whisper large | 10–13 | 6–10 | 10–13 |
| Whisper turbo | 6–9 | 4–6.5 | 6–9 |
| Qwen3-ASR-0.6B | 5–7 | 3–4.5 | 2–4 |
| Qwen3-ASR-1.7B | 10–14 | 5.5–8 | 4–7 |
| Confucius4-R2T2 | 10–14 | 5.5–8 | 4–7 |
| Sherpa-ONNX Whisper medium INT8 | 3–6 | 不适用：CPU 后端 | 不适用：CPU 后端 |
| Sherpa-ONNX Whisper medium FP32 | 6–10 | 不适用：CPU 后端 | 不适用：CPU 后端 |

区间考虑权重表示和推理/加载余量：FP32 权重约为每参数 4 字节，FP16/BF16 约为每参数 2 字节，此外还要留激活、KV 缓存、CUDA 上下文和 Python/库内存。原生 Whisper `.pt` 的当前加载流程在 CPU 同时持有检查点与初始化后的 FP32 模型，再降为半精度并移到 GPU，因此 GPU 足够也必须检查主存。该行为可核对 [Whisper 官方加载实现](https://github.com/openai/whisper/blob/main/whisper/__init__.py) 与本项目 `core/model_manager.py`。Qwen/Confucius 的加载器和分片缓冲方式不同，主存估算单独列出。

Sherpa-ONNX 不经过上述 PyTorch `.pt` 加载流程。这里的 3–6 GiB / 6–10 GiB 是根据两套模型文件体积，并为 ONNX Runtime 会话、图优化、运行时缓冲和音频留出余量的**应用规划值**，没有把所有 769M 参数都视为 INT8，也没有宣称精确内存节省比例。FP32 ONNX 预算不等同于原生 PyTorch FP32 预算；实际图优化与分配策略可能造成不同峰值。资源卡的 `official_vram_gb`、`official_speed_relative` 对这两项均为空。

`disk_gib` 通常只统计用户选定文件或模型目录的直接文件大小，排除压缩包且不递归扫描其他模型；它不是下载大小预测。对 Sherpa-ONNX，`disk_gib` 专门统计此次选中的 encoder、decoder、tokens 三个文件，`folder_disk_gib` 另外记录模型目录的全部直接文件占用，避免把同时保存的 INT8 和 FP32 两套模型误当成一次加载的文件量。未知自定义权重不会自动套用 large 或其他固定型号预算。

选择 Sherpa 模型目录时优先完整 INT8 文件对；只有 INT8 不完整、FP32 完整时才使用 FP32。明确选择某个 `.onnx` 文件时严格匹配同模型、同精度的另一半及 tokens，不因另一套文件齐全而改变选择。资源信息复用推理加载器的路径解析函数，识别与加载保持一致。`Whisper-large-v3-turbo` 内的 `large-v3-turbo.pt` 识别为 turbo；Qwen3 优先读取 `thinker_config.text_config.hidden_size`，1024 / 2048 分别对应 0.6B / 1.7B，文件夹名称仅作后备，资源描述不因缺少可重建的 processor 元数据而把 0.6B 误判为 1.7B。

## 推荐规则与硬件检测

推荐比较当前 **MemAvailable** 和 NVIDIA **memory.free**，总容量仅供展示。低于预算下界标为资源不足；位于下界至上界的 115% 之间标为余量紧；高于该范围标为留有余量。这是应用的保守排序规则，不是官方硬性门槛。GPU 路线同时检查主存加载预算，CPU 路线独立评估；先选更有余量的路线，同等级优先 GPU。

`fit=recommended` 只表示估算资源有余量。`preferred` 表示系列内启发式优选，`top_recommendation` 表示整机首选（最多一个）；每项附带双语原因，明确没有本机测速。CPU 的 Whisper 首选上限为：核心未知或少于 4 个逻辑核心用 tiny；4–7 个用 base；至少 8 个用 small；内存有压力再向下选。核心数并非 CPU 性能基准，内存足够也不代表可实时听写。GPU 的 Whisper 优先 turbo 平衡吞吐，不能容纳时依次检查 large、medium、small、base、tiny；Qwen3 在 GPU 上选择资源有余量的较大版本，在 CPU 上保守优先 0.6B。Confucius 仅判断资源是否有余量并提示需实测延迟。整机首选优先 GPU 路线，在同一路线中优先上述 Whisper 平衡方案，其后是 Qwen3、Confucius；这是本程序的交互与资源偏好，**不是跨模型准确率排行**。所有其他型号仍返回原有资源评估供用户选择。

Sherpa-ONNX medium 使用独立 CPU 评估路线，即使检测到空闲 NVIDIA GPU，也不会为它推荐本适配器未实现的 CUDA 模式。INT8 仅在主存估算留有余量时标记为该后端的优选，FP32 仍返回评估结果；这不替代原生 tiny/base/small 的保守 CPU 低延迟首选，也不因量化而承诺 medium 更快。选择显式 FP32 文件后按 FP32 的主存区间评估。

当前后端的自动 GPU 设备是逻辑 `cuda:0`，不合并多卡显存，也不借第二张更大的卡做推荐。硬件字典保留 `cuda_ordinal`，支持 `CUDA_VISIBLE_DEVICES` 重排；但多 GPU 下 nvidia-smi 物理索引顺序不一定等同 CUDA 的默认顺序。无法在不初始化 CUDA 的前提下确定顺序时，GPU 预算标为未知；通过 `CUDA_VISIBLE_DEVICES=GPU-UUID` 明确选择 GPU 后可按该卡评估。

缺失读数保留为 `None`，不会把总容量当空闲容量，也不会把未知当作零。硬件扫描读取 `/proc`、`/etc/os-release`，并给 `nvidia-smi` / `lspci` 两秒超时；不导入 PyTorch、不申请 GPU 内存、不加载权重。检测到其他厂商 GPU 时显示硬件身份，并说明本应用的 GPU 后端目前使用 CUDA。识别到 NVIDIA 设备不代表已经验证 PyTorch/CUDA 驱动可用。
