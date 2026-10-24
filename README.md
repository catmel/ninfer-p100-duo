# NInfer P100 Duo — Qwen3.8-27B 推理引擎（2× Tesla P100, sm_60, TP2 NVLink）

**English summary**: a fork of the NInfer LLM inference engine, migrated to
dual Tesla P100 (GP100, `sm_60`) PCIe cards with NVLink1.0. Tensor-parallel decode with CUDA
Graphs, speculative MTP decoding, and OpenAI-compatible serving. Qwen3.8-27B Q6_K runs at
**33.4 tok/s decode / ~68 tok/s prefill** on 2×16 GB P100. Provenance and third-party citations:
see the two tables below and [`NOTICE`](NOTICE).

本仓库是 [NInfer](https://github.com/Neroued/ninfer) 引擎族面向 **Tesla P100 双卡** 的二次开发版本：
把原面向 RTX 5090 / V100 的执行层整体移植到 `sm_60`（无 Tensor Core、无 dp4a、无 cp.async 的
GP100），并完成 TP2 双卡验证与性能基线。全部迁移决策、实测证据与失败记录见
[`docs/p100-migration.md`](docs/p100-migration.md)（主台账）。上一代 V100 Duo 形态说明保留在
[`docs/v100.md`](docs/v100.md)。

## 二次开发基线（fork lineage）

| 顺序 | 项目 | Git 仓库 | 在本仓库中的角色 |
|---|---|---|---|
| 1 | Neroued/ninfer | <https://github.com/Neroued/ninfer> | **原始项目**（Apache-2.0），本仓库 fork 自其 commit `feaf4dd` |
| 2 | wamansou/ninfer-tp2-1m | <https://github.com/wamansou/ninfer-tp2-1m> | 双卡张量并行（TP2）与 YaRN 1M 长上下文改造 |
| 3 | geoffwatts/ninfer-v100 | <https://github.com/geoffwatts/ninfer-v100> | Volta (sm_70) 内核与构建支持移植 |
| 3b | huangserva/ninfer-v100-tpx | <https://github.com/huangserva/ninfer-v100-tpx> | TP2 INT8 key-split Volta attention 设计的适配来源（见 `NOTICE`） |
| 4 | tuxKOH/ninfer-V100-Duo | <https://github.com/tuxKOH/ninfer-V100-Duo> | **直接基线**：V100 双卡集成，保留 GGUF Q4_K/Q6_K artifact 转换与执行路径 |
| 5 | **catmel/ninfer-p100-duo（本仓库）** | <https://github.com/catmel/ninfer-p100-duo> | **sm_60 (P100) 迁移**：`NINFER_PASCAL_BUILD` 构建门控、P100 SIMT 反量化 GEMV 路线、双 P100 NVLink/PCIe 全链路验证与性能基线 |

本仓库在第 4 项基础上的改动范围：顶层 CMake 的 arch-60 门控、`src/ops` 各算子族在 `sm_60` 的
可选路由（dp4a 缺失、cp.async 降级、fp16x2 死路等 GP100 事实的处理）、Windows/Linux 双工具链
构建、Tesla P100-PCIe 16GB×2（NVLink1.0 互联）上的 TP2 验证与实测台账。

## 引用的第三方开源项目

**随源码 vendored 的代码**（许可证文件均在本仓库对应目录内）：

| 项目 | Git 仓库 | 版本/锚点 | 许可证 | 用途 |
|---|---|---|---|---|
| llama.cpp (ggml) | <https://github.com/ggml-org/llama.cpp> | `62bf73d25`（vendored 于 [`third_party/llama_cpp_fattn`](third_party/llama_cpp_fattn)）；`src/ops/common/volta_mma.cuh` 转写自其 `ggml-cuda/mma.cuh` | MIT | GGUF 量化格式的**解码语义**（oracle 的独立参考实现同源）；sm_70 flash-attention 路线 |
| cpp-httplib | <https://github.com/yhirose/cpp-httplib> | v0.18.3（`third_party/cpp-httplib`） | MIT | HTTP serve 层（OpenAI 兼容 API） |
| nlohmann/json | <https://github.com/nlohmann/json> | v3.12.0（`third_party/nlohmann`） | MIT | JSON 协议解析 |
| spdlog | <https://github.com/gabime/spdlog> | v1.17.0 / commit `79524dd`（`third_party/spdlog`，见其 `README.ninfer.md`） | MIT | 日志 |
| utf8proc | <https://github.com/JuliaStrings/utf8proc> | `third_party/utf8proc` | MIT | 文本分词/Unicode |

**仅做技术审计引用、未引入任何代码** 的社区 P100 项目（清单、许可证与逐条对照记录见
[`docs/p100-donor-catalog.md`](docs/p100-donor-catalog.md)）：llama.cpp 的 P100 社区分支/补丁集
（p100-llama-cpp、llama-cpp-p100-patches、qwen36-p100-patches）、Strata-for-p100，以及
llama.cpp 官方在 P100 上的公开基线。本项目对它们的 kernel 做了全量源码审计并与其实测经验
互相印证；最终实现为自主编写。

## 支持的模型 artifact

一个构建支持同族全部注册 artifact（`qwen3.8-27b` 家族，启动时选一）：

| Artifact | 量化 | 说明 |
|---|---|---|
| Qwen3.8-27B Coder390 EfficientThink **Q6_K** MTP | GGUF Q6_K + int8 KV | 主形态；16 GB×2 显存内完全容纳，含全词表 MTP 投机头 |
| Qwen3.8-27B **Q4_K** MTP | GGUF Q4_K + int8 KV | 低显存/高吞吐形态（基线数据见性能表） |

Q6_K artifact 下载地址：<https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer>
（下载 `.ninfer` artifact，而非源 checkpoint 或原始 GGUF）。

## 实测性能（2× Tesla P100-PCIe 16GB，NVLink1.0 + PCIe，CUDA 12.8，TP2）

统一条件：`--tp 2 --devices 0,1 --spec mtp --draft-tokens 3 --max-new 160`，空闲 GPU，
固定英文 prompt；decode 为端到端生成速度。

| Artifact | prefill tok/s | decode tok/s | 备注 |
|---|---|---|---|
| Q6_K-MTP | ~68 | **33.35**（×3：33.34/33.36/33.36） | 生产基线（V9 验证态） |
| Q4_K-MTP | 待测 | 待测 | 全量基线测试进行中，随结果更新 |

完整的达标/回退实验矩阵（含全部失败方向与原因）记录在
[`docs/p100-migration.md`](docs/p100-migration.md) §6–§7；数值资格记录（oracle 45 项等）在其 §7.x。

## 构建

要求：Linux x86_64（Windows/MSVC 亦可完整编译）、CUDA 12.4–12.x（**13.x 已移除 Pascal
支持，勿用**）、CMake 3.28+、Ninja、C++20、zlib 头文件与 pkg-config。P100 目标构建门控：

```bash
cmake -B build-p100-duo -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=60 -DNINFER_PASCAL_BUILD=ON
cmake --build build-p100-duo -j
```

或复用容器化构建环境（builder 镜像 = CUDA 12.8 devel + 上述工具链；runtime 镜像 =
FROM builder + `COPY apps/ninfer apps/ninfer-serve`，见 [`Dockerfile`](Dockerfile)）。

## 运行与服务

```bash
# CLI 一次性生成
./build-p100-duo/apps/ninfer --model /path/to/model.ninfer \
  --tp 2 --devices 0,1 --no-thinking --spec mtp --draft-tokens 3 \
  --max-new 160 --prompt 'Explain the tradeoffs of a power-of-two ring buffer, then write C code.'

# OpenAI 兼容服务
./build-p100-duo/apps/ninfer-serve /path/to/model.ninfer \
  --host 0.0.0.0 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 65536 --prefill-chunk 2560 --spec mtp --draft-tokens 3 --no-thinking
```

TP2 依赖 direct P2P：启动日志的 `direct peer P2P qualified` 行为资格证据。
详见 [`docs/cli.md`](docs/cli.md)、[`docs/serving.md`](docs/serving.md)。

## 测试与验证体系

- **数值 oracle**：每个浮点算子以朴素 FP64 独立实现对照（GGUF-native vector 路 15 形状 ×
  FP64 预算 + valid-columns 界 30 项；groupwise-int decode 回归 36/36），工具在 `tools/p100/`。
- **端到端回归**：`tests/`（TP1 golden-token 一致性、双卡图捕获、分裂算子对照）。
- **P100 专项探针**：`tools/p100/` 内核微基准（GEMV 变体矩阵）、拓扑/P2P 资格检查。
- **性能声明口径**：端到端数字只在整机推理层面声明，Op 微基准仅用于内核归因
  （方法论见 [`docs/performance/methodology.md`](docs/performance/methodology.md)）。

## 许可

Apache License 2.0（见 [`LICENSE`](LICENSE)）。上游与第三方版权、逐目录修改声明见
[`NOTICE`](NOTICE)：含对 Neroued/ninfer `feaf4dd`、wamansou/ninfer-tp2-1m、geoffwatts/ninfer-v100、
tuxKOH/ninfer-V100-Duo 的派生声明，以及 llama.cpp/MIT、cpp-httplib/MIT、nlohmann/json/MIT、
spdlog/MIT、utf8proc/MIT 的第三方引用。
