# Qwen3.6-35B-A3B × 双卡 P100 / V100（NVLink）完整支持方案（分析报告）

> 状态：**分析/设计文档，不含任何已实施代码变更**（唯一例外：§3.1 记录的 Q5_K 编解码支持
> 已于 2026-10-25 按用户指示在 27B 线落地，供 Q4LynnStyle 装载与 35B-Q4 选型共用）。
> 撰写日期 2026-10-24，2026-10-25 按实机测量修订；依据：NInfer 上游 fork 点 `feaf4dd` 的 35B-A3B 完整实现源码实测审查
> （`git clone https://github.com/Neroued/ninfer`，63 个相关文件）、本仓库 P100 迁移台账
> （`docs/p100-migration.md`、`docs/p100-migration-ledger.md`）、供体目录
> （`docs/p100-donor-catalog.md`）与双 P100 实机测量。
> 性能预估区间在 §8 明确标注"未实测"，将由 M0 对照实验校准。

## 1. 目标与范围

二开定位（用户口径）：当前框架**完全支持两个模型的解析** ——
`qwen3.8-27b`（dense，已达成）与 `qwen3.6-35b-a3b`（MoE，256 专家 top-8，本方案对象）——
在 **P100 与 V100 两种显卡的双卡 NVLink 形态**上运行。

硬件矩阵：

| 平台 | GPU | SM/架构能力 | 互联 | 显存预算 |
|---|---|---|---|---|
| P100 Duo（本仓库当前形态） | 2× Tesla **P100-SXM2-16GB**（实机 nvidia-smi 确证） | GP100 sm_60：56 SM，**无 Tensor Core、无 dp4a（那是 sm_61）、无 ldmatrix/cp.async/TMA**，HBM2 732 GB/s | NVLink NV4（4 链路，P2P up） | 32 GB |
| V100 Duo（16G 形态） | 2× Tesla V100-SXM2 16GB | GV100 sm_70：80 SM，**有 mma.sync.m8n8k4 fp16**，无 dp4a、无 ldmatrix（sm_75+）、无 cp.async（sm_80+）、无 TMA（sm_90+），HBM2 900 GB/s | NVLink2.0 ×6 | 32 GB |
| V100 Duo（32G 形态，**上游 `docs/v100.md` 实测平台**） | 2× Tesla V100-PCIe/SXM2 **32GB** | 同上 sm_70 | 同上 | **64 GB** |

## 2. 上游 35B-A3B 实现现状（feaf4dd 源码审计结论）

`qwen3.6-35b-a3b` 在 fork 点 `feaf4dd` 是**完整的产品实现**（本仓库于 2026-10-23 迁移启动日
整包删除，删除清单见门控台账末节）。恢复母源：

| 组件 | 路径（feaf4dd） | 内容 |
|---|---|---|
| MoE 算子族 | `src/ops/sparse_moe/`（decode/prefill/small_t 三路 + `sparse_moe_route.cuh` + `src/ops/wrapper/sparse_moe.cpp`） | top-8 路由选择、专家 MLP 三阶段（D1 router / D2 选择 / D3 专家执行） |
| 目标包 | `src/targets/qwen3_6_35b_a3b/`（config/package/variant/load bindings） | 几何、装载计划、Program 绑定 |
| 转换工具 | `tools/convert/qwen3_6_35b_a3b/`（convert.py / draft_head.py / inventory.py） | checkpoint→`.ninfer`（纯 Python，平台无关） |
| 资格测试 | `tests/ops/test_sparse_moe.cpp` + `tests/targets/qwen3_6_35b_a3b/*`（moe/recipe/reference/draft-head/engine，含 py 参考实现） | 语义与装载回归 |
| 评测 | `eval/configs/qwen3_6_35b_{aime,gpqa,needle_haystack}.yaml`、`bench/ops/sparse_moe_bench.cu` | 质量与基准 |

**模型几何**（`impl/config.h` 实测读取）：hidden 2048、layers 40、**每专家 intermediate 512**、
256 专家、top-8（+shared）、词表输出 248320、GDN 线性注意力（16k/32v 头 ×128，conv_kernel 4）
+ 全注意力（16q/2kv × head_dim 256，rotary 64）—— 与 27B 同一混合主干，MoE 替换 dense MLP。

**上游量化路线**（`bindings.cpp` 实测读取）：groupwise-int = **W8G32_F16S RowSplit（W8A16）**，
第 34/38/39 层用 Q6G64_F16S；另有 nvfp4 身份。**W8 全量 ≈ 33 GB > 32 GB 双卡显存预算**，
且 fp8/fp4 硬件解码需 sm_89/sm_1000——**P100/V100 均不可用上游权重形态**。
=> 本方案以 **GGUF-native 低比特权重（Q4_K 主用，Q6_K 视显存余量评估）** 为可行入口
（依据见 §3、§4）。

**指令级可移植性审计**（逐文件扫描结论）：

| 内核 | 架构敏感指令 | sm_60 | sm_70 | 判定 |
|---|---|---|---|---|
| `sparse_moe_route.cuh`（top-8 选择/softmax） | 无（纯 SIMT + shfl） | ✅ 可编可跑 | ✅ | A 类直接复用 |
| `decode/*_kernels.cu`（D1/D2/D3，38KB） | 无命中（bf16 软件仿真、uint8 codes、half scales） | ✅（性能按 GP100 事实重审 §6.1） | ✅ | A/B：语义复用，P100 需按本仓库 sm_60 GEMV 经验重做发射整形（`170-SM target` 注释表明 grid 按 5090 调参） |
| `prefill/*_kernels.cu`（62KB） | **ldmatrix ×10**（`ops/common/mma.cuh`，sm_75+，低架构 `__trap()`） | ❌ 不可用 | ❌ **V100 同样不可用**（ldmatrix 是 sm_75 指令） | B 类：需成替换路径（V100 用 VoltaMMA m8n8k4；P100 用 SIMT dequant tile / dequant→cuBLAS） |
| `small_t/*_kernels.cu`（13KB） | **TMA 命中 ×3**（sm_90+） | ❌ | ❌ | C 类剔除，小 token 数回落 decode/prefill 路 |
| attention（GDN + full）、KV int8、MTP/DFlash2 | 与 27B 共享算子族 | ✅ P100 已全链验证 | ✅ V100 分支已验证 | 直接复用 |
| 转换/装载/调度/runtime | 平台无关 C++/Python | ✅ | ✅ | 直接复用 |

## 3. 显存与带宽预算（决定"为什么必须 Q4"）

35B-A3B 权重总量（全 256 专家常驻，无 offload；offload 在双卡 TP2 + CUDA Graph 下会破坏
图语义，列为否决项）：

| 权重形态 | 总量 | +KV128K(int8)+GDN态+激活/MTP 头 | 32 GB 双卡结论 |
|---|---|---|---|
| W8A16（上游 groupwise-int） | ≈33.1 GB | ≈38 GB | ❌ 溢出 |
| GGUF Q6_K | ≈27 GB | ≈31.5 GB | ⚠️ 贴线（需砍 prefill-chunk/并发，风险高） |
| **GGUF Q4_K** | **≈20.4 GB** | **≈25 GB** | ✅ 余 ~7 GB，可 128K@并发1 |
| GGUF IQ3_S（Strata 风格） | ≈15 GB | ≈20 GB | ✅ 宽松（质量需 oracle/eval 把关） |

每 token 活跃读量（决定速度天花板）：8 专家 ×3 矩阵 ×2048×512×0.56B(Q4_K) ≈ **10.6 MB/层**
×40 层 ≈ 0.42 GB + 注意力/路由/输出头 ≈ **0.55–0.6 GB/token**。

| 平台 | 有效带宽（本仓库 27B 实证的 latency-bound gemv 区间） | 推算 decode 区间 |
|---|---|---|
| P100×2（NVLink） | 119–195 GB/s/卡聚合 × 折损 | **45–70 tok/s**（未实测，§8 M0 校准） |
| V100×2（NVLink2） | 同类 kernel 有效 150–250 GB/s | **70–110 tok/s**（未实测） |

对照系：Strata-for-p100 在其 3B-active MoE（IQ3_S，单卡）实机报 **28–32 tok/s**；llama.cpp
v0.6.0 P100 dense 27B 公开基线 10.3 tok/s（本仓库 27B Q6_K 实测 33.35 tok/s 的同一口径）。

### 3.1 128K 上下文上限：双 V100 先例 vs 双 P100 实测（2026-10-25 增补）

用户提出"之前双卡 V100 能达到 128K，其他项目也能"。两条线都核实过，结论是**该先例真实存在
但建立在 32GB 显存档位上，与当前 16GB P100 不矛盾**：

| 事实 | 数据 |
|---|---|
| 上游 V100 Duo 实测平台 | `docs/v100.md`：Tesla **V100-PCIe-32GB / V100-SXM2-32GB**（2×32=**64 GB** 预算），Q6_K 全量长上下文 sweep 在此达成 |
| 本仓库 P100 Q6_K-MTP 实测上限 | 2×P100-16GB（32 GB 预算）：MC=65536 可跑；≥81920 装载即失败：`Engine runtime reservation requires 5024082176 bytes, but only 4577217536 bytes are available`（**每卡** KV+graph+logits 预留 5.02 GB > 权重占用后仅存的 4.58 GB）——纯容量约束，与 YaRN/rope 无关（原生 rope 域 262144） |
| Q4 系（Q4_K/Q5_K 混合，Q4LynnStyle）预估 | 权重 19.9 GB → 9.95 GB/卡，每卡余 ~6 GB ≥ 5.02 GB 预留，**131072 预期可达**（含 128K 全量），正在实机阶梯验证 |
| llama.cpp 等其他项目 128K on P100 | 同口径成立的原因相同：16GB×2 上跑 27B 级 128K 需 Q4 及以下权重形态（其 FAQ 对 Pascal 长上下文推荐即 Q4_K_XL/Q3 级别），或 32GB 卡 |

推论（对 35B-A3B）：§3 表格中 **Q4_K ≈20.4 GB / IQ3_S ≈15 GB 的"✅ 可 128K"结论对 32 GB 双卡
预算（P100-16G×2 与 V100-16G×2）成立**；Q6_K 级在 32 GB 预算下 128K 为贴线不可靠区，64 GB
预算（V100-32G×2）才宽裕——与 27B 实测上限曲线完全同构，可作为 35B 容量规划的标定锚点。

**Q5_K 支持已于 2026-10-25 落地**（`feat(gguf): decode Q5_K blocks`：tensor/artifact/codec/
kernel/tests/oracle 全链接入，编译全绿；数值 oracle 定序待 GPU 恢复后执行）。这使 Q5_K 成为
35B-A3B 的第三档容量选项（≈23 GB，32 GB 预算下短中上下文），扩展 §3 表格的选型空间。

## 4. 权重路径选型：GGUF-native（复用 27B 已验证路线）

27B P100 已经在 `src/ops/linear/gguf` 上跑通 **GGUF-native 量化 → sm_60 vector GEMV**
全链（oracle 45 项、36 项 decode 回归、33.35 tok/s 生产数据）。35B-A3B 复用同一 codec
基建，需要扩展的是 **routed 语义**：

1. 专家权重按 `[expert][matrix][rows][cols]` 直接以 GGUF 块布局驻留，
   **不做权重重排**（遵守"无 runtime weight repacking"边界）；
2. D3 内核按 `top8 ids` 做 **gather-GEMV**：一次 kernel 发射覆盖 8 个专家 ×3 矩阵的
   row 空间（`PathsPerBlock` 机制上游已有：`(kTopK+1)/PathsPerBlock` 个 path-CTA），
   P100 发射整形按 56-SM 重算（27B 经验：`kVectorMinBlocks` Pascal=1、MT8 ping-pong
   双缓冲、寄存器 96/不越 spill 悬崖）；
3. 转换工具：`tools/convert/qwen3_6_35b_a3b/` 新增 GGUF-Q4_K 输入分支持续到 `.ninfer`
   注册身份（对齐 27B 的 `qwen3.8-27b/gguf-*` 身份命名，如 `qwen3.6-35b-a3b/gguf-q4k-mtp`）。

## 5. TP2 双卡专家切分方案（核心新增设计）

上游 35B 是单卡产品；V100-Duo/本仓库 TP2 只切过 dense 列并行/行并行。MoE 推荐
**expert-split（按专家维切分）** 而非列切：

- rank0 驻留专家 0–127，rank1 驻留 128–255；router 权重两卡复制（256×2048×1B≈0.5 MB，可忽略）；
- 每 token top-8 按 ids 分为本地组 + 远端组：本地专家直接 D3 执行；远端专家由持有卡执行后
  **只回传 2048 维激活（bf16/fp32，≈4–8 KB/token/卡）** 走 P2P 写合并（NVLink 往返延迟被
  CUDA Graph 内 multi-stream 重叠吸收；27B TP2 的 allreduce 通道已验证同一传输面）；
- 负载不均（每 token 8 专家期望各卡 4±1.2 个二项分布）：与 allreduce 并行度损失同量级，
  实测在 M3 里程碑设 3 组 accept 曲线定夺；备选列切方案（两卡各持每专家一半 row）传输
  更规整但 gather 地址计算翻倍——**先 expert-split，退化留列切**。
- 共享专家 + attention + 输出头：沿用 27B 已验证的列/行并行路由。

## 6. 分卡实施要点

### 6.1 P100（sm_60）

- decode 三路 kernel 语义级复用，**但按 GP100 四条实证事实重审**：dp4a 不可用（int8 码点
  走 SIMT 反量化，同 27B Q6KTraits 模式）、`cvt.rn.*` 四分之一速率（bf16 点积按台账 A16
  gate 经验用 `__bfloat1622float2` 批量转换）、HFMA2 死路、6 块/warp 即 spill 悬崖；
- prefill 的 ldmatrix/mma 路径替换：首选 **dequant→SIMT fp32 tile GEMM**（本仓库 27B
  prefill 68 tok/s 同构路线）；次选 dequant→cuBLAS HGEMM；
- small_t 剔除，token<T 的形态由 decode 路兜住；
- MTP/DFlash2 proposal 头：27B 已在 sm_60 验证（本仓库 spec mtp draft-tokens 3 生产运行），
  35B 的头只是形状差异。

### 6.2 V100（sm_70）

- 与 P100 共享 §4 GGUF-native 路线（V100-Duo 已有 Q4_K/Q6_K 转换基建）；
- prefill 替换路径不同：**V100 有 mma.sync.m8n8k4**，用 `third_party/llama_cpp_fattn`
  同源的 `volta_mma.cuh` fragment 装配替换 ldmatrix（27B V100 prefill fattn 已走这条路）；
- decode SIMT 路同 P100 语义、直接编译通过（sm_70 无上述指令冲突）。

### 6.3 共同

YaRN/长上下文：35B 与 27B 同 token domain（262144 原生），128K 无需 rope 缩放；
KV int8 与 GDN 状态路全部复用。

## 7. Git 参考项目（用户许可的取证渠道）

| 项目 | 用途 |
|---|---|
| <https://github.com/Neroued/ninfer> @ `feaf4dd` | **35B-A3B 全实现母源**（恢复基底；已本地浅取核实 63 文件） |
| <https://github.com/ggml-org/llama.cpp> | Q4_K/IQ3_S/Q6_K GGUF 块格式语义 + Qwen3-Next(35B-A3B 同类) MoE 参考内核；本仓库已 vendor 其 fattn/mma（MIT） |
| <https://github.com/tuxKOH/ninfer-V100-Duo>、<https://github.com/geoffwatts/ninfer-v100> | V100 侧 GGUF 转换/执行基建与 Volta 移植模式（本仓库直接基线链） |
| <https://github.com/huangserva/ninfer-v100-tpx> | TP2 INT8 attention 设计先例（NOTICE 已登记） |
| Strata-for-p100（本地审计，见供体目录） | P100 MoE 实机 28–32 tok/s 参照系 + `__launch_bounds__`/packed-codec 经验（只引思想） |
| NAS 本地 `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` | 同族 35B-A3B GGUF：**M0 对照实验载体**（llama.cpp P100/V100 实机速度基线，校准 §3 推算） |

## 8. 里程碑与验证计划（实施期；本文档不动代码）

| 里程碑 | 内容 | 验证 |
|---|---|---|
| **M0 对照**（无需改引擎） | llama.cpp 在 P100×2 / V100×2 跑 NAS 上的 35B-A3B Q4_K GGUF | 实机 tok/s 曲线 → 校准 §3 推算带 |
| **M1 恢复编译** | 从 feaf4dd 引入 63 文件，按台账 A/B/C 分类重接门控：route/decode 直通、prefill ldmatrix 段 `#if >=750` 隔离、small_t 剔除 | sm_60/sm_70 编译矩阵绿 + 单测 `test_sparse_moe` FP64 oracle（top-8 选择逐位、专家点积容差） |
| **M2 TP1 单卡实跑** | GGUF-Q4_K 转换 + 注册身份；单 P100（权重换 IQ3_S 或 TP1 用 V100 16G）e2e | 与 llama.cpp 同 checkpoint 输出 token 一致性（greedy）、质量 eval（AIME/GPQA/NIAH 上游 config） |
| **M3 TP2 双卡** | §5 expert-split + P2P 激活回传 + CUDA Graph | 与 TP1 同 seed greedy 输出一致；accept-rate 曲线；P100 NVLink/PCIe 双拓扑资格矩阵 |
| **M4 调优** | P100 发射整形（复用 27B V9 方法论：ping-pong/块数/寄存器预算）、V100 prefill VoltaMMA | 达标判据：P100×2 decode ≥ §3 下沿 45 tok/s、V100×2 ≥ 70；128K 衰减曲线达标 |

风险登记：① Q4_K 质量需 oracle+eval 双确认（W8→W4 是产品语义变化，遵循"当前设计可被
显式任务变更"）；② expert-split 负载不均若实测 >8% 损失则启用列切退路；③ 248K 词表输出头
在 sm_60 的行分块沿用 27B full-vocab MTP 已验证形态。

## 9. 结论

- 35B-A3B 在本框架**不是从零实现**：路由、decode 专家内核、混合主干 attention、转换与评测
  工具链都在 fork 点母源中完整存在，且审计显示 **decode 路零架构敏感指令**；
- 真正的新工作收敛在三处：**prefill 专家 GEMM 的双卡替换路径**（sm_60 SIMT / sm_70
  VoltaMMA，替换 sm_75+ 的 ldmatrix 段）、**TP2 expert-split 传输设计**、**GGUF-Q4_K 权重
  入口**（显存约束决定，W8 溢出 32GB 预算）；
- P100 与 V100 两条路线共享同一 §4 权重基建与 90% 内核代码，差异集中在 prefill 一个算子族；
- 速度目标有外部参照系（Strata P100 MoE 28–32 单卡；本仓库 27B 33.35 双卡 P100），推算带
  P100×2 45–70、V100×2 70–110 tok/s，**以 M0 llama.cpp 对照实验先行校准后再立项承诺数字**。
