# P100 门控站点台账(NInfer V100→P100 移植,范围 Qwen3.8-27B dense)

- 生成日期:2026-10-23;扫描基线:`ninfer-p100-duo` 工作树(20:35 复检版)。**前提事实:① 主线程已完成 35B-A3B/SparseMoe 整包删除(`src/ops/sparse_moe/`、`src/ops/wrapper/sparse_moe.cpp`、`include/ninfer/ops/sparse_moe.h`、`src/targets/qwen3_6_35b_a3b/` 及 bench/tests/tools/docs 35B 专属文件,原 11 个 VOLTA 门控站点随之消失,不再计入本台账);② 顶层 CMake 已定义 `NINFER_PASCAL_BUILD` 并把 cutlass 以 headers-only 扩到 arch-60(供 gguf/ggml_k 的 bfloat16 头类型,不建 TensorOp 核);③ `src/CMakeLists.txt` 的 w4a4/a8 EXCLUDE 与 stub 块已扩为 `^(60|70|86|89)$`,SM8X 定义排除 60;④ arch-70 专属源列表对 arch-60 不启用。** 门控行清点(去重):VOLTA 指令行 **211**(90 文件)+ SM8X 专有指令行 **10**(pdl/variant 双名行与 VOLTA 重合,不重计)。
- 扫描目录:`src/ include/ apps/ bench/ tests/ tools/`;门控:`NINFER_VOLTA_BUILD`、`NINFER_SM8X_COMPAT`、`__CUDA_ARCH__` 阈值、未受保护硬件指令。
- 硬件判定依据:P100 = sm_60(GP100)。**无 tensor core**(mma.sync/wmma/tcgen05/wgmma 全部不可用),**无 __dp4a**(那是 sm_61),HFMA2 满速,48 KB 静态 smem/块,L2 4 MB,56 SM,`cp.async` 需 ≥800,PDL 需 ≥900,FP8/FP4 硬件解码需 ≥890/≥1000。工具链 CUDA 12.4–12.x(13 已删 Pascal 编译),Windows(MSVC)+ Linux。
- 构建门控:`NINFER_PASCAL_BUILD`(顶层 CMakeLists.txt ~31–40 行,arch=60 时全局定义;条件校验 CUDA ≥12.4 且 <13)。
- 判定口径:
  - **A = SIMT 可直接复用**:分支体是纯 CUDA C/half2/fp32/host 路由常量,sm_60 可编可跑;把门控改名或双名 `defined(NINFER_PASCAL_BUILD) || defined(NINFER_VOLTA_BUILD)`。
  - **B = 需 P100 新路由**:Volta 分支走 mma.sync.m8n8k4 / QPN / VoltaMMA / CUTLASS-Sm70 / VoltaFlash;sm_60 必须换成 fp16 HFMA2 tile/ SIMT dequant 扩程 / dequant→cuBLAS HGEMM / 非-MMA flash。
  - **C = 剔除**:仅服务 35B-A3B/SparseMoe(以及其专属 16q/2kv GQA、G32 bf16 等范围外几何),新项目直接删。
  - **D = 编译即死可保留**:声明块/枚举/整文件由 `NINFER_VOLTA_BUILD` 或 `__CUDA_ARCH__==700 / >=750/800/890/900` 保护,sm_60 编译自然成空 TU 或 trap-stub;只需保证**运行时路由不选中**。
- 关键结构性事实(Volta 移植留下的地基,判定 A/B 的基础):
  1. `src/ops/common/mma.cuh`:全部 Ampere+/Blackwell PTX(ldmatrix/mma_bf16/f16/s8/fp8/tf32/nvfp4)集中于此,`__CUDA_ARCH__ >= 750/800/890` 之外走 `__trap()`——sm_60 **可编译**,误路由才炸。
  2. `src/ops/common/memory.cuh`:`cp.async` 在 `<800` 自动降级为同步 load/store(82–114 行);Volta/Pascal 复用同一 SIMT kernel 的机制已在 V100 端口验证。`pipe_copy`(`__pipeline_*`)靠 CUDA 头文件在 `<700` 的同步仿真降级——首编时留意告警。
  3. `src/ops/common/math.cuh`:`cvt.rn.bf16x2.f32` 有 `>=800` 保护(32 行),低架构走软件转换。bf16 intrinsics(`__float2bfloat16_rn` 等)在 `<800` 由 CUDA 头仿真,编译无虞,仅性能降级。
  4. Volta 张量核头(`volta_mma.cuh`、各 `*volta_qpn*.cuh`、`*volta_mma_gemm*.cuh`、`gqa_attention_decode_i8_tc_volta.cuh`、`i8_key_split_volta.cuh`、`gqa_attention_prefill_volta.cuh`、`sliding_window/volta.cuh`)全部包在 `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700`——sm_60 设备函数**直接消失**,任何"PASCAL 也调用它"的写法都是链接错误,故这类路由点只能 B。
  5. `gqa_attention_decode_bf16/i8.cuh` 带 `__CUDA_ARCH__ >= 800 / < 800` 双分支,`< 800` 分支是**纯 SIMT Volta decode**——sm_60 自动选中,零门控工作(本移植最大现成红利)。

## ① 统计摘要

(口径:去重后的预处理门控指令行;双名门控行只计一次;已删除文件的 11 个站点不统计。**本节数字由明细表 ② 逐行机器对账得出,与 221 个现存站点一一对应,无缺漏无重复**。)

| 判定 | 数量 | 说明 |
|---|---|---|
| A | 92 | SIMT/路由可复用,改门控即活 |
| B | 91 | 需 P100 新路由(多数第一阶段"删分支退 SIMT"即正确,性能项后补) |
| C | 2 | 剔除:仅剩 `bf16_gdn_gating_proj_plan.cpp:53/:447`(k35Routes/fused-norm-gating);sparse_moe 面的 11 个 C 站点已随主线程整包删除而消失 |
| D | 36 | 编译即死,保留即可(声明/枚举/被保护分支/SM8X schedule 死码 10 处) |
| 合计 | **221** | VOLTA 指令行 211(90 文件)+ SM8X 专有指令行 10(另有 3 行 CMake 结构门控不计判定,见 ② 末「构建门控」) |

按算子族分组(判定分布;SM8X 专有 D 已并入所属族并标注):

| 算子族 | A | B | C | D | 备注 |
|---|---|---|---|---|---|
| linear(w8/q4/q5/q6/bf16/fp8/nvfp4 + linear.cpp) | 32 | 20 | 0 | 11 | 含 SM8X 专有 D×5(w8_config×2、w8_rowsplit_gemm_splitk×3);SIMT rowsplit/GEMV 全家 sm_60 可编,宽 T 融合反量化(QPN/VoltaMMA/CUTLASS)集中于此 |
| linear_add / linear_pair / linear_swiglu | 20 | 21 | 0 | 14 | 含 SM8X 专有 D×5(linear_add splitk×4、pair splitk×1);add/pair 全 SIMT 合成,swiglu 的 fp8/nvfp4/q4 融合带是 B |
| attn_input_proj + gdn_input_proj + gdn_gating_proj | 15 | 38 | 2 | 9 | **B 最重族**:CUTLASS 宽 T + QPN + TP2 计划表;C×2 即 35B 残留路由 |
| attention(gqa/swa/vision/bidirectional)+ 几何 | 9 | 12 | 0 | 2 | SIMT decode 双分支白捡;VoltaFlash/TC-decode 是 B |
| linear_attention(GDN 路由) | 1 | 0 | 0 | 0 | chunked→recurrent 强制 SIMT |
| targets/qwen3_6_27b + core/pdl | 6 | 0 | 0 | 0 | prepack/A16Only/PDL 门控全 A |
| tests | 9 | 0 | 0 | 0 | 期望值双名即可 |
| 合计 | 92 | 91 | 2 | 36 | 221 |

B 族最重的三块(新路由工作量排序):① dense 宽 T GEMM(w8/q4/q5/nvfp4/fp8 的 VoltaMma+QPN+CUTLASS 全家,约 20 个 .cu 入口 + 各 plan 路由带 → 新 fp16 HFMA2 SIMT 扩程 dequant GEMM 或 dequant→cuBLAS HGEMM);② attention prefill/decode TC 路由(gqa/vision VoltaFlash、i8 TC decode → 非-MMA flash 变体;第一阶段先落 ChunkedSmallT/SIMT decode);③ 投影融合宽 T 带(attn_input/gdn_input/linear_swiglu 的 QPN-split → SIMT compose)。另加 gguf(`__dp4a`+CUTLASS 实例化)与 ggml_k(`wmma`)两个无条件编译炸弹的 B 重写(见 ④)。

## ② 全量明细表

### src/core

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| src/core/pdl.cuh:23 | SM8X\|\|VOLTA | 命中→`kernel<<<>>>` 普通发射;未命中→PDL 属性发射(`cudaLaunchAttributeProgrammaticStreamSerialization`) | A | 门控加 `NINFER_PASCAL_BUILD` 三名字;PDL 分支 host 属性在 sm_60 运行期会 `cudaErrorInvalidValue`(53/60 行设备侧 `>=900` 已保护) |

### src/ops/attn_input_proj(bf16/fp8/nvfp4/q4_q5/w8)

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| attn_input_proj/bf16/bf16_attn_input_plan.cpp:15 | VOLTA | 全 T→`bf16_attn_input_cutlass_sm70_launch`(#else→Ampere mma_launch,同样 trap) | B | Pascal 不进此分支;新路由=fp16 HFMA2 tile 或 dequant→cuBLAS HGEMM;第一阶段先让 SmallTSplit10/11 SIMT 扩程 |
| attn_input_proj/bf16/bf16_attn_input_plan.h:23 | VOLTA | cutlass launch 声明 | D | 保留单 VOLTA 名即可 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:6 | VOLTA | cutlass 头 include | D | 保留 VOLTA 名 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:37 | VOLTA | T≥33→CUTLASS+workspace 要求;T2–32→QPN | B | Pascal 删此带;T≥2 走 `fp8_attn_input_small_t_launch`(SIMT,已可用) |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:69 | VOLTA | chunk 命中 QPN→`launch_fp8_attn_input_volta_qpn` | B | 同上,退 SIMT small_t;后期 HFMA2 QPN 等价核 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:97 | VOLTA | maxT≥33 时 cutlass workspace 字节 | B | 删,Pascal 恒 0 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:128 | VOLTA | TP2 计划选 CutlassSm70 | B | Pascal 计划表去掉 CutlassSm70 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:134 | VOLTA | TP2 QPN 镜像 chunk 常量 | B | 与 :69 同批删 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:158 | VOLTA | TP2 QPN launch 发射 | B | 同 :69 |
| attn_input_proj/fp8/fp8_attn_input_plan.cpp:188 | VOLTA | TP2 cutlass workspace | B | 同 :97 |
| attn_input_proj/fp8/fp8_attn_input_plan.h:52 | VOLTA | qpn supported 声明 | D | 保留 VOLTA 名 |
| attn_input_proj/fp8/fp8_attn_input_volta_qpn.cu:10 | VOLTA | 整文件体=`fp8_volta_qpn_kernel` 包装(m8n8k4) | B | P100 编译时自然空 TU;B 新实现的同位替代入口 |
| attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:5 | VOLTA | cutlass 头 include | D | 保留 |
| attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:34 | VOLTA | T≥33→CUTLASS | B | 删带退 SIMT small_t(2..32) |
| attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:72 | VOLTA | TP2 计划 CutlassSm70 | B | 同上 |
| attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:120 | VOLTA | TP2 cutlass ws | B | 删 |
| attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:140 | VOLTA | dispatch T≥33→cutlass | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:4 | VOLTA | include(q4/q5 launch.h+cutlass.h) | D | 保留 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:37 | VOLTA | 计划表:小 T 保留 ParentSplitFixed(SIMT),宽 T→CUTLASS(注释详述) | B | 小 T 表 A 用;Pascal 计划枚举去掉 CutlassSm70/VoltaMmaFused,宽 T 走 Independent 扩程 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:173 | VOLTA | 单桶容量抬到 VoltaMmaFused 上限 | B | 删(Pascal 无 fused 带) |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:183 | VOLTA | VoltaMmaFused→ws(#else throw) | B | 删 schedule id |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:190 | VOLTA | CutlassSm70→ws | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:223 | VOLTA | T≥33→cutlass 探测/fused 带 | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:247 | VOLTA | VoltaMmaFused:`launch_q4/q5_volta_mma`×3 | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:274 | VOLTA | cutlass launch | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:310 | VOLTA | TP2 fused mma | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:327 | VOLTA | TP2 fused ws | B | 删 |
| attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu:91 | VOLTA | q4 parent-split 通用化到任意 T(免 grouped mma) | A | 双名门控,Pascal 白拿宽 T SIMT |
| attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu:192 | VOLTA | q5 同上 | A | 双名 |
| attn_input_proj/w8/w8_attn_input_kernels.h:9 | VOLTA | `w8_dflash2_tp2_attn_input_volta_launch` 声明 | A | 双名(launcher 本体 SIMT,内嵌宽 T mma 带另见 ⑥) |
| attn_input_proj/w8/w8_attn_input_plan.cpp:103 | VOLTA | TP2 SIMT 覆盖表(全 T) | A | 双名 |

### src/ops/gdn_gating_proj

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:29 | VOLTA | `k27Routes`:[512,2048]/[1024,2048]/[1536,2048] SIMT | A | 双名 |
| gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:53 | VOLTA | `k35Routes`(35B 专属几何) | C | 随 35B 删除 |
| gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:175 | VOLTA | 27B small-T 合法化(cols≥2,1..32) | A | 双名 |
| gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:447 | VOLTA | `fused_norm_gating=false`(fused 仅 35B) | C | fused 路由块随 35B 删 |

### src/ops/gdn_input_proj(fp8/nvfp4/q4_q5/w8)

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:6 | VOLTA | include | D | 保留 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:44 | VOLTA | ws:cutlass | B | 删 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:56 | VOLTA | T≥33→cutlass;T2–32→QPN | B | 删带退 SIMT small_t |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:83 | VOLTA | qpn launch | B | 同上 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:104 | VOLTA | TP2 cutlass | B | 删 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:110 | VOLTA | TP2 qpn chunk 常量 | B | 删 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:129 | VOLTA | TP2 qpn launch | B | 删 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:154 | VOLTA | TP2 cutlass ws | B | 删 |
| gdn_input_proj/fp8/fp8_gdn_input_plan.h:55 | VOLTA | qpn 声明 | D | 保留 |
| gdn_input_proj/fp8/fp8_gdn_input_volta_qpn.cu:9 | VOLTA | qpn 包装文件体 | B | B 新实现同位入口 |
| gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:4 | VOLTA | include | D | 保留 |
| gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:31 | VOLTA | ws:cutlass | B | 删 |
| gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:64 | VOLTA | T≥33→cutlass | B | 删带退 SIMT |
| gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:105 | VOLTA | dispatch cutlass | B | 删 |
| gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:125 | VOLTA | TP2 cutlass | B | 删 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu:73 | VOLTA | q4 Independent 循环通用化 | A | 双名 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu:174 | VOLTA | q5 同上 | A | 双名 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:4 | VOLTA | include | D | 保留 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:34 | VOLTA | 计划表:小 T SIMT 保留 + WideTVoltaMmaFused/Cutlass id | A | 双名(表体 SIMT);fused id 的 B 子项见 :164–:271 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:164 | VOLTA | fused mma 容量抬升 | B | 删 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:175 | VOLTA | cutlass 容量 | B | 删 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:247 | VOLTA | `launch_q4/q5_volta_mma`×3 | B | 删 |
| gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:271 | VOLTA | cutlass launch | B | 删 |
| gdn_input_proj/w8/w8_gdn_input_kernels.h:26 | VOLTA | SIMT row-view split 声明 | A | 双名 |
| gdn_input_proj/w8/w8_gdn_input_plan.cpp:23 | VOLTA | 计划表 SimtRowViewSplit(#else a3b 直路由 throw) | A | 双名;a3b 分支随 35B 删 |
| gdn_input_proj/w8/w8_gdn_input_plan.cpp:119 | VOLTA | conv 融合→Materialized(#else 16 列 SplitKMmaFused) | A | Pascal 走 Materialized(SIMT 投影+独立 conv) |
| gdn_input_proj/w8/w8_gdn_input_plan.cpp:164 | VOLTA | `w8_row_view` helper + row-view 覆盖表 | A | 双名 |

### src/ops/kernel / launcher(attention 族)

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| kernel/gqa_attention_geometry.cuh:19 | VOLTA | 27B DecodeSplits=560(80 SM 定标;#else 85) | A | 双名+按 56 SM 重新定标常量 |
| launcher/bidirectional_gqa_attention.cu:5 | VOLTA | include bidirectional…volta.cuh(纯 SIMT warp-dot) | A | 双名 |
| launcher/bidirectional_gqa_attention.cu:110 | VOLTA | `noncausal_gqa_volta_partial_kernel`(SIMT) | A | 双名 |
| launcher/gqa_attention.h:63 | VOLTA | kVoltaFlashQBlockTokens + flash launch/meta 声明 | B | 声明随 VOLTA 名保留;B 新 flash 路由的入口 |
| launcher/gqa_attention_decode.cu:8 | VOLTA | include tc-volta 内核头(`==700` 自保护) | D | sm_60 include 即空;可删可留 |
| launcher/gqa_attention_decode.cu:24 | VOLTA | `kVoltaTcDecodeMinWidth=65` host 常量 | D | 保留;Pascal 路由不用 |
| launcher/gqa_attention_decode.cu:110 | VOLTA | 发射 `gqa_attention_decode_i8_bf16out_tc_volta_kernel` | B | Pascal 禁入;decode 走 `decode_i8.cuh` `<800` SIMT 分支(现成) |
| launcher/gqa_attention_decode.cu:163 | VOLTA | TC 路由分支(width≥65→TC) | B | 去掉分支,全 T 落 SIMT fallback;或写 HFMA2 TC 等价 |
| launcher/swa.cu:5 | VOLTA | include sliding_window/volta.cuh | A | 双名(头函数体 `==700` 包裹?注意:volta.cuh 34 行 `==700` 保护的是内部片段;实测 sm_70 编译通过、Pascal 需核对该头 sm_60 分支为空时的调用点) |
| launcher/swa.cu:98 | VOLTA | `sliding_window_volta_kernel`(SIMT warp-dot) | A | 双名(同 :5 核对注) |
| launcher/swa.h:31 | VOLTA | `swa_dflash2_launch` 声明(SIMT split-kv) | A | 双名 |
| launcher/vision_attention.h:24 | VOLTA | vision VoltaFlash 声明 | B | 保留 VOLTA 名;B 新路由入口 |

### src/ops/linear(w8/q4/q5/q6/bf16/fp8/nvfp4/linear.cpp)

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| linear/bf16/bf16_dispatch.cpp:12 | VOLTA | DFlash2 [256,5120]→`launch_bf16_n256_k5120`(SIMT matvec) | A | 双名 |
| linear/bf16/bf16_dispatch.cpp:30 | VOLTA | 宽 T→`launch_bf16_cutlass_sm70` | B | 删,Pascal 用 small_t SIMT 扩程或 cuBLAS HGEMM |
| linear/bf16/bf16_launch.h:19 | VOLTA | n256 SIMT 声明 | A | 双名 |
| linear/bf16/bf16_launch.h:27 | VOLTA | cutlass 声明 | D | 保留 |
| linear/fp8/fp8_config.h:393 | VOLTA | 词表 [248320,2048] SIMT gemv schedule 注册 | A | 双名(P100 重扫 r 值) |
| linear/fp8/fp8_config.h:420 | VOLTA | 词表小 T 上限(10 行 SIMT 扫描) | A | 双名 |
| linear/fp8/fp8_dispatch.cpp:15 | VOLTA | `kVoltaBuild` 标志(词表 SIMT 准入) | A | 改名 `kPascalBuild` 或双名 |
| linear/fp8/fp8_dispatch.cpp:78 | VOLTA | chunk=kVoltaQpnMaxTokens(32) 否则 mma 上限 | B | Pascal 用 `fp8_linear_small_t_max(problem)`;QPN 带删 |
| linear/fp8/fp8_dispatch.cpp:99 | VOLTA | qpn 命中→`fp8_volta_qpn_gemm_launch` | B | 删,Pascal 全量 small_t/gemv SIMT |
| linear/fp8/fp8_gemv.cu:45 | VOLTA | 词表 T=1→`launch_fp8_gemv`(SIMT) | A | 双名 |
| linear/fp8/fp8_launch.h:14 | VOLTA | qpn 常量+声明 | D | 保留 |
| linear/fp8/fp8_small_t.cu:68 | VOLTA | 词表 T≥2→`launch_fp8_small_t`(SIMT) | A | 双名 |
| linear/fp8/fp8_volta_qpn_gemm.cu:9 | VOLTA | qpn 文件体(m8n8k4) | B | B 新实现同位入口 |
| linear/nvfp4/nvfp4_config.h:14 | VOLTA | A16Only 策略(nvfp4 无 A8) | A | 双名 |
| linear/nvfp4/nvfp4_config.h:275 | VOLTA | [1024,6144]/[1536,6144] 冷缓存 winner SIMT small_t schedule | A | 双名;常量(49KB smem/182 注册块)按 P100 重测 |
| linear/nvfp4/nvfp4_dispatch.cpp:55 | VOLTA | 计划/准入签名带 WorkspaceArena | A | 双名(为宽 T 新路由留参) |
| linear/nvfp4/nvfp4_dispatch.cpp:60 | VOLTA | 宽 T→`launch_nvfp4_volta_mma` | B | 删带 |
| linear/nvfp4/nvfp4_dispatch.cpp:94 | VOLTA | 14336 行 qpn2 带(T2–48) | B | 删 |
| linear/nvfp4/nvfp4_dispatch.cpp:120 | VOLTA | ws:2*qpn scratch | B | 删 |
| linear/nvfp4/nvfp4_dispatch.cpp:141 | VOLTA | `launch_a16(x,w,out,ws,stream)` 签名 | A | 与 :55 同批 |
| linear/nvfp4/nvfp4_launch.h:16 | VOLTA | qpn/mma 声明 | D | 保留 |
| linear/nvfp4/nvfp4_volta_mma_gemm.cu:11 | VOLTA | mma 文件体 | B | B 入口 |
| linear/nvfp4/nvfp4_volta_qpn_gemm.cu:9 | VOLTA | qpn 文件体 | B | B 入口 |
| linear/q4/q4_dispatch.cpp:128 | VOLTA | needs_volta_fallback(非 SIMT r8 表→c8) | A | 双名 |
| linear/q4/q4_dispatch.cpp:141 | VOLTA | fallback 应用 | A | 双名 |
| linear/q4/q4_dispatch.cpp:152 | VOLTA | 精确路由入口 fallback 检查 | A | 双名 |
| linear/q4/q4_dispatch.cpp:168 | VOLTA | qpn 带(T2–16)+volta_mma 带 | B | 删,Pascal 走 c8/gemv SIMT |
| linear/q4/q4_launch.h:14 | VOLTA | qpn/mma 声明 | D | 保留 |
| linear/q4/q4_rowsplit_gemm_simt.cuh:228 | VOLTA | T>4 激活 smem staging(48KB 预算内) | A | 双名(P100 同为 48KB/块;Full 特化预算已核) |
| linear/q4/q4_rowsplit_gemm_simt.cuh:301 | VOLTA | staging 的 syncthreads 消费路径 | A | 双名 |
| linear/q4/q4_small_t_mma.cu:56 | VOLTA | draft-head 投影→q4_simt_r8_c8(注册表外) | A | 双名 |
| linear/q4/q4_volta_mma_gemm.cu:13 | VOLTA | mma 文件体 | B | B 入口 |
| linear/q4/q4_volta_qpn_gemm.cu:9 | VOLTA | qpn 文件体 | B | B 入口 |
| linear/q5/q5_dispatch.cpp:126 | VOLTA | needs_volta_fallback | A | 双名 |
| linear/q5/q5_dispatch.cpp:139 | VOLTA | 精确路由 fallback+通用 SIMT 扩程(T≥33) | A | 双名 |
| linear/q5/q5_launch.h:14 | VOLTA | volta_mma 声明 | D | 保留 |
| linear/q5/q5_rowsplit_gemv.cuh:108 | VOLTA | `__launch_bounds__(256,4)` | A | 双名(P100 64K regs/SM 同预算,重测) |
| linear/q5/q5_rowsplit_gemv.cuh:226 | VOLTA | max-smem-carveout 请求 helper | A | 双名(Pascal 有该属性,语义弱;可保留) |
| linear/q5/q5_rowsplit_gemv.cuh:252 | VOLTA | carveout 调用(r4_c16 特化) | A | 双名 |
| linear/q5/q5_rowsplit_gemv.cuh:267 | VOLTA | carveout 调用(residual 特化) | A | 双名 |
| linear/q5/q5_volta_mma_gemm.cu:11 | VOLTA | mma 文件体 | B | B 入口 |
| linear/q6/q6_dispatch.cpp:94 | VOLTA | q6_launch_needs_volta_fallback | A | 双名 |
| linear/q6/q6_dispatch.cpp:107 | VOLTA | 非 SIMT 表→c8 | A | 双名 |
| linear/w8/w8_dispatch.cpp:21 | VOLTA | DFlash2 shard 表(T≤16 SIMT c4/c8) | A | 双名;宽 T 带见 :29 |
| linear/w8/w8_dispatch.cpp:29 | VOLTA | k=12800→volta_mma 否则 mma_r64 | B | 删,Pascal SIMT c8 扩程 |
| linear/w8/w8_dispatch.cpp:40 | VOLTA | k=25600 词表带:≤8 SIMT,>8 volta_mma | B | 删带,Pascal gemv/small_t SIMT 接管 |
| linear/w8/w8_dispatch.cpp:93 | VOLTA | 10240 行 n=5120 volta_mma | B | 删 |
| linear/w8/w8_dispatch.cpp:204 | VOLTA | needs_volta_fallback 拦截 | A | 双名 |
| linear/w8/w8_dispatch.cpp:283 | VOLTA | qpn(T2–32)+volta_mma(≥33) 带 | B | 删,Pascal rowsplit SIMT 覆盖 |
| linear/w8/w8_launch.h:13 | VOLTA | qpn/mma 声明 | D | 保留 |
| linear/w8/w8_rowsplit_gemm_simt.cu:16 | VOLTA | carveout 请求 helper | A | 双名 |
| linear/w8/w8_rowsplit_gemm_simt.cu:41 | VOLTA | carveout 特化调用 | A | 双名 |
| linear/w8/w8_rowsplit_gemm_simt.cu:82 | VOLTA | r8_c8 Full 超预算→关闭特化走通用 | A | 双名(Pascal spill 行为不同,先保守保留) |
| linear/w8/w8_rowsplit_gemm_simt.cuh:151 | VOLTA | `__launch_bounds__(32R,5)` | A | 双名(56 SM 重标定) |
| linear/w8/w8_small_t.cu:63 | VOLTA | MTP o_proj/down_proj→SIMT r8_c8 改道 | A | 双名 |
| linear/w8/w8_volta_mma_gemm.cu:9 | VOLTA | mma 文件体 | B | B 入口 |
| linear/w8/w8_volta_qpn_gemm.cu:9 | VOLTA | qpn 文件体 | B | B 入口 |
| linear/linear.cpp:155 | VOLTA | q4 volta_mma 带工作区上限 | B | 删,Pascal 返 0(或随新带上调) |

### src/ops/linear_add / linear_pair

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| linear_add/bf16/bf16_linear_add_plan.cpp:39 | VOLTA | 宽 T→`bf16_linear_add_volta`(SIMT warp-dot) | A | 双名 |
| linear_add/bf16/bf16_linear_add_plan.h:37 | VOLTA | 上者声明 | A | 双名 |
| linear_add/fp8/fp8_linear_add_plan.cpp:8 | VOLTA | include | D | 保留 |
| linear_add/fp8/fp8_linear_add_plan.cpp:22 | VOLTA | LinearThenAdd schedule id | A | 双名 |
| linear_add/fp8/fp8_linear_add_plan.cpp:42 | VOLTA | 准入:T==1→A16,T≥2→LinearThenAdd | A | 双名(Pascal 可直走 #else A16 fused,前提 load 不 prepack) |
| linear_add/fp8/fp8_linear_add_plan.cpp:81 | VOLTA | `launch_linear_then_add`(T≥33 内含 cutlass) | A | 双名;内部宽 T cutlass 子带=Pascal 改 Materialized |
| linear_add/fp8/fp8_linear_add_plan.cpp:133 | VOLTA | ws:cutlass→1024 否则 0 | A | 双名(Pascal 恒 0) |
| linear_add/fp8/fp8_linear_add_plan.cpp:148 | VOLTA | dispatch LinearThenAdd | A | 双名 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:7 | VOLTA | include | D | 保留 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:21 | VOLTA | LinearThenAdd id | A | 双名 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:42 | VOLTA | 准入(T≥2 因 prepacked QPN 布局) | A | Pascal 不 prepack→可走 A16 fused;双名保正确 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:75 | VOLTA | compose:gate/down SIMT | A | 双名 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:127 | VOLTA | ws | A | 双名 |
| linear_add/nvfp4/nvfp4_linear_add_plan.cpp:143 | VOLTA | dispatch | A | 双名 |
| linear_add/q5/q5_linear_add_plan.cpp:4 | VOLTA | include | D | 保留 |
| linear_add/q5/q5_linear_add_plan.cpp:53 | VOLTA | 宽 T→q5 通用 SIMT | A | 双名 |
| linear_add/q5/q5_linear_add_plan.cpp:167 | VOLTA | VoltaMmaFused 容量抬升 | B | 删 |
| linear_add/q5/q5_linear_add_plan.cpp:205 | VOLTA | WideTVoltaMmaFused→ws(#else throw) | B | 删 |
| linear_add/q5/q5_linear_add_plan.cpp:251 | VOLTA(!VOLTA 反向) | 未定义 VOLTA 时忽略 ws | A | Pascal 语义同 VOLTA:接受 ws 参数 |
| linear_add/q5/q5_linear_add_plan.cpp:278 | VOLTA | fused mma launch | B | 删 |
| linear_add/q5/q5_linear_add_plan.cpp:289 | VOLTA | cutlass launch | B | 删 |
| linear_add/w8/w8_linear_add_plan.cpp:202 | VOLTA | DFlash2 O/DOWN 精确 T SIMT 覆盖表 | A | 双名 |
| linear_pair/w8/w8_pair_plan.cpp:24 | VOLTA | [1024,2048] split2/4 SIMT 表 | A | 双名 |
| linear_pair/w8/w8_pair_plan.cpp:41 | VOLTA | [2048,2048] concat/c8 SIMT 表 | A | 双名 |

### src/ops/linear_attention

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| linear_attention/gated_delta_net/gated_delta_net.cpp:254 | VOLTA | 强制 T_full=0→recurrent SIMT(免 chunked mma) | A | 双名;chunked 内核(rowsplit_grouped_mma/ldmatrix)为 D trap |

### src/ops/linear_swiglu

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:7 | VOLTA | include | D | 保留 |
| linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:34 | VOLTA | QPNSplitWS 结构 | B | 随 qpn 带删 |
| linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:81 | VOLTA | T≥33→cutlass+silu_mul | B | 删,Pascal 走 Materialized SIMT |
| linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:105 | VOLTA | T∈17..32→qpn_split | B | 删 |
| linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:146 | VOLTA | ws 容量:cutlass/qpn split | B | 删 |
| linear_swiglu/fp8/fp8_linear_swiglu_plan.h:27 | VOLTA | qpn/qpn_split 声明 | D | 保留 |
| linear_swiglu/fp8/fp8_linear_swiglu_qpn_split.cu:11 | VOLTA | qpn_split 文件体 | B | B 入口 |
| linear_swiglu/fp8/fp8_linear_swiglu_volta_qpn.cu:9 | VOLTA | 旧全融合体(注释:仍正确但不用) | B | 删或留档 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:8 | VOLTA | include | D | 保留 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:27 | VOLTA | VoltaQpnFused/Split/Cutlass 枚举 | D | Pascal 不用可删枚举项 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:46 | VOLTA | 准入:T==1 QpnSplit,T≥2 QpnFused/#else A16 | B | Pascal 不 prepack→A16 SIMT fused(small_t+decode 现成)或补 fused SIMT |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:114 | VOLTA | bf16_to_fp16 scratch kernel(纯 SIMT) | A | 双名(服务 B 路由的 compose) |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:172 | VOLTA | ws:split 融合容量 | B | 删 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:253 | VOLTA | T==1 split 路由 | B | 删 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:276 | VOLTA | 路由 launch 块 | B | 删 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h:25 | VOLTA | qpn/qpn_split 声明 | D | 保留 |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cu:12 | VOLTA | bf16→fp16 kernel + split 体 | B | B 入口(头 12 行内 bf16→fp16 部分纯 SIMT) |
| linear_swiglu/nvfp4/nvfp4_linear_swiglu_volta_qpn.cu:9 | VOLTA | qpn 文件体 | B | B 入口 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:8 | VOLTA | include | D | 保留 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:39 | VOLTA | kVoltaMmaMinCols 常量 | B | 随 fused 带删 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:46 | VOLTA | 计划表:SmallTExact(SIMT compose)保留,CutlassSm70 新增 | A | 双名;Cutlass id 的 B 子项见下 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:149 | VOLTA | SmallTExact compose(内含 VoltaMmaFused 探测) | B | 探测带删 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:175 | VOLTA | cutlass 分支 ws | B | 删 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:226 | VOLTA | compose launch(内含 qpn 带) | B | qpn 带删,SIMT compose 留 |
| linear_swiglu/q4/q4_linear_swiglu_plan.cpp:277 | VOLTA | cutlass launch | B | 删 |
| linear_swiglu/w8/w8_linear_swiglu_plan.cpp:20 | VOLTA | DFlash2 GATE_UP [34816,5120] SIMT pair 表 | A | 双名 |

### src/ops/wrapper

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| wrapper/attn_input_proj.cpp:316 | VOLTA | TP2 DFlash2 QKV→`w8_dflash2_tp2_attn_input_volta`(SIMT rowsplit;内部宽 T 带调 volta_mma) | A | 双名;launcher 内宽 T mma 段=B 子项(删段全 T 走 SIMT) |
| wrapper/gdn_input_proj.cpp:949 | VOLTA | SIMT→tensor-core 融合 seam 常量 1024 | A | 双名(Pascal 暂无融合带,常量保留待 B 新路由复用) |
| wrapper/gqa_attention.cpp:27 | VOLTA | kVoltaChunkTokens=6(小 CTA 覆盖 80 SM) | A | 双名(按 56 SM 微调) |
| wrapper/gqa_attention.cpp:285 | VOLTA | volta_flash_route_possible 门 | B | Pascal 关闭(第一阶段)/新 flash 打开(第二阶段) |
| wrapper/gqa_attention.cpp:405 | VOLTA | 路由选择:flash vs ChunkedSmallT | B | Pascal 第一阶段=强制 ChunkedSmallT(SIMT,现成可用);目标=Pascal 非-MMA flash |
| wrapper/gqa_attention.cpp:506 | VOLTA | prefill 工作区:flash 元数据 | B | 随 :405 |
| wrapper/gqa_attention.cpp:569 | VOLTA | prefill 调用 `gqa_attention_volta_flash_launch` | B | 随 :405 |
| wrapper/swa.cpp:143 | VOLTA | DFlash2 部分槽上限(16 槽×8 头=128 CTA) | A | 双名 |
| wrapper/swa.cpp:149 | VOLTA | DFlash2 查询→`swa_dflash2_launch`(SIMT) | A | 双名 |
| wrapper/vision_attention.cpp:37 | VOLTA | 非交错 staging 结构(服务 flash 路由) | B | 随新 vision flash 路由 |
| wrapper/vision_attention.cpp:132 | VOLTA | pre-stage T≥65→flash | B | Pascal 第一阶段禁入(Ampere vision 内核为 ldmatrix trap,也须 SIMT 替代)→vision attention 需 B 新路由 |
| wrapper/vision_attention.cpp:163 | VOLTA | staging 选择 | B | 同 :132 |
| wrapper/vision_attention.cpp:194 | VOLTA | `vision_attention_volta_flash_launch` 调用 | B | 同 :132 |

### src/ops/sparse_moe / wrapper——已从树上删除,不计站点

原 11 个 VOLTA 门控站点(`sparse_moe/prefill/sparse_moe_prefill_kernels.cu` 7 处 + `wrapper/sparse_moe.cpp` 4 处,C 类)随主线程整包删除而消失;其专家 GEMM 的 Volta SIMT/QPN/TC-epilogue 路由经验只作为 ⑤/⑥ 中同名可复用内核的旁证,不再登记。

### src/targets/qwen3_6_27b

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| targets/qwen3_6_27b/impl/load/bindings.cpp:5 | VOLTA | include prepack/split_launch 头 | A | 双名或删(视 B 路由) |
| targets/qwen3_6_27b/impl/load/bindings.cpp:249 | VOLTA | nvfp4 gate/up/down load 期 QPN prepack | A | prepack kernel 纯内存变换 sm_60 可编;Pascal **第一阶段不 prepack**(避免布局耦合),保留 A16 直读 |
| targets/qwen3_6_27b/impl/load/bindings.cpp:1249 | VOLTA | device-scope guard(单设备) | A | 双名 |
| targets/qwen3_6_27b/impl/load/bindings.cpp:1315 | VOLTA | fp8 12288×2048 QPN prepack | A | 同 :249 |
| targets/qwen3_6_27b/impl/variant.cpp:48 | SM8X\|\|VOLTA | 线性策略=A16Only(无 int8/fp8 张量核) | A | 门控三名(加 PASCAL) |

### tests

| 文件:行 | 门控 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|---|
| tests/ops/linear/linear_test_common.cpp:4 | VOLTA | prepack 头 include | A | 双名 |
| tests/ops/linear/linear_test_common.cpp:320 | VOLTA | QPN-prepacked 测试形状 | A | 双名(或随 prepack 决策) |
| tests/ops/linear/test_bf16_a16.cpp:158 | VOLTA | 测试形依赖 VOLTA 构建 | A | 双名 |
| tests/ops/linear/test_w8_a16.cpp:18 | VOLTA | DFlash2 W8 形状准入 | A | 双名 |
| tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:5 | VOLTA | prepack include | A | 双名 |
| tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:277 | VOLTA | 同上 | A | 双名 |
| tests/ops/test_linear_split.cpp:613,832 | VOLTA | DFlash2 W8/Q4 tp2 形状期望 | A | 双名 |
| tests/ops/test_linear_swiglu_split.cpp:750 | VOLTA | 注册表策略期望(VOLTA 下 T=17 A16Only 必须放行) | A | 双名(Pascal 同样放行 SIMT 路由) |

### 构建门控(结构行,不计 A–D;当前态已随主线程 20:35 快照核对)

| 文件:行 | 门控 | 摘要 |
|---|---|---|
| CMakeLists.txt:31–40(行号随主线程改动漂移) | PASCAL/VOLTA | arch=60:`<12.4 或 ≥13 → FATAL` + `add_compile_definitions(NINFER_PASCAL_BUILD=1)`;arch=70:`NINFER_VOLTA_BUILD=1`;cutlass FetchContent(headers-only)条件已扩为 `^(60\|70)$`,注释明确「sm_60/sm_70 build no CUTLASS TensorOp kernels, headers only」——**满足 gguf/ggml_k 的头类型需求,但 TensorOp 实例化仍须源码级门控(见④)** |
| src/CMakeLists.txt:~321–339 | 60\|70\|86\|89 | w4a4/a8/causal_cache-fp8 EXCLUDE + stubs 挂载分支**已含 60**;`NINFER_SM8X_COMPAT` 条件已改为 `NOT 70 AND NOT 60`(60 不定义 SM8X)✓;遗留小项:stub 文件名仍叫 `nvfp4_sm86_stubs.cpp`/`fp8_sm86_stubs.cpp`(语义混乱源,改名不阻塞) |
| src/CMakeLists.txt:~341–346 | arch-60 专属块 | 主线程新加:`if(60) target_link_libraries(ninfer_ops PRIVATE nvidia::cutlass::cutlass)`——为 gguf/ggml_k 的 bfloat16 转换头类型;并注明「compute routes gain NINFER_PASCAL_BUILD paths during the migration」,即 ④-2 的门控由主线程在 gguf.cu/ggml_k.cu 源码侧落实 |

### NINFER_SM8X_COMPAT 专有站点(sm86/89 兼容路径)

| 文件:行 | 路由摘要 | 判定 | P100 计划 |
|---|---|---|---|
| linear/w8/w8_config.h:58 | `W8SmallTMmaDefaultSchedule` 在 SM8X 下 KWarps=4/MinBlocks=2(#else sm120 参数) | D | Pascal 两者皆不取:上游 dispatch 已改道 SIMT,mma schedule 参数天然死码 |
| linear/w8/w8_config.h:210 | 同族 schedule(ActiveTokens→kKWarps 4/8/16) | D | 同上 |
| linear/w8/w8_rowsplit_gemm_splitk.cu:33 | exact-T splitk `launch_active_cols` SM8X 参数(KWarps/MinBlocks) | D | SM8X 未定义→取 #else sm120 常量;该文件在 Pascal 只要求"不被路由"(w8_small_t_mma_kernel `>=800` trap) |
| linear/w8/w8_rowsplit_gemm_splitk.cu:133,163 | medium T(49..128):SM8X→`launch_w8_exact_t_composite` 分块 | D | 同上 |
| linear_add/w8/w8_linear_add_gemm_splitk.cu:19,35 | kLastExactCols=32/KWarps 参数 | D | 同上 |
| linear_add/w8/w8_linear_add_gemm_splitk.cu:100,136 | SM8X 分块循环(>32 列切片) | D | 同上 |
| linear_pair/w8/w8_pair_gemm_splitk.cu:132 | SM8X 中 T 分块循环 | D | 同上 |

注:SM8X 路径本质是"Ampere/Ada **mma 内核**的参数与分块适配",并非纯 dequant→fp16——对 P100 参考价值有限(判定集中 D,靠 Pascal 路由改道解决,不靠改名)。真正给 P100 的 A 类红利的是 VOLTA 分支里的 SIMT 家族与 `__CUDA_ARCH__<800` 双分支内核(见⑤)。

## ③ 移除面账目(35B-A3B / SparseMoe;27B dense 保留项已排除)

**主线程已于本快照前执行整包删除,以下为核对结果(Test-Path/grep 复核)**,防止回潮用:

已删除并核对(现存 = 无):

- `src/targets/qwen3_6_35b_a3b/`(8 文件)、`src/ops/sparse_moe/`(10 文件)、`src/ops/wrapper/sparse_moe.cpp`、`include/ninfer/ops/sparse_moe.h`
- `tests/targets/qwen3_6_35b_a3b/`(10 文件)、`tests/ops/test_sparse_moe.cpp`、`bench/ops/sparse_moe_bench.cu`、`bench/targets/qwen3_6_35b_a3b/`
- `tools/reference/qwen3_6_35b_a3b/`、`tools/convert/qwen3_6_35b_a3b/`
- `docs/performance/qwen3.6-35b-a3b.md`、`docs/maintainer/qwen3.6-35b-a3b-model.md`
- 代码钩子已清:src/CMakeLists(源列表/`add_subdirectory`)、`targets/registry.{h,cpp}`(27B 单臂 variant)、`runtime/engine/engine.cpp`(Executor 单臂)、`core/nvtx.h`、`include/ninfer/types.h` 注释、tests/bench CMake 与 27B 化后的 `test_sampling_defaults`/`test_gqa_attention`/`test_gdn_gating_proj`/`linear_bench`/`gdn_replay_bench`

**尚未删除、后续要清的 35B 专属文件**(当前树仍存在,逐文件):

- `tests/convert/qwen3_6_35b_a3b/`:test_convert.py(:5)、test_draft_head.py(:5,8)、test_inventory.py(:1,5,7)、test_recipe.py(:4)、`__init__.py`(:1)
- `tests/convert/qwen3_6_27b/test_official_resources.py`(:13,25)与 `tests/targets/qwen3_6_27b/test_official_resources.py`(:12,18–28):27B 测试里仍引用 35B 资源/identity,需 27B 化改写而非删除
- `tests/data/tp1-golden/MANIFEST.md`(:89)、`tests/README.md`(:60,139,145,146,288,291,292,331)
- `tools/README.md`(:17,20,40–43,63,64)、`tools/artifact/migrate_v1_to_v2.py`(:30,63,64,67)、`tools/bench/README.md`(:122)、`tools/bench/run_serve_concurrency.py`(:224,225)、`tools/bench/run_serve_corpus.py`(:25,384,385)
- `bench/README.md`(:140,197,198,208,211,259,557,580–582,652–673,694,718——含 sparse_moe bench 章节残文)
- `docs/cli.md`(:38,116,120,131,184,185,218,244)、`docs/serving.md`(:20,24,39,95,593)、`docs/performance.md`(:1662–2333 共 17 处)、`docs/README.md`(:30,62,63)、`docs/maintainer/{linear-benchmark(19 处),model-weight-execution(:99,1050,1093),paged-kv-cache(:603–612,1400–1402),artifact-container(:203,204,466),replayssm-gdn(:36,564–586),softmax-attention(:141–144,457),concurrent-inference-architecture(:1426),qwen3.8-27b-dflash2(:31,525,546,549)}`、`docs/v100.md`(:5,91,98,152;V100 文档本身应被 p100 文档取代)
- 勿动:`docs/p100-migration.md`、`docs/p100-donor-catalog.md`、`tools/p100/device_probe.cu`(主线程/兄弟子代理产物);`docs/maintainer/tp2-yarn-1m.md` 若主线程认为 YaRN/1M 也不进 P100 范围则一并删(当前仍在)

共享算子文件里的 **35B 残留几何路由**(代码,非注释)单列到 ⑦,随算子重写逐个处理。

**注意保留**:`ops/linear_topk/`(DFlash2 unary top-k,27B 投机解码面)与 `candidate_selector/dynamic_grouped_conv/rmsnorm_rope/context_kv_materialize/swa_dflash2/w8_dflash2_attn_input` 均为 **DFlash2 27B 功能**,不属于移除面;它们的问题只在 arch-70 源列表归属(见⑥)。`gqa_attention_decode.cu:260` 一带的 16-q-head 形是 35B 的;27B vision 的 16q 走另一几何(`Gqa27Tp2Geometry`),删 35B 分支时勿连带误删。

## ④ 首批编译会炸的点(arch=60 首编预测,按炸序)

1. **`src/ops/linear/ggml_k/ggml_k.cu` — 未受保护 `nvcuda::wmma`**。412–514 行 `gemm<>` 内核直接用 `wmma::fragment/fill_fragment/load_matrix_sync/mma_sync/store_matrix_sync`(16x16x16 half→f32),**无 `__CUDA_ARCH__>=700` 保护、无 VOLTA 门**,而该文件在 `src/CMakeLists.txt:66` 无条件编译。WMMA 在 sm_60 是硬编译错误(mma 需 Volta+)。**处置**:把 `gemm<>` 包进 `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700`+host 端 throw(仿 `dflash2_sm70_stub.cu` 模式),或从 arch-60 列表剔除并挂 stub;**B 新路由**:ggml K-quant(GGUF 兼容路径)在 P100 走 fp16 HFMA2 SIMT dequant matvec 或 dequant→cuBLAS HGEMM。
2. **`src/ops/linear/gguf/gguf.cu` — 两处硬炸**(`src/CMakeLists.txt:67` 无条件编译):
   - :123/:179 两处 **`__dp4a`**(int8 4 路点积,sm_61+)——sm_60 上 nvcc/ptxas 直接报错,**已确证编译即死**;首爆点。
   - :457–465 **CUTLASS `Sm70 + OpClassTensorOp + GemmShape<8,8,4>`** 无条件实例化 + :522–531 `gemm.can_implement/initialize/operator()`——主线程已把 cutlass 以 headers-only 挂给 arch-60(供 :7–11 的 `cutlass::bfloat16_t/half_t` 转换类型),**但 TensorOp Gemm 实例化在 sm_60 仍被 ptxas 拒绝**;CMake 注释("sm_60 instantiates no CUTLASS kernel")意味着计划就是把 :457+ 实例化与 launch 块加 `NINFER_PASCAL_BUILD` 门去掉,只留头类型使用。
   - **处置与 B 路由**:P100 的 GGUF A8 路线改 int8→fp32 反量化 SIMT 点积(或 x 侧 bf16、W 侧 fp32 FMA;Pascal 无 dp4a,可用 half2/`__dp2a`?——`__dp2a` 同为 6.1+,只能标量或 fp16x2),词表类形状可 dequant→cuBLAS。**依赖链**:`linear_topk.cpp:5`、`variant.cpp:24-25`、`bindings.cpp:16`、tests include `gguf.h`——头可编,问题只在 .cu。
3. **`src/CMakeLists.txt` w4a4/a8 EXCLUDE 与 stub 块**:~~未含 60~~ **主线程已扩为 `^(60|70|86|89)$` 且 SM8X 定义排除 60(核对通过)**——`nvfp4/*_w4a4.cu`、`fp8/*_a8.cu`、`causal_cache/*_fp8.cu` 在 arch-60 同样被剔除并挂 `ops/nvfp4_sm86_stubs.cpp`/`ops/fp8_sm86_stubs.cpp`。遗留核对项:两个 stubs 文件名仍叫 "sm86",建议改名或加注释避免语义混乱(不阻塞编译)。
4. **PDL 运行地雷(非编译)**:`src/core/pdl.cuh` 未命中 SM8X/VOLTA/PASCAL 门时走 `cudaLaunchKernelEx`+ProgrammaticStreamSerialization;sm_60 上返回 `cudaErrorInvalidValue`。门控三名即解。
5. **低危核对项**:
   - `pipe_copy/__pipeline_*`(memory.cuh:116–128,q4_swiglu_gemv/w8/q5 rowsplit SIMT 用):依赖 CUDA 头在 `<700` 的同步降级——首编留意告警,双缓冲收益在 P100 归零(带宽余量变小,不是正确性)。
   - `cudaFuncAttributePreferredSharedMemoryCarveout`(q5_gemv/w8_simt carveout):Pascal 上该属性存在但只有动态 smem 语义,行为近 no-op,安全。
   - `bf16` 算术(`__hmul2/__hadd2/__hfma2` for `__nv_bfloat162`、`__float22bfloat162_rn` 等):树内 600+ 处,全部由 CUDA 头 `<800` 软件仿真保护——**可编可跑,仅慢**;`math.cuh:36 cvt.rn.bf16x2.f32` 有 `>=800` 门+回退。
   - `ld.global.cg.v*` asm(bf16/fp8/nvfp4_gemv):sm_60 合法,无风险。
   - `mbarrier.cuh`:未受保护 mbarrier asm,但**当前无任何文件 include 它**(死头,Pascal 无害);若 B 新 flash 想用 mbarrier 风格 barrier,P100 上不可用,须用 `__syncthreads`/named barrier。
   - `third_party/llama_cpp_fattn`(gqa/vision volta_flash 专用 include 目录,321–326 行):内核内部就是 m8n8k4 asm——Pascal 不编它(随 arch-70 列表走)即无害。

## ⑤ A 类立即可复用清单(主线程第一口粮)

改双名门控 `defined(NINFER_PASCAL_BUILD) || defined(NINFER_VOLTA_BUILD)`(或改名 `NINFER_PASCAL_BUILD` 单名后两处并存)即得全功能 P100 内核,按收益排序:

1. **GQA decode SIMT 双分支**——`kernel/gqa_attention_decode_bf16.cuh:448+`、`gqa_attention_decode_i8.cuh:627+` 的 `__CUDA_ARCH__ < 800` 分支(纯 fp32/half2 Volta decode):**零门控白拿**,sm_60 自动选中;仅需把 `gqa_attention_geometry.cuh:19` DecodeSplits 按 56 SM 重定标,并删掉 :110/:163 TC 分支(Pascal 不命中即可,包在 VOLTA 名单名下也行)。
2. **W8 rowsplit SIMT 全家**——`w8_rowsplit_gemm_simt.cu/.cuh`(r8_c4/c8、c16 split4、r16_c8)、`w8_small_t.cu:63` MTP 改道、`w8_dispatch.cpp:21/:204` SIMT 表与 fallback 拦截、`linear_add/w8:202`、`linear_pair/w8:24/41`、`linear_swiglu/w8:20`:DFlash2 与 MTP 全部形状 T=1..16 精确覆盖。
3. **Q4/Q5/Q6 通用 SIMT**——`q4_rowsplit_gemm_simt.cuh:228/301`(smem staging,48KB 预算与 P100 完全一致)、`q5_rowsplit_gemv.cuh:108/226/252/267`、`q4/q5/q6_dispatch` fallback 五处、`q4_small_t_mma.cu:56` draft-head;`q4_q5_attn_input_small_t.cu:91/192`、`q4_q5_gdn_input_independent.cu:73/174` 的宽 T 通用化——**这是 27B groupwise-int 主 GEMM 在 P100 的产能主体**;唯一性能缺口是 T≥33 预填充带宽效率(Pascal 16B 向量化 load 在 GP100 有效)。
4. **NVFP4/FP8 A16 SIMT(dequant 路线)**——`fp8_config.h:393/420`+`fp8_gemv.cu:45`+`fp8_small_t.cu:68` 词表;`nvfp4_config.h:14/275` gate/up/down SIMT small_t;前提:bindings 的 QPN prepack(:249/:1315)在 Pascal 不启用,直读原布局。
5. **bf16 面**——`bf16_dispatch.cpp:12`+`bf16_launch.h:19`(DFlash2 n256×5120 SIMT matvec)、`bf16_linear_add_plan.cpp:39`(SIMT warp-dot wide-T)、`bf16_gdn_gating_proj_plan.cpp:29/175`、`bf16_attn_input` SmallT 系。
6. **注意力投影合成路由**——fp8/nvfp4 `linear_add` 的 `LinearThenAdd` 六件套(§表)、`wrapper/attn_input_proj.cpp:316` DFlash2 TP2 QKV(SIMT 主体)、`gdn_input_proj/w8` 三件套(SimtRowViewSplit)、`gdn_input_proj/q4_q5:34`。
7. **GDN 线性注意力**——`gated_delta_net.cpp:254` 强制 recurrent SIMT;chunked 内核自动 trap 死。
8. **SWA/DFlash2 全套 SIMT**——`swa.cu:5/98`(非因果 warp-dot)、`swa.h:31`+`swa.cpp:143/149`+`swa_dflash2_volta.cu`(split-kv 2048 窗)、`wrapper/gqa_attention.cpp:27` chunking 常量。
9. **布局/内存 kernel**——`fp8_prepack_sm70.cu`、`nvfp4_prepack_sm70.cu`、`append_volta.cu`(bf16→fp16 KV 追加)、`dflash2_linear_topk_volta.cu`(top-16)、`dflash2_selector_volta.cu`、`dflash2_dynamic_conv_volta.cu`、`dflash2_rmsnorm_rope_volta.cu`:纯 CUDA C,把文件并进 arch-60 源列表即编即跑。
10. **门控外围**——`variant.cpp:48` A16Only、`pdl.cuh:23` 普通发射、`bindings.cpp:1249` 单设备 guard、tests 8 处期望。

第一阶段里程碑建议:**只做"改名/双名 + 删 B 分支 + CMake 扩 60"就能让 27B groupwise-int 与 nvfp4(A16)在 P100 上跑通 decode(T=1..16)与 ChunkedSmallT 预填充**;宽 T 高吞吐(⑥ 的 B 族)是第二阶段性能工程。

## ⑥ 交叉核对:src/CMakeLists.txt arch-70 专属源列表(268–327 行)

| CMake 行 | 文件 | 性质 | P100 计划 |
|---|---|---|---|
| 270 | ops/linear/bf16/bf16_n256_k5120_volta.cu | SIMT matvec(DFlash2) | **A**:并入 arch-60 列表(改名或共用) |
| 271 | ops/attn_input_proj/w8/w8_dflash2_attn_input_volta.cu | SIMT rowsplit 主体 + 内部宽 T `launch_w8_volta_mma` 带(81–83 行) | **A+B**:并入;宽 T mma 带门控到 VOLTA-only |
| 272–273 | ops/linear_topk/linear_topk.cpp、dflash2_linear_topk_volta.cu | host 分派 + SIMT top-16 | **A**(DFlash2 保留;linear_topk.cpp 本体在无条件列表,只 _volta.cu 需进 60) |
| 274 | ops/launcher/swa_dflash2_volta.cu | SIMT split-kv | **A** |
| 275–276 | ops/wrapper/candidate_selector.cpp、dflash2_selector_volta.cu | SIMT walk | **A** |
| 277–280 | dynamic_grouped_conv.cpp、prepare/add plan、dflash2_dynamic_conv_volta.cu | SIMT 2-tap 深度卷积 | **A** |
| 281–282 | rmsnorm_rope.cpp、dflash2_rmsnorm_rope_volta.cu | SIMT | **A** |
| 283–284 | context_kv_materialize.cpp、append_volta.cu | SIMT bf16→fp16 追加 | **A** |
| 285 | ops/linear/bf16/bf16_cutlass_sm70.cu | CUTLASS Sm70 mma | **B**:新实现入口(HFMA2 tile/cuBLAS)或剔除退 SIMT 扩程 |
| 286 | attn_input_proj/bf16/bf16_attn_input_cutlass_sm70.cu | CUTLASS | **B** |
| 287 | linear_add/bf16/bf16_linear_add_volta.cu | SIMT warp-dot | **A** |
| 288–291 | attn_input_proj fp8/qpn、fp8/nvfp4/q4_q5 cutlass | QPN+CUTLASS | **B**(剔除或新实现) |
| 292–296 | gdn_input_proj fp8 qpn + fp8/nvfp4/q4_q5/w8 cutlass | 同上 | **B** |
| 297 | ops/linear/fp8/fp8_volta_qpn_gemm.cu | QPN | **B**(入口) |
| 298 | ops/linear/fp8/fp8_cutlass_sm70.cu | CUTLASS | **B** |
| 299 | ops/linear/fp8/fp8_prepack_sm70.cu | 纯内存布局变换 | **A**(保留进 60,服务后续 QPN 等价布局;或随 B 决策暂删) |
| 300–303 | nvfp4 qpn/mma/cutlass/prepack | QPN+MMA+CUTLASS;prepack 同 299 | **B×3,A×1** |
| 304–305 | q4_volta_mma_gemm.cu、q4_volta_qpn_gemm.cu | m8n8k4 | **B**(入口) |
| 306 | q5_volta_mma_gemm.cu | 同上 | **B** |
| 307–308 | w8_volta_mma_gemm.cu、w8_volta_qpn_gemm.cu | 同上 | **B** |
| 309 | linear_add/q5/q5_linear_add_cutlass_sm70.cu | CUTLASS | **B** |
| 310–315 | linear_swiglu q4/fp8/nvfp4 cutlass+qpn+qpn_split(6 文件) | CUTLASS+QPN | **B** |
| 316 | ops/launcher/gqa_attention_volta_flash.cu | vendored llama.cpp m8n8k4 flash | **B**(新写 Pascal 非-MMA flash;第一阶段从 60 列表不含它,路由关闭) |
| 317 | ops/launcher/vision_attention_volta_flash.cu | 同上 | **B** |
| 318–326 | cutlass 链接、flash 两文件 special flags/include dir | — | arch-60 分支不需要 cutlass(B 新路由若用 cuBLAS 则链 `CUDA::cublas`) |

CUTLASS 依赖结论:P100 不建 CUTLASS TensorOp 内核(arch-60 仅 headers-only 供 gguf/ggml_k 的转换类型);宽 T GEMM 的 B 路由优先手写 HFMA2 SIMT(dequant 融合)与 cuBLAS HGEMM/GemmEx(fp32 计算型)。arch-60 源列表需把上面标 **A** 的 13 个文件(270–284、287、299、303 组的 SIMT 成员)并入——行号随主线程改动漂移,以文件名与性质为准。

## ⑦ 35B 残留几何路由清单(文件:行;主线程已删整包,以下是共享文件里尚未清理的代码级残留)

逐条列出当前树里仍在的 35B-A3B 专属几何/路由/语义(**含注释级**,按"随算子重写逐个处理"口径;勿与本应保留的 27B `W8G32_F16S`/`12288` 形混淆——那是 27B GDN 的正常形状):

| 文件 | 行 | 残留内容 | 处置建议 |
|---|---|---|---|
| src/ops/linear/w8/w8_config.h | 71,85–86,200–202 | `W835bMtpProjectionGeometry`、`kW835bMtpProjection*SmallT` 常量、35B MTP 路由 schedule 别名 | 删(随 w8_small_t 改) |
| src/ops/linear/w8/w8_small_t.cu | 56–58,123–129 | `k35bMtpProjectionLaunchers` 表 + [2048,4096] MTP 匹配分支 | 删分支(27B MTP 形走相邻 case) |
| src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp | 54,59,60,64,86,401,435,444 | `k35Routes` 表(1024/1536/512 × G32)、`is_35` 判定、fused norm-gating 强制 false | 删整块(GATE_UP/GATE_DOWN [1536,2048] 路由属 35B);见② :53/:447 两个 C 站点 |
| src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh | 6,9 | fused norm-gating 内核的 35B 语义注释(内核本体 `>=800` trap,D) | 随上删或注释更新 |
| src/ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp | 32,152,167 | a3b 直路由 `throw` 文案与注释(#else 分支) | 更新文案(Pascal/86 均不可达) |
| src/ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.h/.cu | .h:13,15,16;.cu:27,203 | a3b [12288,2048] G32→qkv/z 几何注释与实例化(文件整体 VOLTA-only,B) | 随 cutlass B 路由决策处理 |
| src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.h | 6,7 | a3b QK+Value 分体 cutlass 注释 | 同上 |
| src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu / plan.cpp | gemm_mma.cu(1 处)、plan.cpp:101,152,195 | a3b 注释/计划文案 | 文案更新 |
| src/ops/gdn_input_proj/w8/w8_gdn_input_decode.cu / gemm_splitk.cu / gemm_mma.cu / kernels.h | 各 1 处(约 57,73 等) | a3b 形状注释 | 文案更新 |
| src/ops/kernel/gqa_attention_geometry.cuh | 27,53,58,72 | `Gqa35Geometry`(16q/2kv D256/R64)定义与 tp2 引用位 | 删几何定义 |
| src/ops/launcher/gqa_geometry_dispatch.cuh | 33,41,43,76 | 35B 几何 dispatch 分支 | 删分支 |
| src/ops/launcher/gqa_attention_decode.cu | 260 | 「Three Q row tiles for the 35B group of eight」Q8 路由块(24/12/6-warp 组) | 删该 else 组(27B 走 24-head 组) |
| src/ops/launcher/gqa_attention_volta_flash.cu | 70,559,563,586,587 | `Gqa35Geometry` 实例化、kTokenMmas=8 特化、meta/launch 的 16q 支 | 删(文件整体属 arch-70/B) |
| src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh | 61,130,220,256,264 | 35B GEMM 形状断言/注释(mma 内核,D trap) | 内核重写时去 |
| src/ops/linear_pair/w8/w8_pair_gemm_mma.cuh | 79,158,172,179 | a3b pair 形状([1024,2048] 等)注释/特化 | 同上 |
| src/ops/linear/fp8/fp8_volta_qpn_gemm.cuh、linear/q5/q5_volta_mma_gemm.cu、linear/w8/w8_dispatch.cpp、linear_add/q5/q5_linear_add_plan.cpp、linear_pair/w8/w8_pair_plan.cpp、linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu / plan.cpp | 各 1 处 | 35B 形状注释/文案 | 文案更新 |
| src/ops/common/volta_mma_splits.h | 15,68 | a3b 形状调参说明 | 文案更新 |
| src/ops/context_kv_materialize/context_kv_materialize.cpp、wrapper/dynamic_grouped_conv.cpp、wrapper/linear_pair.cpp、wrapper/attn_input_proj.cpp:55/225/307、wrapper/gdn_input_proj.cpp:272/409/414、wrapper/linear_add.cpp:44/48、wrapper/linear_swiglu.cpp:58/101/102、wrapper/embedding.cpp 多处 | 见行 | 多为 a3b/35B 文案与 16q 断言注释(注意:12288/G32 命中是 27B 正常形,勿删) | 逐处甄别,只删真 35B 面 |
| src/targets/qwen3_6/impl/runtime/yarn_rope.h | 166 | 「35B-A3B: D128/R128…」对照注释 | 文案更新 |
| src/targets/qwen3_6/impl/frontend/digest.cpp | 16 | **假阳性**(SHA 常量 `0x80deb1fe`… 前一行 `35b01` 十六进制) | 不动 |
| include/ninfer/ops/gdn_gating_proj.h | 56,92 | 35B 融合语义声明注释 | 随算子签名重写 |
| include/ninfer/ops/gdn_input_proj.h | 345(以及 23,29,53,122,153–154,218,229,282–283 中确属 a3b 者) | a3b 语义注释 | 文案更新 |
| include/ninfer/ops/mtp_pack.h | 18 | 35B token 布局注释 | 文案更新 |
| bench/ops/gdn_gating_proj_bench.cu | 4,5,123,145,156,262 | 35B G32 几何 fixture | 删 35B case(主线程已 27B 化,残留待清) |
| bench/ops/gdn_layer_bench.cu | 1,479 | a3b 层形 | 删 |
| bench/ops/rope_bench.cu | 578,581,683,686,689 | D256/R64 35B rope 形 | 删 |
| bench/ops/argmax_bench.cu/l2norm_bench.cu/output_stage_bench.cu/sampling_select_bench.cu/w8_linear_add_bench.cu/w8_linear_swiglu_bench.cu | 各 1(多为首行注释) | 35B 文案 | 文案更新 |
| tests/ops/test_gated_delta_net.cpp | 559–582 | 35B GDN 形用例组 | 删用例 |
| tests/ops/test_rope.cpp / tests/test_rope_cli.cpp | 427–439 / 19–248 | D256/R64 35B 用例 | 删 |
| tests/ops/test_causal_conv1d_silu.cpp | 492 | a3b 文案 | 更新 |
| tests/ops/test_mtp_split.cpp | 2402 | 35B MTP 引用 | 更新 |
| tests/ops/linear/test_w8_a16.cpp 与 pair/swiglu test_common 若干行、test_linear_split.cpp:609–616/814–835、test_mtp/output_head/quantized_weight 等 | 见行 | 多为 27B 正常 `W8G32/12288` 形(假阳性);真 35B 者以 `35b|a3b` 精确串复核 | 逐处甄别 |
| tests/targets/qwen3_6_27b/test_load_plan.cpp:109–110、tests/targets/test_shard_map.cpp:134–135、tests/test_ninfer_artifact_reader.cpp 多处、src/artifact/reader.cpp:108/782、src/artifact/storage_layouts.cpp:53/157–158、src/artifact/typed_binding.cpp:23/78–79、src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}、variant.cpp:623/895 | 见行 | 35B identity/格式/artifact 布局引用(artifact framing 含 35B target_key 枚举等) | 随 artifact 面 27B 化重写 |
| src/targets/qwen3_6/impl/runtime/layouts_impl.h:565/572/575、text_context_impl.h、vision/bindings.cpp:70/144 | 见行 | 35B 布局/vision 文案 | 更新 |

**未找到的项**(主线程消息举例,但当前树中不存在,可能在主线程未提交暂存区或口径更新):`gdn_gateup.cu`、`Q4G64_F16S_35B` token——全库 `*_35B|F16S_35B` 检索为空;请以实际代码为准。

---

### 附:判定依据速查(硬件↔代码事实)

| 事实 | 证据 |
|---|---|
| Ampere PTX 全树集中于 `common/mma.cuh` 且有 `>=750/800/890` trap 保护 | mma.cuh:5–16 注释 + 各 helper 行 |
| `cp.async` 在 `<800` 自动降级同步拷贝 | memory.cuh:35–114 |
| Volta m8n8k4 原语(含 QPN/VoltaMMA/TC-decode/flash 头)全部 `__CUDA_ARCH__ == 700` 等值保护 | volta_mma.cuh:31、各 `*volta*.cuh` 首段 |
| GQA decode `<800` SIMT 分支现成可用 | gqa_attention_decode_bf16.cuh:35/448、decode_i8.cuh:71/627 |
| `__dp4a`/`wmma`/`gguf CUTLASS` 未受保护且无条件编译 | gguf.cu:123/179、ggml_k.cu:412+、src/CMakeLists.txt:66–67 |
| SM8X 站点=mma 内核参数适配,非纯 dequant SIMT | w8_config.h:58/210、*_splitk.cu 各站 |
