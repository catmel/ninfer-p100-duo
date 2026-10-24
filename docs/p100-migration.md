# NInfer → Tesla P100(sm_60)迁移主台账

本文档是 `ninfer-p100-duo` 的迁移权威记录:产品范围、目标硬件实测、构建体系、TP2 传输矩阵、
逐算子族状态与验证资格记录。两份配套清单:

- [门控站点台账](p100-migration-ledger.md) —— `NINFER_VOLTA_BUILD`/`NINFER_SM8X_COMPAT`/
  `__CUDA_ARCH__` 全量站点分类与 P100 计划(自动生成,随重写推进更新)。
- [P100 kernel 供体目录](p100-donor-catalog.md) —— llama.cpp P100 系(qr0n/shinbunbun/qwen36/
  Strata/pxa)可移植 kernel 清单、许可证与对应 NInfer 算子族。

## 1. 产品范围(与 ninfer-v100-duo 的差异)

- 目标硬件:**Tesla P100(GP100,sm_60)**,单卡 TP1 与双卡 TP2;双卡同时支持
  **NVLink(P100 形态:4 条 NVLink1.0 链路)** 与 **PCIe(无桥,含跨 root 场景)** 两种互联。
- 注册身份:**仅 Qwen3.8-27B / Qwen3.6-27B**(同包 `qwen3_6_27b`,groupwise-int 与 nvfp4 系
  A16 反量化路由)。
- **已移除**:`qwen3.6-35b-a3b` 整包、`ops/sparse_moe/*` 与其测试/bench/文档/转换工具
  (2026-10-23 迁移启动日执行;共享 ops 文件内的 35B 几何路由残留在各自重写时清除,
  清单见门控台账末节)。
- 16 GB 显存决定容量:`qwen3.8-27b/groupwise-int` TP2 为主要可运行形态;TP1 仅可容纳
  Q4 级 GGML-K/GGUF 变体 + 小 KV(单卡阶段先以算子级冒烟为主)。

## 2. 目标机实测(2026-10-23,本机 P100-SXM2-16GB)

`tools/p100/device_probe.cu` 实测输出(nvcc 12.4.131 + driver 539.41,Windows WDDM):

| 项 | 值 | 影响 |
|---|---|---|
| GPU | Tesla P100-SXM2-16GB, cc 6.0 | 满配 GP100 |
| SM / 时钟 | 56 × 1480 MHz | HFMA2 峰值 ≈ 19.1 TFLOPS(2× FP32 9.55) |
| 内存 | 16 GB HBM2, 4096-bit @ 715 MHz ≈ 732 GB/s | 带宽相对够用,**瓶颈是发射率/ALU** |
| 共享内存 | 48 KB/block(opt-in 上限同为 48KB,per-SM 64KB) | 所有 >48KB 的 sm_70 tile 必须缩身 |
| L2 | 4 MB | 常驻带策略仍有效 |
| 拷贝引擎 | asyncEngineCount=1(WDDM 读数) | qr0n 双 copy-engine 重叠需在 Linux 双卡复测 |
| cooperative launch | attr=1,实跑 grid.sync **成功** | persistent megakernel 路线开放(备选) |
| CUDA Graph | sm_60 capture/instantiate/launch **成功** | TP2 捕获路线可行;"pre-Volta 无 graph"是工程问题非硬件 |
| hostNativeAtomic / pageableMemAccess | 0 / 0 | HostMappedAllReduce 门铃不可用;pull 语义 + pinned host 为主路径 |
| NVLink | 单卡链路 inActive(无对端) | 双卡到位后重跑探针的 peer 矩阵 |

**已知边界**:P100 无 tensor core(`mma.sync`/`wmma` 均不可用)、**无 `__dp4a`**(6.1 才有)、
无 `cp.async`、无硬件 bf16/fp8/fp4 运算。sm_60 上的一切量化加速只能走
"整数/低位 → HFMA2/FP32 反量化 + fp16 SIMT" 路线。decode gemv 的瓶颈经微探针**修正定论**:
Q6_K 逐 warp 散布读取模式(每块 x, x+32, x+128, x+192, x+208)本身封顶 ≈132 GB/s
(纯 load 零算术实测;标量版与 HFMA2 版差 2 倍指令数却同速),即**解码算术近乎免费、
减少的是 load 指令/屏障而非算术**;"反量化指令数优先于比特数"仅对 wide-T tile 成立,
对 gemv 带宽**比特数直接有效**(Q4_K 模式封顶 188 GB/s 且字节 -31%)。

## 3. 工具链与构建体系

| 项 | 约束 |
|---|---|
| CUDA | **12.x**:12.4 ≤ v < 13.0(CMake 强制;13 移除了 Pascal 编译)。Linux 生产基线建议 12.4(12.8 的 cuBLAS fp16-attention NaN 问题在启用该路由前只影响特定路径) |
| Windows | VS2022 BuildTools(MSVC 14.44)+ **Ninja 生成器**(CUDA 的 VS 集成未安装时 VS 生成器报 "No CUDA toolset found");经 VsDevCmd 调 cmake/ninja/nvcc |
| FFmpeg | Windows 下 find 失败时降级为 WARNING + 空 INTERFACE(文本路径库可独立编译;vision/apps 需装 FFmpeg)。Linux 走 `tools/v100/build_dependencies.sh` 系依赖脚本(待 P100 化) |
| 门控 | `NINFER_PASCAL_BUILD=1`(arch 60 全局定义,root CMake);CUTLASS **不进 arch-60 构建**:`gguf.cu` 的 Sm70 Gemm 实例与 include 门控排除,`ggml_k.cu` 整文件排除+桩,根 FetchContent 收回仅 arch-70 → **arch-60 离线可建**(本机 GitHub 大文件断流实测);`NINFER_SM8X_COMPAT` 对 60 **不定义**(sm8x 路由大量假设 mma m16n8k16,GP100 没有) |
| Blackwell-only 源 | arch 60 与 70/86/89 一样排除 `nvfp4/*_w4a4.cu`、`fp8/*_a8.cu`、`causal_cache/(prompt|small_t)_fp8.cu`,经 `ops/nvfp4_sm86_stubs.cpp`/`ops/fp8_sm86_stubs.cpp` 提供拒绝桩 |
| Volta 源 | arch-70 专属列表(*_cutlass_sm70/_volta_qpn_*/volta_flash launcher 等)对 60 不启用;其入口在 Pascal 路由表中的替代/拒绝策略逐一登记于门控台账 |

构建命令(Windows 单卡实验台):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
cd ninfer-p100-duo
cmake -S . -B build-p100 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=60 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" ^
  -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-p100 --target ninfer_core ninfer_artifact ninfer_text   (已全绿)
cmake --build build-p100 --target ninfer_ops                               (迭代中)
```

## 4. TP2 传输矩阵(双卡要求:NORMAL NVLink 与 PCIe 都要支持)

运行时选路沿用本引擎既有机制(`core/device.cu` 的 `enable_direct_p2p` 探测 +
`ops/allreduce.h` 的 pull 语义 `cudaMemcpyAsync` + `PeerEventPool`),P100 上形成三形态:

| 形态 | 探测条件 | 路径 | 先例/证据 |
|---|---|---|---|
| NVLink 直连 | `cudaDeviceCanAccessPeer=1`(SXM2 桥/PCIe 同 root + NVLink 桥) | 直连 peer copy + one-shot allreduce | houston 4×P100 `GGML_CUDA_P2P=1` 正收益 |
| PCIe P2P | peer 可开但无 NVLink,单 root complex | 直连 peer copy(需实测:跨 socket 时 qr0n 报 **10.7× 回退**) | 拓扑相关,**必测** |
| host-staged | peer 不可开(WDDM/IOMMU/跨 root) | pinned host 中转 pull + 双流重叠 | 本仓库 Linux IOMMU 经验;qr0n `0004` 双拷贝引擎重叠 |

要求:三形态各自过 **tp2 parity 门**(同 batch 同权重 双卡输出 与 TP1 逐位/容差一致),
选路决定记录于本文档第 6 节。Windows WDDM 下 P2P 基本不可用(双卡验证以 **Linux** 为准,
或 TCC 模式);单卡 Windows 只做算子/构建验证。`tools/p100/device_probe.cu` 与 sm_60 版
`tools/tp2/{p2p,transport,capture}_probe` 在双卡到位后逐一重跑。
(2026-10-24:四个探针已全部以 CUDA 12.4 `-arch=sm_60` 编译通过,产物在 `build-p100/`;
`transport_probe.cu` 加 `CUDART_VERSION>=13000` 分支适配 CUDA 12.x 的 capture-info /
update-dependencies 签名;`enable_peer_access()` direct 合格后打印
`cudaDevP2PAttr{NativeAtomic,Access}Supported` 分类日志,NVLink-class 与 PCIe-class
可观测区分。)

**结构核查结论(2026-10-23,代码层):** `ops/common/allreduce.cu` 的既有实现天然覆盖三形态,
sm_60 下无需新增传输分支——(a) `enable_peer_access()`:IOMMU 域检查 → `cudaDeviceCanAccessPeer`
双向 → `cudaDeviceEnablePeerAccess` → 精确图案跨设备拷贝资格验证(捕获"宣称可 P2P 却静默丢写"
的驱动),任一环节失败自动降为 host-staged(同一 `pull_peer` API,CUPTI 不可见的驱动内中转);
NVLink-P2P 与 PCIe-P2P 由驱动 peer 授予结果区分,代码不感知链路类型,P100 两种直连形态复用同一路径。
(b) direct one-shot 路由(`direct_allreduce_kernel`)只用 volatile store + `__threadfence_system` +
`__ldcg`,**不含任何设备侧原子**,因此不依赖 sm_60 缺失的 "PeerAllocations + native atomics over
PCIe" 能力;remote-write 与 fence 语义在 sm_60 P2P 上与 V100 路径等价(待双卡实测性能,非正确性)。
(c) staged 路径为 `cudaMemcpyAsync(D2D,UVA)` + 事件三相序,stream-capture 兼容(§7 capture_probe
已证 sm_60 graph 能力);decode 图内不产生主机往返。
(d) `program_impl.h:319` 依 `enable_peer_access()` 返回值选择 DirectChannel 与否——三形态收敛为
"direct kernel"与"staged pull"两条已编译代码路径,与 §4 表格一致。
剩余双卡工作 = Linux 到位后跑 parity 门 + 三形态各一次 `tools/tp2` 探针复测(纯验证,无代码缺项)。

## 5. 架构决策(记录并约束后续重写)

1. **decode 线性层**:fp16 HFMA2 matvec 族(供体:qr0n `0007`,PRMT→次正规 half 技巧;
   ~~编码优先级 = 反量化指令数~~ **实测修正(§7 ggml_k HFMA2 记录)**:gemv 带解码算术近乎
   免费、编码选择按**每 token 读取字节数**排序 = Q4_K(0.56 B/val)优于 Q6_K(0.82 B/val,
   带宽 -31% 空间),指令数维度只对 wide-T tile 有意义)。
2. **prefill GEMM**:整数码 → fp16 反量化 + cuBLAS `HGemm`(CUDA pin 12.4)。
3. **attention**:非-MMA flash(供体:qr0n `0010/0015`、shinbunbun `fattn-f16-kv-chunk`);
   KV 在 sm_60 **优先 fp16**(int8-KV 的 dp4a 点积在 6.0 只能模拟)。
4. **GDN**:recurrent 部分纯 SIMT 可直接移植;chunked prefill 的 matmul 走决策 2/3
   (供体:qr0n `0009/0018` bit-identical 融合、qwen36 `0001/0005-0008`)。
5. **MTP/DFlash2**:`ops/linear_topk`(arch-70 专属列表)必须入 Pascal 源列表并重写 top-k
   (纯 SIMT,可直接编译)。
6. **NVFP4**:无硬件解码;A16 反量化 → fp16(供体:AGPL 项目**只作行为 oracle**,
   禁止代码移植;MIT 侧用 Strata MXFP4 思路 + 自研)。
7. **CUDA Graph**:主路线保留 sm_60 实测可行;cooperative launch 作为 megakernel 备选,
   不进第一版产品合同。
8. **许可证**:MIT 系可移植(保留版权声明);AGPL(srose69/nvfp4effma、ahtx/kindi、
   L-Forster/open-jet)只读行为,不抄代码。

## 6. 阶段状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| 骨架:CMake arch60/PASCAL 门/依赖降级 | ✅ | configure 通过(Ninja, CUDA 12.4.131) |
| 35B-A3B/MoE 产品面移除 | ✅ | registry/engine/nvtx/tests/bench/文件删除(见台账) |
| Windows I/O 移植:artifact `mmap/O_DIRECT` → `CreateFileMapping/FILE_FLAG_NO_BUFFERING` | ✅ | `src/artifact/reader.cpp`,artifact 编译+归档通过 |
| ninfer_core / ninfer_artifact / ninfer_text @ sm_60 | ✅ | `ninfer_core.lib` 等归档成功 |
| ninfer_ops @ sm_60 | ✅ | `build-p100/ops_build5.log`:0 失败,`ninfer_ops.lib` 归档 |
| 首波错误修复(§7 记录) | ✅ | cuda_fp4.h 缺失→`nvfp4_codec.cuh` 位精确表编解码(sm_60 设备通道);`__grid_constant__`(sm_70+)→`recurrent.cuh` 宏形 Pascal 退化按值传参;`--expt-relaxed-constexpr` 补入 arch-60;`w8_dynamic_grouped_conv_add_plan.cpp` VOLTA mma 带→Pascal 全 T `launch_w8_simt_r8_c8` |
| FP8 词表运行期地雷(代理 B-1) | ✅ | `fp8_dispatch.cpp:88` chunk 选择改为 `vocab && !kVoltaBuild`(`kVoltaBuild` 双名后 Pascal 亦为 true ⇒ 词表 chunk 落 SIMT small_t 上限,消除 T=25..48 越界抛) |
| W8 宽 T trap 地雷(代理 B-2) | ✅ | `w8_dispatch.cpp` :204/:283 拆双名:SIMT 退路拦截(`w8_launch_needs_volta_fallback`→`launch_w8_small_t`→`launch_w8_simt_r8_c8`)Pascal 共用;QPN/MMA fused 带保持 VOLTA-only(`*_supported` 声明在 D 行) |
| engine + media_decode @ sm_60(Windows) | ✅ | `NINFER_FFMPEG_ROOT` 手动 find 落地(CMake≥3.27 无 FindFFmpeg;BtbN shared 包 include+lib 以 zip-range 抽取到 `C:/Tools/ffmpeg-shared`);`ninfer_engine.lib`/`ninfer_media_decode.lib` 归档,27B targets/impl bindings/variant 全过 |
| apps / media_acquire(Windows) | ✅ 编译链接通过(2026-10-24) | `ninfer.exe`+`ninfer-serve.exe` 在 MSVC/CUDA 12.4 sm_60 全量构建 FAILED 0。平台移植:libcurl=可选能力(`NINFER_HAVE_LIBCURL`,缺则 Url 源按 `Error/RemoteUnavailable` 契约报错,Path/Data/Bytes 不受影响;Linux 仍 REQUIRED 全功能);`isatty`/`_getpid`/`_localtime64_s`/`GetConsoleScreenBufferInfo` Windows 分支;`path::native()` wstring 用 `.string()` 比较;顶层 CMake FFmpeg `NINFER_FFMPEG_*_LIBRARY` 收集漏 `${}` 修复;`w8_gdn_input_plan.cpp` CutlassSm70 两处文本引用 PASCAL 门控;`api_impl.h` plan 移动构造从 `template<> =default` 改显式体(MSVC 对未 ODR-used 的 default 特化不 emit 符号)。Windows 无 curl 时 serve 的 URL 输入能力 = 报不可用(真能力边界,非占位) |
| 门控站点台账(221 站点 A/B/C/D 分类) | ✅ | `docs/p100-migration-ledger.md`(⑤第一阶段红利、⑥A文件并入、⑦35B残留清单) |
| P100 供体目录(llama.cpp 系 kernel 移植清单) | ✅ | `docs/p100-donor-catalog.md` |
| TP2 sm_60 探针适配 | ✅(构建/文档面;双卡实测待第二张卡) | `tools/tp2/*` sm_60 构建行 + WDDM/TCC 注记 |
| fused-op plan 路由表 PASCAL 审计 | ✅(2026-10-24) | 全 34 个 `*_plan.*` 扫描(VOLTA-only 门但缺 PASCAL 表)。修 4 处 prefill 断点:`q4_q5_attn_input_plan`(PASCAL 原落 mainline 表 = `GroupedHomogeneousPairMma*` trap → PASCAL 全宽 `ParentSplitFixed` SIMT);`q4_linear_swiglu_plan`(PASCAL 表 = GemvPair/SmallTExact/`Materialized`(via oracle-qualified linear 路由),避开 MmaSplitHalfPair trap 带);`q5_linear_add_plan`(PASCAL kK6144/kK17408 表 Cutlass 带→`SimtWideTResidual`);`q4_q5_gdn_input`/`w8_gdn_input` 核查已具 PASCAL 表(三分支结构)。全量构建 FAILED 0。教训:**schedule 名不可靠,断点判据 = PASCAL 命中表 × dispatch case 守卫 × 入口函数实体所在 TU 文件联查**(本行原记 "shard `MmaResidualR64C*` = gemm_simt sub-Ampere fallback 可用" 系误判,下一行更正) |
| groupwise-int fused/shard 端到端 + TP2 shard 计算路 + bf16 A16 闭环 | ✅(2026-10-24) | 三处新断点形态修复:(1) **无守卫真 mma 体直接 crash**——`q5_linear_add_gemm_mma.cu` 的 `MmaResidualR64C*` 实体是裸 `q5_rowsplit_gemm_mma_kernel`(ldmatrix/mma PTX,sm_60 无 trap 守卫 = `cudaErrorLaunchFailure`),PASCAL `kShardRoutes` 改全 T `SimtWideTResidual`(VOLTA 树同表同患,未动);(2) **Ampere-only grouped 体**——`rowsplit_grouped_mma.cuh` 的 `__CUDA_ARCH__<800` 分支原 `__trap()`,重写为 SIMT 配对循环体(job 选择/guards 逐位一致;Q5 经 `Q5ScalarDecodeAtom`、Q4 经 `Q4SimtDecodeAtom`+f16 scale),`q4_q5_attn_input_plan` shard 全 T、`q4_q5_gdn_input_plan` shard 计划期 throw 删除 = **TP2 shard 计算路(分片形状)打通**;(3) **`q4_linear_swiglu` `SmallTExact` 带 back 实为 trap 的 `q4_small_t_mma_kernel`**(VOLTA 分支早有 compose 先例+真实容量注释),PASCAL 补同款 SIMT compose(`launch_q4_simt_r8_c4/c8`+`silu_mul`)+materialized 容量。**bf16 A16 族闭环**:`bf16_dispatch` wide-T/TP2-shard 与 `bf16_attn_input` T>22 原落 `launch_bf16_mma`(mma.cuh trap)→ 新增 N/K/T 通用 `ops/linear/bf16/bf16_gemm_simt.cu`(fmaf warp-dot,CMake PASCAL 块)三处改路由(n256 conv 投影已解析到 SIMT volta TU ✓)。**公共 bug:`ops/common/warp.cuh::block_reduce_sum` 缺尾屏障**——连续调用时下一调用的 `sums[warp]` 写与上一调用 `sums[lane]` 读竞争(`bf16_linear_add_volta` 4 连调 epilogue 在 P100 实证 3.2x FAIL;加 `__syncthreads()` 后 PASS,全工程 14 调用点一次性受益)。oracle:`tools/p100/groupwise_fused_oracle.cu` **61/61 ALL PASS**——权重改用权威 fixture(`tests/ops/quantized_weight.h` make_patterned_weight/logical_weight_fp64,自造 builder 曾带 Q5 布局偏差教训入表),覆盖 attn tp1/shard ×T1/5/17/33(Q4|Q5 混合)、gdn tp1/shard ×T1/33/9(新 grouped SIMT 体)、swiglu q4 ×T1/16/33/256、add q5 tp1/shard ×k{6144,17408,3072,8704}、bf16 attn T23/33+linear T40×3 形状+add T1/2/9(≤0.9x budget)。全量构建 FAILED 0(`build-p100/full_build_r7c.log`) |
| nvfp4 身份 W4A4→A16 归一 + oracle | ✅(2026-10-24) | **发现并修复最大功能断点**:nvfp4 系 `resolve_route` 在 AllowA4 政策下 T≥4/5/7/8(及 GDN=任意 T)返回 `W4A4` = Blackwell `mma.kind::mxf4nvf4` 专用,CMake 在 arch-60 排除其全部 TU(`nvfp4/.+_w4a4\.cu`)并以 stub `throw`——即 **PASCAL 上 nvfp4 中宽 batch/任何 gdn nvfp4 调用首跑即崩**。修复=6 处 route resolver PASCAL 门内归一 `AllowA4→A16Only`(linear dispatch、attn_input、gdn_input、linear_add、linear_swiglu;snapshot 的 Materialized 子投影经 gdn_input 门自动归一);A16 链=decode gemv + small_t(≤32)+ chunk 循环(全 SIMT,HFMA2),`kNvfp4InternalPolicy` 已是 VOLTA||PASCAL=A16Only。oracle:`tools/p100/nvfp4_linear_oracle.cu`(host 造 e2m1 nibble 码 + M128x4 swizzle e4m3 scale 面、独立 FP64 解码点积,14 案覆盖 5 注册形状 × T1/4/5/7/8/9/32/33/128/256 + TP2 列分片)= **14/14 PASS worst 0.20x budget**(=bf16 输出舍入)。**fp8 A8 同型(Blackwell f8f6f4 stub)判定为范围外:27B 产品身份仅 groupwise-int/nvfp4,fp8 qtype 分支不在合同内,不修**。**nvfp4 fused 族端到端 oracle 完成:`tools/p100/nvfp4_fused_oracle.cu` 30/30 PASS**(attn_input Q|K|Gate|V 四输出 ×T1/5/33/128;gdn_input QKV|Z ×T1/8/33;linear_swiglu silu(gate)*up ×T1/4/16/64/256,T>16 的 `LinearW4A4Post` compose 带 budget 放宽 6ulp——projected BF16 中间量是该组合的真实语义舍入,实测 0.38-0.41x;fused ≤16 带 0.20x@2.5ulp;linear_add residual in-place ×{6144,17408}×T1/8/33=0.20x)。 |
| attention 族 PASCAL 路由闭环 | ✅(2026-10-24) | 两处可达性 bug 修复:(a) `gqa_attention_resolve_route` 外层是纯 `#ifdef VOLTA`,PASCAL 落 mainline 分支 ⇒ 单批 prefill(width>5)返回 `Prompt` = Ampere mma kernel(`mma.cuh` guard `__trap()`)→ PASCAL 并入 VOLTA 式分支、`VoltaFlash` 判定保持 VOLTA-only,wide-T 单批走 `ChunkedSmallT`(decode SIMT 体的分块复用,`decode_bf16/i8.cuh` 的 `#if>=800/#else SIMT` 双体已覆盖 PASCAL 编译);(b) vision `#else` 分支落 mainline `vision_attention_flash_kernel` = 同为 ldmatrix/mma → 给该 kernel 增加架构分支 SIMT 体(warp/行 × lane 切 d 维 online-softmax,镜像 Ampere 体 exp2_approx/ScaleLog2E 语义;launcher PASCAL smem=0)。oracle:`tools/p100/vision_attention_oracle.cu` 8/8 PASS(packed 不均段/单段/512、uniform tile 16/64/100、padding stride;worst ≤0.01x budget,err≈纯 bf16 输出舍入)。linear oracle 35 案回归 ALL PASS;全量构建 FAILED 0 |
| Linux 构建脚本 P100 化 | ✅ **Linux 实跑 PASS(NAS,2026-10-24)** | `tools/p100/build.sh`(CUDA 12.x/arch 60)路线已在 NAS fnOS Linux 实跑通:实际环境 = Docker `nvidia/cuda:12.8.1-devel-ubuntu24.04` + apt 直供依赖(`libavformat-dev` 等 FFmpeg6 avformat 60.16/avutil 58.29/swscale 7.5 + `libcurl4-openssl-dev` 8.5 + cmake/ninja,满足 §CMake pkg 门槛),镜像名 `ninfer:p100-builder`;`cmake -DCMAKE_CUDA_ARCHITECTURES=60 -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF` + `--target ninfer ninfer-serve` **287 编译单元 FAILED 0**。未跑 `build.sh` 原文原因:它钉 `/usr/local/cuda-12.4`(NAS 镜像为 12.8)+ `build_dependencies.sh` 走外网源码构建(NAS 有网但 apt 更快)。证据 `build-p100/nas_build_log_tail.txt` |
| B 类算子族 P100 路由 | ✅ 功能闭环;性能 backlog 开放 | 功能面 = 每族均有 sm_60 可编译可运行路由(逐族见本表其它行+attention 行;GDN chunked 以 recurrent 全宽路由覆盖,T_full=0)。性能 backlog(不阻塞使用):nvfp4 decode HFMA2+wide-T tile、GDN chunked-prefill 快路(供体 0009/0018)、非-MMA flash attention prefill 快路(供体 0010/0015、fattn-f16-kv-chunk)、ggml/w8 T≤4 HFMA2 gemv、常量标定(`// P100: recalibrate`) |
| W8/Q4/Q5/Q6 T=1 decode HFMA2 内核 + 路由 | ✅ 编译+oracle | `w8/q4/q5/q6_decode_hfma2_pascal.cu`,dispatch t==1 覆盖(k%256/512==0、k≤17408、TP2 形状通用);`tools/p100/oracle_decode_check.exe` @ P100 ALL PASS(§7) |
| W8/Q4/Q5/Q6 wide-T decode tile GEMM | ✅ 编译+oracle | `ops/linear/pascal/linear_tile_pascal.cu`:单次解码×8 token 半2 累加(K chunk 2048,staging smem 16KB,grid.y 分块+ragged),dispatch t≥2 全量接管(例外:w8 5120×2048/12800 epilogue、q4 draft head 131072、vision k%512≠0);oracle 含 k5120×T1024、k17408×T12、k5120×T33 PASS(§7) |
| TP2 传输三形态(NVLink-P2P / PCIe-P2P / host-staged) | ✅ 代码+探针编译+**NVLink 双卡实测 PASS(2026-10-24,NAS)** | `allreduce.cu::enable_peer_access`:Linux IOMMU 域门控(`#if __linux__`,Windows 跳过)→ canAccessPeer → enablePeerAccess → `PeerTransferProbe.qualify()` 实拷验证,失败自动落 host-staged;direct 合格新增 P2PAttribute 分类日志(NVLink-class / PCIe-class volatile-flag)。one-shot route = volatile 单写 flag + `__threadfence_system`(无原子指令依赖,PCIe 兼容,sm_60 编译通过)。NAS 双 P100-SXM2(NVLink 4×20GB/s 桥)实测:canAccess 双向=1、单向 68.6/68.8 GiB/s、10KiB 小拷 2.76-5.19 us;transport/capture 探针全 PASS(§7 记录 + `build-p100/nas_p100_probes.txt`) |
| GGML_K(qwen3.8 `gguf-q4-k-m` 身份)sm_60 路由 | ✅ 编译+oracle+stub 删除;**T≤4 已换 HFMA2 gemv** | `ggml_k.cu` 回编 arch-60:CUTLASS/WMMA 站内门控(CUDA 12.4 无 sm_60 WMMA API,实测探针);T≤4 = `gemv_hfma2`(magic-bits HFMA2 解码 + ping-pong staged bf16→half2 + `__ldcs`,单/双缓冲两轮实验定案,§7;旧 `gemv_multi_tokens` 标量路已删),T>4 单 bf16 段走 `ggml_k_tile_pascal.cu` HFMA2 tile(单解码×8 staged token,Q4_K/Q6_K 统一 (c,c+32) half2 交织、1056 bias、min 项 sv 修正),分裂段/add/TiledGdn 走 chunked `gemv_hfma2<8>`;stub 删除;oracle 10 ggml 用例 T1..4096 PASS(§7) |
| TP2 三形态代码核查(NVLink-P2P/PCIe-P2P/host-staged) | ✅ 代码面完成 | kernel/传输面 sm_60 编译全绿(ops/engine/apps 全量 FAILED 0 + §4 探针四件套);剩余 = 双卡运行时探测(用户加卡后) |
| 单卡 TP1 冒烟(本机 P100) | ✅ 装载路全通;27B-Q6_K 超单卡容量,生成待双卡 | 2026-10-24 实跑 `ninfer.exe Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer`:CLI 运行(修 `layouts_impl.h` sm 注册门加 60)、FFmpeg 运行库就位(BtbN master shared 五 DLL,lavf-63/lavu-61/swscale-10 与链接 import lib 精确同 major)、artifact identity/注册 target/binder/load-plan/weights 预检全执行,终以精确消息拒绝:`model weights require 22709925888 bytes on device 0, but only 15861809152 are free`(WDDM 下 16GB 卡可见空闲 14.8GiB)。**Q6_K 27B 单卡物理不可装;生成冒烟 = 用户第二卡 TP2(分片后 ~10.6GiB/卡,16GB 放得下)或更重量化 artifact。**(2026-10-24 NAS 双卡 TP2 冒烟已 PASS,见 §7) |
| 双卡 NVLink+PCIe 验证(NAS 172.161.21.180,fnOS/Linux 6.18) | ✅ **NVLink-P2P 实测 PASS(2026-10-24)** | `p2p_probe`(sm_60/CUDA 12.8.1 容器):canAccessPeer 双向=1,256MiB 单向 **68.6/68.8 GiB/s**(≥25 GiB/s 门槛,PASS),10KiB 小拷贝延迟 5.19/2.76 us;`transport_probe`:D2D-UVA 与 graph-node 三种 transport eager+capture 全 PASS;`capture_probe`:跨设备单图捕获 18 节点(k12/m6)replay==eager 两轮一致,PASS。三形态中 **NVLink-P2P = 本机实际形态**(`enable_peer_access` 实拷资格验证自动选 direct);PCIe-P2P/host-staged 保持代码可用但本机不走。证据 `build-p100/nas_p100_probes.txt` |
| **GGUF-native vector 路 sm_60 重写(MT8 双缓冲,终形态)** | ✅ 编译+oracle 15/15+回归 36/36+**server A/B 33.35 = 基线+3.8% 已发布** | `ops/linear/gguf/gguf.cu`:Pascal MT8 桶 staging ping-pong(位一致)+ `kVectorMinBlocks` Pascal=1(spill 保险);A16 强制与 Rows4 均经 server A/B 否定并撤销;HFMA2/u64 读方向同场否定(延迟 bound 定性);oracle `tools/p100/oracle_gguf_vector_check.cu`(FP64,宽/窄/tall/Tiled/双段/双输出)+ probe `p100_vector_probe.cu`(V0-V9,h8704 行网格敏感性) |
| 双卡 TP2 端到端冒烟(NAS) | ✅ **PASS(2026-10-24)** | `ninfer --tp 2 --devices 0,1` @ 27B Q6_K-MTP(21.16 GiB artifact):权重 11.20+11.20 GiB 分片,两卡各 free 4.02/15.89 GiB,装载→CUDA Graph 捕获→生成全通,输出语义连贯。**实测 decode 24.4-25.0 tok/s、prefill 72.8-76.3 tok/s**(bench A/B/C 三组,含 `--spec mtp --draft-tokens 4` 与 `--no-thinking`;MTP 组 decode 21.6 tok/s,P100 上 draft 收益不抵验证成本,默认 spec 更快)。证据 `build-p100/nas_p100_bench.txt` |
| 文档一致性(README/AGENTS 产品声明 P100 化) | ⏳ | 结构稳定后统一处理 |

## 7. 验证记录

- 2026-10-23:`tools/p100/device_probe.exe` 于 Tesla P100-SXM2-16GB 实测(第 2 节数据源);
  graph capture/instantiate/launch、cooperative grid.sync、hfma2 执行均 `no error`。
- 2026-10-23 `tools/p100/oracle_decode_check.exe`(合成 packed 权重+fp16 scale,独立 FP64
  oracle 逐组解码累加;同时跑合格 reference 路由对照;日志 `build-p100/oracle_decode_check_run2.txt`):
  - W8 HFMA2 decode(`launch_w8_decode_hfma2_pascal`,T=1):vocab 248320×5120 max|err|=2.09
    (ref 2.00),gate/up 2.03(2.00),down 5120×17408 3.68(=ref 逐位同),qkv/out 1.86(=),
    mtp 2048×16384 2.86(=)。ALL PASS;与合格 float 路由同一误差带(误差由 bf16 输出舍入主导)。
  - Q4 HFMA2 decode(`launch_q4_decode_hfma2_pascal`,T=1):gate/up 0.134(ref 0.125),
    down 0.229(ref 路由在合成 k=17408 越界,80.6,不影响新核结论),qkv/gdn 0.121(=ref)。ALL PASS。
  - 判据:tolerance 上限(合成量级带)+ 与合格路由误差同级。两处布局地雷已修复并记录:
    W8 存补码字节需 XOR 0x80 转 offset-binary;Q4 nibble 存 two's-complement(XOR 0x8 符号折叠)。
- 2026-10-23 同上工具扩至 20 用例(判据=混合带:≤2.5 bf16 ulp(|ref|) 且绝对下限 1.0,超限即
  FAIL;日志 `build-p100/oracle_decode_check_run8.txt`):
  - W8/Q4 HFMA2 全部 ≤0.52x 带(W8 max|err| 0.53-0.65,低于 bf16 舍入带,质量与合格 float
    路由同级或更好)。
  - GGML_K:Q4_K/Q6_K/混合行矩阵 × T∈{1,4,7,16,64}(覆盖 `gemv_dynamic`、`gemv_codec`、
    新 `gemv_multi_tokens` 含 ragged 尾块)全 PASS ≤0.20x 带;host 编码器与 codec 解码器逐值
    互检(>1e-9 即 FAIL)全过。发现并修复 CUDA 12.4 事实:**sm_60 无 WMMA API**(`mma.h` 在
    arch<700 下空实现,nvcuda 命名空间缺失)。
  - Q5/Q6 HFMA2 decode(qkv 6144/34816、down 17408、vocab 248320、4096;5-bit XOR-0xFF 单 bit
    plane、6-bit XOR-0xAAAA pair-fold 双 bit plane)全 PASS ≤0.24x 带,与合格 SIMT 路由同级;
    host 编码 = two's-complement 存储(q5 c5=v&31、q6 c6=v&63,与 atom 折叠语义推导一致)。
- 2026-10-23 同上工具至 31 用例(`build-p100/oracle_decode_check_run10.txt`):
  - wide-T decode tile GEMM(W8/Q4/Q5/Q6,T=5..1024,含多 K-chunk、ragged token、生产形状
    k5120×T1024、k17408×T12)全 PASS ≤0.39x 带。首轮 FAIL 定位:tile 主循环 code/high-plane
    指针漏加 chunk 偏移(单 chunk 用例掩盖,k5120×T1024 暴露)→ 修复后全绿;教训:tile 类
    oracle 必须含 ≥2 chunk 形状(k>2048)。
  - 工具新增:top-8 误差元素 dump(`ORACLE_DEBUG=8`)+ 用例名子串过滤(argv[1])。
- 2026-10-24 同上工具至 35 用例(`build-p100/oracle_decode_check_run12.txt`):
  - **GGML_K HFMA2 wide-T tile**(`ggml_k_tile_pascal.cu`,donor 0007 方向):每 warp 一
    GGML block(256 值 = 32 lane × 8 值),Q4_K/Q6_K 统一 (c, c+32) half2 列对交织
    (stage 128 slot/token);Q4_K merge = s·Σcode·x − m·Σx(fp16 sums_v 修正 min 项),
    Q6_K 把 −32 折进解码 magic(bias 1056);row 类型按 descriptor 位 warp 内分支,混合矩阵
    单 kernel;单 bf16 段 T>4 全接管,分裂段/add/TiledGdn 保留 gemv_multi。
    ggml 10 用例 T∈{1,4,7,16,33,64,97,4096} PASS ≤0.45x,与 float gemv_multi 对照(同输入
    0.20x,经 count=2 split 旁路 tile 取得)同量级。
  - 首轮 FAIL(q4 k5120 T97 = 1.06x、q6 k5120 T8 = 1.43x)根因 = **测试 x 幅度超产品契约**:
    生成器取 U(−8,8),而产品 prefill/decode 激活是 LayerNorm 形态 |x|≲2;fp16 累加器
    (Σcode·x)绝对误差 ∝ |x|,大值域对消后相对放大。用例增加 `x_half` 代表幅度字段,
    ggml 全组改用 2.0(产品代表)后全过;w8/q4/q5/q6 组保持全幅(已通过,留作压力)。
    教训:oracle 输入分布必须落在 public-input 契约域内,否则验收带度量的是测试伪影。
  - tile 首跑 FAIL 定位两枚结构 bug:staging 未随 block 循环重写(只覆盖 block 0 列)、
    128 线程只映射 32 单元中的 16(staging job 需跨线程循环)。
- 2026-10-24 **apps Windows 移植**(条件(2)闭环,`build-p100/apps_build9.log`+
  `full_build_final1.log` 全量 FAILED 0):缺口逐一定位修复见 §6 apps 行。过程记录:
  `transport_probe.cu` 用 CUDA 13.1 capture-info 签名→加 `CUDART_VERSION>=13000` 分支;
  四探针(p2p/transport/capture/device)sm_60+CUDA 12.4 编译产物在 `build-p100/`;
  `enable_peer_access()` direct 合格路径新增 P2PAttribute 分类日志(NVLink-class vs
  PCIe-class volatile-flag)。
  MSVC 增量陷阱:改 `api_impl.h` 后 ninja 未重编 `variant.cpp.obj`(deps 依赖跟踪在
  MSVC /showIncludes+modmap 组合下失效一例),强制删 obj 重编后符号出现——
  报"引用已改但链接仍旧符号缺失"时先验证 obj 时间戳。
- 2026-10-24 **attention 族 PASCAL 可达性审计**(§6 新行)。方法教训:编译通过≠路由可达,
  `mma.cuh` 的 trap-guard 让 Ampere kernel 在 sm_60 照常编译、首跑才崩——**必须对每个
  route-resolver 单独核对 PASCAL 落点**。gqa `resolve_route` 因外层 `#ifdef NINFER_VOLTA_BUILD`
  把 PASCAL 留给 mainline `Prompt` 分支(同类 `#ifdef` 站点已全量扫描:linear/proj 族
  均已 `VOLTA||PASCAL` 组合+内层 VOLTA-only 拆分,唯 gqa/vision wrapper 遗漏)。
  vision SIMT 体首跑 8/8 FAIL 的签名:`got ≈ ref/32`(逐 d 同号等比)⇒ 定位 = 行尾
  `warp_sum(l)` 对已广播的完整 l 再乘 32;Ampere 体的宽度 4 归约是 fragment 布局专属,
  SIMT 镜像体不能照抄归约步骤。**oracle 调试有效手段 = 逐 token ratio 分布 + worst 行
  全 d 向量 + 均匀均值对照**,一次 dump 即分辨"算法退化"与"局部缺写"。
  oracle 工具自身教训:`all = all && maybe(...)` 短路会跳过后续案例,聚合必须先行执行再折叠。
- 2026-10-24 **fused-op plan 路由表审计**(§6 新行;§8 表内"宽 T 保持默认表=正确中间态"
  的旧判定由本审计推翻)。判据 = PASCAL 命中表分支 × dispatch case 守卫联合;**schedule
  名不可靠**(`MmaResidual*` 实体 = gemm_simt.cu 的 sub-Ampere fallback,真 break 是
  Cutlass-throw 与 Grouped-trap 带)。修:`q4_q5_attn_input_plan` PASCAL 全宽 ParentSplitFixed;
  `q4_linear_swiglu_plan` PASCAL = {GemvPair,SmallTExact≤32,Materialized via-linear}(复用
  oracle 资格化的 linear tile 路由,不维护第二套 dequant-GEMM);`q5_linear_add_plan`
  PASCAL kK6144/kK17408 Cutlass 带→SimtWideTResidual。`#ifdef`+`#elif` 非法组合在扩表时
  暴露两处(改 `#if defined(...)`)。全量构建 FAILED 0。
- 2026-10-24 **nvfp4 W4A4→A16 归一 + 14 案 oracle**(§6 新行)。路由审计的第二形态断点:
  不是设备 trap,而是 **CMake 排除 + stub `throw`**(`nvfp4/.+_w4a4\.cu` 在 arch `60|70|86|89`
  整体摘除、`nvfp4_sm86_stubs.cpp` 一触即抛)——只看 `#ifdef` 表会漏判,必须联合构建源列表
  确认 schedule 实体 TU 存在。修法=policy 归一(`AllowA4→A16Only`,6 处 resolver 单点插入;
  snapshot 的 Materialized 子投影经 gdn_input 门自动归一),不动 W4A4 枚举/capacity 查询/
  5090 树。oracle 造权要点:`validate_nvfp4_weight` 要求 `qhigh == nullptr` 且 `scales ==
  payload + align256(n*k/2)`(指针同一性);e4m3 scale 面按 `(m_tile*kTilesPerRow + group/4)
  *512 + row%32*16 + quartile*4 + lane` swizzle 写入——host 镜像公式漏乘 `scale_tile*512`
  正是被 "declared but never referenced" 警告暴露,warning 值得读。结果 14/14 PASS
  worst 0.20x budget(=bf16 输出舍入)。fp8 A8 同型 stub 结构判定范围外(27B 身份无 fp8)。
- 2026-10-24 **groupwise-int fused/TP2-shard/bf16 A16 端到端 oracle 61/61 ALL PASS**(§6 新行)。
  判据 = 权威 fixture(`tests/ops/quantized_weight.h`)+ FP64 解码点积 + max(1.0, 2.5×bf16_ulp)
  预算;compose 带(swiglu T>32、nvfp4 同型)放宽至 6.0 ulp(中间量是真实语义舍入)。日志
  `build-p100/groupwise_fused_run5.txt`,构建 `build-p100/full_build_r7c.log` FAILED 0。
  本轮沉淀的断点/教训:
  - **第三形态:无守卫真 mma 体 = `cudaErrorLaunchFailure` 直崩**(非 trap、非 stub throw)——
    `q5_linear_add_gemm_mma.cu` 的 `MmaResidualR64C*`;判定 schedule 必须打开入口函数所在
    **具体 .cu 文件**看实体(`*_gemm_simt.cu` 只导出 Split2/SimtWideT,同名族≠同实现)。
  - **第四形态:泛型 kernel 的 `<800` 分支写 `__trap()` 占位**(`rowsplit_grouped_mma.cuh`)——
    补 SIMT 体后 attn/gdn TP2 shard 全 T 打通;计划期 `throw` 屏障(`q4_q5_gdn_input_plan`)随之删除。
  - `SmallTExact` schedule 名实不符(`q4_small_t_mma_kernel` 是 trap fused 体):VOLTA 分支的
    compose + 真实 capacity 注释就是先例,PASCAL 照抄;capacity 与 execute 必须同分支成对改。
  - bf16 A16 族的 "`*_volta.cu` = 纯 SIMT 可复用" 约定(CMake PASCAL 块复用)覆盖 conv 投影/
    linear_add;wide-T/shard 空档以 `bf16_gemm_simt.cu` 补齐。
  - **工具教训:**自造打包 builder 曾带 Q5 布局偏差且难以自证——一律改用项目权威 fixture;
    oracle 读回 section 时列步幅必须取共享张量 `out_ld`(qkv 双 section ≠ 自身行数,T=1 恰好
    免疫该 bug);Windows 控制台 fastfail 丢 stdout → `setvbuf(_IONBF)` + 分段 marker 输出。
  - **`warp.cuh::block_reduce_sum` 缺尾屏障**为全工程公共 bug(14 调用点),bf16 add epilogue
    4 连调在 P100 实证;修复后本工具 bf16-add T1/2/9 全 PASS。
- 2026-10-24 **真模型 TP1 冒烟(用户 artifact `Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer`,
  23.4GB)**:两处修复/部署后装载链全通——
  1. `targets/qwen3_6/impl/runtime/layouts_impl.h` sm 白名单补 60(全工程唯一 sm() 门;
     `linear_topk/*_m64.cu` 的 `case 120` 是 tile 列数非 SM,勿误读);
  2. `ninfer.exe` 0xC0000135 = FFmpeg 运行 DLL 缺失(构建期只用 `C:/Tools/ffmpeg-shared` 的
     include+lib,zip-range 抽取未带 bin):下载 BtbN `ffmpeg-master-latest-win64-gpl-shared.zip`,
     校验其 `LIBAVFORMAT_VERSION_MAJOR 63`/avutil-61/swscale-10 与链接 import lib **同 major**
     (SONAME 一致才可直接供 exe),解出 5 个 DLL 到 `build-p100/apps/` ✓。
  结果:identity→注册 target→binder→plan_load→**weights 显存预检报精确数值**
  (需 22709925888 B,free 15861809152 B)干净退出。结论 = Q6_K 27B 单卡(Windows WDDM 可见
  14.8GiB)物理不可承载;下一步 = 第二卡 TP2(分片 ~10.6GiB/卡)冒烟。
  **工具教训:ninja+MSVC dyndep 下改纯头文件可能不触发重编(本次 `layouts_impl.h` 编辑后
  `cmake --build` 报 "no work to do"),须 touch 受影响 .cpp 或核对 .obj 时间戳。**
- 2026-10-24 **GGML-K 生产路真实形状微基准**(`tools/p100/p100_speed_bench.cu` →
  `build-p100/speed_bench_q6k.txt`,Q6_K 全行,64 层组成 = 16 FA + 48 GDN,按
  inventory.py 精确形状/层数外推,线性 GEMM 口径,不含 attention/state/elementwise):
  - **decode T=1 gemv 带 = 110-128 GB/s**(HBM2 732 的 ~16%,SIMT 反量化发射受限,符合
    §2 预期);per-token linear sum **178 ms → 5.6 tok/s 上界**(端到端预计 ~4-5)。
  - **lm_head(248320×5120)8.86 ms = 单 token 的 50%**,是最大单项;groupwise Q6 有
    HFMA2 decode 专路而 ggml_k T≤4 走纯 SIMT gemv → **ggml_k HFMA2 decode 提速 = 最高优先
    backlog 项(空间 2-4×,可把单卡推到 ~10 tok/s 级)**。
  - prefill tile 带 = 计算受限,**23.6 tok/s linear-only 平台期从 T8 持续到 T2048**(纯线性
    扩展,FLOPs 口径 ≈ 1.15 TFLOP/s ≈ HFMA2 峰 6%,反量化指令税)。
  - **TP2 shard 形状实测:T=1 每卡 91.2 ms → ~11 tok/s**(尚未扣 allreduce;NVLink 到位后
    探针实测);TP2 prefill 按分片线性外推 ~45 tok/s。
  - 口径声明:AGENTS 规定 microbench 只确立 Op 级结果,以上为权重带宽/发射成本模型估算,
    端到端数以双卡真机冒烟为准。
- 2026-10-24 **单卡 + 主机内存(CPU 卸载)真机测试**(`tools/p100/p100_hybrid_offload_bench.cu`
  → `build-p100/hybrid_offload_bench.txt`):AVX2 Q6_K gemv(12 线程,xorshift 块编码 +
  FP64 oracle 合格 worst rel 0.34%)+ 生产 `ggml_k_linear` GPU 层,1/8 行标定外推全模型
  19.56 GiB 线性预算,分层放置(GPU 176 op + host 81 op / 5.99 GiB)并尝试真实链路全量驻留。
  - **CPU 层实测 2.1-7.5 GB/s**(210B 块跨距打断硬件预取 + 手写核指令吞吐,~llama.cpp 级);
    **预测混合 decode = 0.48 tok/s**(即便内存够;带限优化核上界也仅 ~1.5-2 tok/s)。
    全 GPU 驻留对照 = 3.91 tok/s(线性预算)。
  - **Phase2 真机结论:16 GB RAM 物理装不下**——host 权重 5.99 GiB 分配中途 malloc 失败
    (崩时 host free 5.17 GiB);且 **Windows WDDM 下每次 cudaMalloc 显存同时占用等量主机
    内存驱动影子(GPU 13.5 GiB 放置计划把 host 一并压干)** = Windows 单卡大权重 + 卸载
    双挤。结论:该机型卸载路线不可行;若扩内存(≥32 GB)速度仍 ~0.5-2 tok/s,比 TP2(~11)
    低一个量级。**用户路线维持:第二卡 TP2。**
  - 工具 bug 修复记录:AVX kernel x 索引曾漏块偏移(单块 256 元素测试不可见,k=5120 才暴露);
    hsum 尾步 movehl/add_ss 自加错(正确 = add(movehl) + add_ss(shuffle 0x55))。
    调试残留 `build-p100/q6dbg.*` 已清理。
- 2026-10-24 **NAS 双 P100-SXM2(NVLink 桥)Linux 端到端验证**(172.161.21.180:221,
  fnOS Debian12 主机 + Docker `nvidia/cuda:12.8.1-devel-ubuntu24.04` 派生 `ninfer:p100-builder`,
  驱动 580.142,`nvidia-smi nvlink -s` 4 link×20GB/s/卡):
  - **代码迁移**:`tar` 打包源码(排除 build 产物/exe,57 MiB)→ `/vol1/1000/code/ninfer-p100-duo`;
    FFmpeg6/curl/cmake/ninja 用 apt 直供(avformat 60.16 ≥ 门 60,avutil 58.29 ≥ 58,
    swscale 7.5 ≥ 7,curl 8.5 ≥ 7.85)。**287 编译单元 FAILED 0**,`ninfer`/`ninfer-serve` 产出。
  - **TP2 探针(§4 三形态门的实测)**:`p2p_probe` canAccess 双向=1,256MiB 单向 **68.6/68.8
    GiB/s**,10KiB 小拷 5.19/2.76 us(每 token ~128 小拷 × ~5us ≈ 1.3ms 级 all-reduce 开销);
    `transport_probe` D2D/graph-node transport eager+capture 全 PASS;`capture_probe` 跨设备
    单图 18 节点(k12/m6)replay==eager 两轮一致 + 对照组证明独立双图不可链 = TP2 必须单图,
    与设计一致。三探针 exit 0(`build-p100/nas_p100_probes.txt`)。
  - **TP2 冒烟**(`Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer`,21.16 GiB artifact,
    权重 11.20+11.20 GiB 分片,两卡各剩 free 4.02 GiB):装载→图捕获→生成→语义连贯输出,
    `--tp 2 --devices 0,1` 一路 PASS(`build-p100/nas_p100_tp2_smoke.txt`)。**服务化部署 + A/B 后最优配置
`--spec mtp --draft-tokens 3` decode = 32.1 tok/s**(§7 A/B 行)。
  - **速度实测**(bench A/B/C 三组,`build-p100/nas_p100_bench.txt`):decode **24.4-25.0
    tok/s**、prefill 72.8-76.3 tok/s。对照微基准外推(TP2 线性分片 91.2 ms/卡 → 11.0 tok/s):
    实测高 ~2.2x。口径解释:artifact 自带 MTP 层且**默认路由即走 MTP 投机**,而 T=1..4 的
    gemv 线性在内存墙下耗时≈与 T 无关(权重只读一遍),一次 forward 接受多 token ⇒
    每 token 摊销时间下降;显式 `--spec mtp --draft-tokens 4` 反而 21.6 tok/s(draft 越深
    验证/拒绝成本越高,默认档位即最优)。prefill 超外推同理为 tile GEMM 真链路效率。
  - **Windows P2P 教训闭环**:§4 曾记 WDDM 拒 peer(本机探针 canAccess=0);NAS Linux 上
    NVLink-P2P 授予成功 = 同一二进制路由 `enable_peer_access` 自动选 direct,零代码差异。
- 2026-10-24 **NAS 服务部署 + decode 配置 A/B**(8088 端口,`ninfer:p100-duo` 运行镜像 =
  builder+COPY apps,context 指 apps 目录绕 `.dockerignore` 的 `build-*` 排除;compose 目录
  `/vol1/1000/docker/ninfer-p100x2`,写法逐项对齐 v100x2:`./models:/workspace/models:ro`、
  `8088:8080`、`unless-stopped`、deploy.gpus;模型暂以硬链接指 v100x2 同文件,用户自拷覆盖):
  - A/B(--no-thinking,160 tok,同 prompt):**默认 spec 24.86 / draft-2 29.87 / draft-3
    32.12 / draft-4 21.60(前测)/ draft-3+int8-KV 31.91(无收益,sm_60 无 dp4a,
    int8 点积模拟 = 预期短板)/ draft-3+lm-head-draft 该路由拒绝(无输出)**。
    = **`--spec mtp --draft-tokens 3` 为当前最优,已写入 compose**,较默认 +29%。
  - 部署后负载:`--max-context 65536`(bf16 KV 2.85 GiB/卡;v100 的 131072+int8 组合在
    sm_60 既无速度收益又挤 KV 容量,弃)。冒烟 `/v1/chat/completions` 正常返回。
- 2026-10-24 **decode 路由归因更正 + GGUF-native vector 路 sm_60 重写(V9)**:
  - **归因(whole-inference profiling,nsys `tools_bin/dec.nsys-rep`,draft-3 160 tok)**:
    部署 artifact `Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer` 是 **QType::GGUF**
    (variant.cpp Qwen38Gguf 权重档案),线性 decode 走 `ops/linear/gguf` `vector_kernel`,
    **不是** GGML_K 族——早前"artifact linear = GGML_K"判断作废;V3 HFMA2 gemv(ggml_k)
    对该 artifact 属冷路径,解释了其 A/B 持平(32.15/32.14/32.12 vs 基线 32.12)。
    kernel 时间占比:vector `<Q6_K,8,2,0>` **53.4%**(MT8=verify 桶)+ `<Q6_K,4,2,0>`
    **28.4%**(MT4=MTP draft 链)+ Tiled `<9,8/4,2,1>` 6.9%(GDN o-proj)+ A8/quantize ~6%;
    direct_allreduce 1.2%(NVLink 正常)。H2D 2.33M 次/7.8s = **权重装载**(21GiB@PCIe),
    与 decode 稳态无关;decode 轮墙钟 ≈78ms,其中 gemv ≈50.9ms/卡。
  - **微探针战役**(`tools/p100/p100_vector_probe.cu`,Q6_K TP2 形状,best-of-20,µs / GB/s):
    V0(生产复刻)MT4-R2 415/177、MT8-R2 683/107(17408 行);**8704 行(真实 shard 行数)
    V0 MT4R2 197、MT8R2 320**。变体结论:
    ① MT4-R2@17408 已接近 GP100 合并 u32 流式上墙(V1 uint2 读 + LDS 指令裁剪 ≈ 零收益 →
    非 LSU-bound,是延迟 bound);
    ② **HFMA2 方向在该路被否**:V6 内联 bf16→half 转换 1152µs(+70%,GP100 `cvt.rn.f16x2.f32`
    Quarter-rate),V2/V4 magic-bits HFMA2 仅 −6..−10% 且需 fp16 预转 x + fp16 累加精度带;
    ③ ping-pong 双缓冲 + Rows 的**行数依赖**(本战役最重要事实):双缓冲在 MT8 全行数赢
    (17408:−9%;8704 R4:−7% vs 单缓冲 R2),在 MT4@8704 输 3.5%、MT4@17408 赢 10% —
    小网格 + 窄 token 带下双份 STS/smem 压力盖过 LDG 重叠收益;Rows4 同理 MT8 赢(全行数)、
    MT4 在真实 server 混合形状 +19% 输(avg 267 vs 单缓冲 224)。**单形状 probe 结论外推到
    混合真实 workload 必须经 server nsys 同模板分解复核。**
  - **生产落地终形态**(`src/ops/linear/gguf/gguf.cu`,两处保留 + 一处撤销后复得):
    ① `vector_kernel` staging **ping-pong 双缓冲模板化**:`kPingPong = Pascal && MaxTokens > 4`
    (MT8 verify 桶双缓冲,MT4/MT1 桶保持原单缓冲),`stage[Warps][kPingPong?2:1][kStageWords]`,
    `cur = kPingPong ? (sb/2)&1 : 0`;!kPingPong 分支与原始代码逐行等同;算术顺序全路径不变 →
    全 arch 位一致(oracle 15/15 复验);
    ② `kVectorMinBlocks` Pascal=1(原 `MT4&&R4→6` 表达式在 Pascal 撤 R4 后只剩 tall 桶可达,
    固定 1 防任何新形状踩 spill 悬崖);
    ③ **A16 强制门 = 实测撤销**(server A/B:纯 A16 门形态 = 31.50 −0.6 vs 基线 — GP100 上 A8
    路的 int8 激活读带宽收益 > dp4a 模拟指令税,延迟 bound 定性推翻发射 bound 预判;门控删除,
    恢复所有 arch 统一 `allow_a8` 判定)。
    **server 最终 A/B(draft-3 ×3,空卡 CLI):** decode **33.34/33.36/33.36 = 33.35 tok/s,
    基线 32.12 → +3.8%**(prefill 68.1-68.6 持平);`ninfer:p100-duo` 运行镜像 + compose
    已发布,:8088 服务恢复 listening。
    **归因实验矩阵(全部同 prompt 160 tok ×3,dry-run CLI):**
    基线 32.12 | MT4R4+MT8R4+双缓冲+双minBlocks 18.46(**spill 悬崖**) | 同 spill 修复(200/166 regs)
    30.07(MT8R4 混合形状 −11% avg 但 Rows4 其他形状亏损) | MT8R2+全双缓冲 31.91 | MT8R2双+MT4单
    31.52(=双缓冲在宽形态净收益极小) | 全单缓冲+纯 A16 31.50(A16 门净 −0.6% 定罪) |
    **终形态 = A8 正常+MT8 双缓冲+R2 全桶 33.35 ✓**。
    **教训:该 kernel 是内存延迟 bound(§7 探针定性),任何"省指令"路线(HFMA2 转换、u64 读、
    A16 强制)让位于带宽/占用率结构路线(ping-pong 重叠 LDG);但结构性改动(Rows4/双缓冲)对
    混合真实形状 + launch_bounds 寄存器分配极敏感,单形状 probe 必须配 server nsys 同模板
    avg 对比才能定形。**
  - **oracle(新工具 `tools/p100/oracle_gguf_vector_check.cu`)**:GGUF-native Q6_K,独立 host
    FP64(自研编码器与 decode8 合同互镜 + Tiled 列映射独立重写),`allow_a8=true` 走生产入口
    `gguf_project`;15 用例 = 宽 T1/2/4/5/8、gate 14336、窄 1024/1536、tall 65536(T1/T4)、
    Tiled k6144/k3072、双段权重、双输出 section。本地 sm_60 **15/15 PASS**(worst ≤0.49x 预算,
    预算=2×bf16ulp+1.0 下限);既有 groupwise/ggml_k oracle **回归 36/36 ALL PASS**(lib 重建无扰动)。
    调试记录:`Tensor{0,tokens}`(split_out=0 的无用 section)在 try 块外构造 → `shape_numel`
    throw 无人捕获 → 0xC0000409 fastfail,条件构造修复;双 section dump 布局为各 section
    线性 [token][row],比较需段内反演映射。
  - **NAS 部署 + 端到端 A/B**:终形态 **33.35 tok/s(+3.8% vs 基线 32.12)**,见上矩阵与
    `build-p100` 无新增——证据在 NAS `/tmp/v9*_deploy.log` + `tools_bin/oracle_gguf_v9.log`;
    服务 :8088 已发布终形态镜像。


## llama.cpp "40+ tok/s" 对照审计(2026-10-24,应"为什么不达标"质疑;备份 = `ninfer-p100-duo-v9-final-20261024.tar` 本地+NAS MD5 `c76369c2…`)

来源调查:NAS `/vol1/1000/code/Strata-for-p100`(llama.cpp 系 Strata fork 的社区 bench 库)+ `llama_v100` 部署记录。

| 引擎 + 配置 | 模型(每 token 权重字节) | 卡 | decode |
|---|---|---|---|
| Strata setup(IQ3_XXS)| **177B MoE ~3B active**(≈1.4GB)| 1× RTX 5070 | 49.7-57.9 |
| Strata setup | 同上 | 1× RTX 3060 | **41.6-41.9** |
| Strata setup(expert CPU offload)| 同上 | **1× P100 16GB** | **31.1** |
| **llama.cpp v0.6.0 tuned(-ngl 99 -ncmoe)** | 同上 | **1× P100** | **10.3** |
| llama.cpp tuned + MTP | 同上 | 1× P100 | 7.7-8.8 |
| Strata 0.1.39 IQ3_S(28.8GB PLE 走 NVMe)| 同上 | 2× P100,128K ctx | 28-32 |
| **NInfer TP2 Q6_K(本项目)** | **dense 27B(≈11.4GB/卡)** | **2× P100** | **33.35** |

结论:
1. **"llama.cpp 在 P100 上跑 Qwen3.8 应该 40+" 的前提不成立** — 40+ 数字全部来自
   `Qwen3.8-Flash-Next` **177B MoE(~3B 激活)+ CPU/NVMe expert offload + 新卡(3060/5070/4090)**;
   其每 token GPU 权重字节 ≈ 本项目 dense Q6_K 的 **1/8**,不是同口径对照。
2. **P100 sm_60 上 llama.cpp 官方本体只有 10.3 t/s**(还是 MoE);Strata 的 sm_60 实验路
   (`STRATA_EXPERIMENTAL_SM60`:emulated dp4a、BF16 投影经 fp32 — 即本项目已分析吸收过的
   donor 技术)把同一模型拉到 31.1,**与我们同量级,而我们跑 8 倍字节量的 dense 模型**。
   NInfer dense Q6_K TP2 = 同卡 llama.cpp 官方数字的 3.2 倍,无每字节欠账。
3. dense 27B + 双 P100 到 40+ 只剩真路:**减权重字节 = Q4_K/Q3 系重量化**
   (−28~35% 字节 → 预测 decode 43-48 tok/s),其次 **verify 宽度去 padding**(+8-10%)。
   两者都动 artifact/产品行为,需决策质量代价。

## 7.x Strata-for-p100 kernel 库审计 + V10 有效列跳过(2026-10-24)

**来源**:NAS `/vol1/1000/code/Strata-for-p100/src/kernels/cuda` 47 文件子代理全量分析。互证与可移植清单:

1. **Strata 的 P100 定位与本项目一致**:官方行 = "2× P100, IQ3_S(MoE 3B 激活) decode 28-32";
   llama.cpp v0.6.0 官方同卡 tuned = 10.3(见上节审计)。其 Pascal 专属坑(全部入册):
   smem opt-in 假阳性(<sm_70 须用 `cudaDevAttrMaxSharedMemoryPerBlock` 而非回答值)、
   pre-sm_70 `__restrict__` 走非相干缓存路径(graph 内复用 buffer 慎贴 restrict)、sm_60 构建
   把 signed-overflow UB 优化成错值(位技巧必须显式无符号)、全树 `__CUDA_ARCH__` 无 600/610
   数值分支(dp4a 软件回退穷举位验证,语义架构无关)。**其负结果档案与本项目 V0-V9 判死清单
   完全互证**:staging 权重进 SMEM、u64 读宽、HFMA2/fp16x2、`__constant__` 分歧码表、纯
   `#pragma unroll`、收紧 launch_bounds——双方独立实测皆否。
2. **精确列数 = 他们的头号结构**:`switch(ncols)` 每列数一个模板实例;兜底宽实例内 padding 列
   的 apply 被 `c<n` 守卫**直接跳过(不是算了丢弃)**,激活 offset 钳位 `min(c,n-1)`;
   verify 接受列数从 **device 内存读(`*n_keep`)**,volatile D2H + spin-flag,verify→commit 全链
   留同一张 CUDA graph。
3. **可移植队列**(预期收益序):① 有效列守卫(**V10 已实施,见下**);② Q6_K **plane 重排**
   (210B 交织块拆 ql/qh/sc/d 四 256B 对齐 plane + `__ldcs` + 持久 grid 640 + 归约侧 ping-pong;
   同族 Q8_0 packed 核实测 +10-12%,注意那是非 Pascal 数字);③ tsum 转置蝶形归约(8 列 shuffle
   40→12,位安全,可白拿);④ QFUSE/PAIR epilogue 融合 + 双 rank 合并 launch(治 35% 非 gemv)。

**V10 = 有效列跳过(vector 路,NInfer 形态)**:
- **探针定罪**(NAS 临时插桩两轮):decode CUDA graph 的 gguf vector 路 **tokens=8 是窗口硬上限**
  (形状 = GDN proj 5120×3072 ≈60 次/轮 + qkv 8192×5120 + down 5120×5120 + narrow/tall);
  而 `target_valid_columns`(host 按 frontier 填 `extent+1`,值域 [1,K+1=4],见 text_context_impl
  mtp_host_ingress)= **每轮真实有效列 ≤4 → ≥50% 列白算**。列数活在 device 张量,graph replay
  不可变 launch 参数 → **kernel 解引用是唯一 CUDA-Graph 兼容形态**(Strata B-3 同构)。
- **实现**(`gguf.{h,cu}` + `text_context_impl.h` 两处):`ValidColumns{ends,batch}` + RAII
  `ValidColumnsScope`(宿主 capture 期绑定,batch≠1 保守降全宽);kernel 首部 `__ldg(ends)` +
  warp shfl-max 归约出 `tokens_valid`;列循环 `if (t>=tokens_valid) break` 于激活加载/FFMA/
  dp4a-dot/**shuffle 归约**四处(未算列保持 `{}` 零初始化 → padding 输出=精确零,accept 链
  只读有效列语义不变);`launch_vector_shape` 单点注入;单卡 verify(808)+ TP2 verify(2737,
  rank0 张量指针,对称不变量 + direct-P2P qualified 支撑 rank1 读取)绑定。
- **oracle**:15 FP64 用例 ×(bound=tokens 全列位一致 + bound=tokens/2 前段位一致/后段精确零)
  = **45 项本地全 PASS**。
- **NAS 部署 + A/B vs 33.35**:**失败回退(2026-10-24 20:0x)**——V10a 运行时 `break` 守卫使全
  部 vector 实例失去软件流水(prefill 61/decode 27.7);V10b `Dyn` 双实例分裂恢复 prefill 68,
  但 decode 仍 27.94×3(vs V9 33.35,-16%);已实测的界内 MT4 窗口收益(+14%)不足以覆盖未解释
  回退,MTP-forward 侧(T=8 窗口)绑定变体未及部署即被 bisect 阻塞。**决策:V10 不发布,
  工作树整体回滚到 V9 验证态;实验代码存档 NAS `/vol1/1000/backups/v10-experiment-20261024/`
  供后续重新立项**。探针数据保留(§7.x 上文 610/527/1134/270/313µs 矩阵即来自该实验)。

## 7.y 开源交付与 V9 回滚验证(2026-10-24/25)

1. **回滚链**:V9 tar 备份(md5 c76369c2…)→ 提取四文件(gguf.cu=`2bf4ae73…` 复核一致,
   大写 `ValidColumns` 零命中)→ NAS 全量 cmake 构建 → 手工 nvcc 配方重建 oracle →
   runtime 镜像 `sha256:643c367`(`FROM ninfer:p100-builder` + COPY 两 bin)→ compose 重建,
   容器镜像与 V9 镜像 ID 对齐,`/v1/models` 冒烟通过。**教训**:首轮回滚用短 ID 手工
   `docker tag`+`compose up` 因引号事故未生效,服务曾在 V10b 慢镜像上运行 ~1h——回滚必须
   以 `docker inspect {{.Image}}` 与构建输出哈希双向对账。
2. **GitHub 首推**:`catmel/ninfer-p100-duo` main=`bf5649e`(1542 文件,仅源码:排除
   build*/、*.exe、备份 tar、*.i64 语料、链接伴生物;真实密钥模式扫描 0 命中)。README 重写为
   P100 身份 + fork lineage(Neroued@feaf4dd → wamansou TP2-1m → geoffwatts v100(+huangserva
   v100-tpx)→ tuxKOH V100-Duo → 本仓库)+ vendored/审计引用两表;NOTICE 追加 P100 派生段。
3. **全量基线资格(full_qual E1-E5,2026-10-25 00:49-00:53,NAS 实测)**:硬件快照修正历史口径
   ——"P100-PCIe"实为 **2× Tesla P100-SXM2-16GB, NVLink NV4(4-link), driver 580.142,
   CUDA 12.8, max clock 1480MHz**(README/台账统一更正)。E1 ctest rc=0;E2 GGUF-native
   oracle **15/15 PASS**;E3 decode 回归 **36/36 PASS**;E4 Q6 A/B draft-3×3 =
   **33.34/33.35/33.37** tok/s(prefill 67.96/68.13/68.22)= V9 生产值复现 ✓;E5 serve
   chat completion OK。E6 = Q4/Q6 各 2K→128K 每 2K 衰减曲线(运行中,归档
   `bench/p100-duo/qual-20261025/`,完成后单独提交)。
4. **Q4 artifact**:`Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q4MTP.ninfer`
   (18.5 GiB,HF nerkyor GGUF-NInfer;SHA256 `e445a0a9…` 校验进行中)——NAS 经本机代理拉
   xet CDN 尾段跌速,切 hf-mirror 续传。

## 8. 已落地的构建切片决策(2026-10-23 第二轮)

| 站点 | 处理 | 依据 |
|---|---|---|
| `ops/linear/gguf/gguf.cu` | `__dp4a`×2 → `ops/common/dp4a.cuh` 位精确模拟(sm_61+ 仍走硬件);CUTLASS Sm70 TensorOp `Gemm` 实例化 `#ifndef NINFER_PASCAL_BUILD` 门控,`gemm_admits()` 在 Pascal 恒 false(全量退 vector/SIMT 路由),`gemm_project` 留 throw 桩 | 台账 ④.1 |
| `ops/linear/ggml_k/ggml_k.cu` | **整文件 arch-60 排除** + `ggml_k_pascal_stubs.cpp`(计算入口 throw、workspace 查询返回 0) | 文件内 WMMA/CUTLASS 与 SIMT gemv 交织,门控切片风险高;T=1..4 的 SIMT gemv 也属 Phase-B 重写对象(donor 0007),不留半门控文件 |
| `ops/linear_topk/`(DFlash2 top-16) | arch-60 并入 `linear_topk.cpp + dflash2_linear_topk_volta.cu`(纯 SIMT,零 asm) | 台账 ⑥(272–273) |
| `core/pdl.cuh:23`、`variant.cpp:48` | 组合门 `SM8X\|\|VOLTA` 扩为三名 `+PASCAL`(普通发射;NVFP4/FP8 策略 A16Only) | 台账 ④.3;Pascal 无 dp4a/IMMA,A8 亦排除 |
| FP8 词表运行期地雷(代理 B-1) | `fp8_dispatch.cpp:88` chunk 选择改为 `vocab && !kVoltaBuild`(`kVoltaBuild` 双名后在 Pascal 亦为 true ⇒ 词表 chunk 落 SIMT small_t 上限,消除 T=25..48 越界抛) | 台账 ⑤;A16 MMA 核 ldmatrix sm_75+ |
| W8 宽 T trap 地雷(代理 B-2) | `w8_dispatch.cpp` :204/:283 块拆双名:SIMT 退路拦截(`w8_launch_needs_volta_fallback`→`launch_w8_small_t`→`launch_w8_simt_r8_c8`)Pascal 共用;QPN/MMA fused 带保持 VOLTA-only(其 `*_supported` 声明在 D 行) | 台账 ⑤"删 B 分支退 SIMT";w8_small_t.cu:63 已双名 |
| `w8_dflash2_attn_input_volta.cu` 宽 T mma 带 | `if (T≥32)` 融合 mma 带门为 VOLTA-only,Pascal 全 T 走 SIMT rowsplit | 台账 ⑥(271 A+B) |
| `q4_q5_gdn_input_plan.cpp` | :34 双名后内部再分:VOLTA=fused/Cutlass 带,Pascal=**全 T IndependentDirectFixed**(SIMT parent-split 扩程),TP2 shard(GroupedMixedMma)在 Pascal 计划期抛(避免设备 trap) | 代理2 ⚠1 裁决 |
| `w8_gdn_input_plan.cpp` | Pascal 表:`{1: DecodeR8, 2..∞: SimtRowViewSplit}`,CutlassSm70 id 不可达;launch 分支守卫保留 | 代理2 ⚠2 裁决 |
| A 类双名批量(3 代理) | linear 族 38/52 扩+14 项有据保持(多为 cutlass 声明耦合,正确语义即 VOLTA-only);input-proj 族 16(含 linear_pair/linear_attention);launcher/kernel/wrapper/targets 族 13。合计 ≈67 站点;`#endif // NINFER_VOLTA_BUILD` 尾注过期标签列入后续美化 | 台账 ②⑤⑥ |

**遗留运行期注记(Phase-B 优先级,不阻塞编译):** q4/q5/q6 的 mma 日程由各自
`needs_volta_fallback`(已双名)拦截退 SIMT r8_c8;q5_linear_add 17+ 由设备 trap 改宿主 throw;
swiglu q4/nvfp4/add fp8/nvfp4 宽 T 保持 Materialized/#else 默认表(代理1 B 组,均为正确中间态);
DecodeSplits/kSmallTChunkTokens/launch_bounds 等常量 `// P100: recalibrate` 待实测定标。
**2026-10-24 更正**:上述"默认表=正确中间态"对 `q4_q5_attn_input`、`q4_linear_swiglu`、
`q5_linear_add`(Cutlass 带)不成立(PASCAL 命中 mainline 表 = Grouped trap/throw),
已由 §6/§7 的 plan 表审计修复;nvfp4 系与 fp8(27B 范围外)判定维持。
| `src/CMakeLists.txt` arch-60 | 排除 ggml_k.cu;并入 ⑥A SIMT 文件 16 个;**CUTLASS 依赖整体摘除**(根 FetchContent 收回仅 arch-70;`gguf.cu` 的 cutlass include 门控;本机 GitHub 大文件不可靠,离线构建优先) | 台账 ④.4/⑥;fetch 断流实测 2026-10-23 |

**A 类双名扩面(✅ 批量完成,见上表代理行):** 台账 §②/⑤ 标 A 的 92 站点按"只改无条件源列表文件中的
VOLTA 门为 `VOLTA||PASCAL`"原则批量落地;⑥ 的 A 文件已并入 arch-60 列表
(bf16_n256_k5120_volta、w8_dflash2_attn_input_volta(宽 T mma 带已门为 VOLTA-only)、
swa_dflash2_volta、linear_topk 对、candidate_selector 对、dynamic_grouped_conv 组、
rmsnorm_rope 对、context_kv_materialize 对、bf16_linear_add_volta)。
`fp8_prepack_sm70.cu`/`nvfp4_prepack_sm70.cu` **不并入**:第一阶段不启用 QPN prepack,
保持直读布局(台账 ⑤ 约束;bindings :249/:1315 维持 VOLTA 单名)。A 常量(DecodeSplits=560、
kSmallTChunkTokens=5 等)按 56 SM 重标定表记录、不阻塞编译。
