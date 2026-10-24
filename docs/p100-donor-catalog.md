# NInfer → Tesla P100 (sm_60/GP100) 迁移:CUDA Kernel 供体目录

> 盘点范围:`C:\Users\yaoml\Desktop\code\llama_cpp\` 下 5 个本地开源项目(p100-llama-cpp-main、
> llama-cpp-p100-patches-main、qwen36-p100-patches-main、Strata-for-p100-0.1.41 + p100-strata-main、
> 以及各 README 里的上游仓库线索)。目标:为 NInfer(Qwen3.8-27B dense 混合架构:全注意力 +
> gated delta net 线性注意力 + SwiGLU MLP + MTP/DFlash2 投机 + TP2)在 sm_60 上逐算子族找可移植供体。
> 编制日期 2026-10-23,全部基于本地文件实测记录,未联网核对。

## 0. 硬件事实基线(供体文档中反复引用的前提)

| 事实 | 数值/结论 | 出处 |
|---|---|---|
| tensor core / wmma / mma.sync / cp.async / `__nanosleep` | sm_60 全部不可用(7.0+/8.0+ 特性) | 各 README 一致 |
| `__dp4a` | **6.1 起**才有;GP100 (6.0) 走标量乘法模拟,实测仅 3.95 TOP/s | QR `docs/kv-cache-type.md` `bench/sweep.cu` |
| HFMA2(half2 FMA) | **满速**,15.80 TOP/s;FP32 9.5 TFLOPS,FP16 峰值 19.1 TFLOPS(1328MHz)/18.7(1266MHz) | QR README、`bench/sweep.cu` |
| GP100 vs sm_61(P40/GTX10xx) | GP100:HFMA2 满速、无 dp4a;sm_61:有 dp4a、半速 F16(5.6 TFLOPS)。**两者最优路径相反** | QR README"GP100 and the other Pascals are not the same machine" |
| 指令速率(GP100) | LOP3/HFMA2 满速;PRMT、SHR、BFE **半速**;I2F/F2F 1/4 速;64 位整数除法无硬件(极慢) | QR docs/fp16-mmvq.md、SB patches.md 01/12/27 |
| 共享内存 | 48 KB/block 静态上限(实测 CUDA 12.4+ 拒绝 >48KB 静态;56KB、93.7KB 的变体直接不可编译/不可用) | QR patch 0013、ST `gdn_chunk_scan_kernel` 注释 |
| L2 / DRAM | L2 4MB;16B 合并读可达 608 GB/s(读主导 ~570);跨 UPI 双路机 P2P 是 10.7x 灾难 | QR `docs/investigation.md`、`docs/prefill-and-multi-gpu.md` |
| 调度特性 | GP100 无独立线程调度:跨 warp spin 死锁风险,只有 warp-内 butterfly 安全;每 SM 2 个 16512 warp 槽,48KB/block 时 56 SM 只能驻留 56 个 1024 线程块 | QR 0012/0017 提交说明 |
| 工具链 | CUDA 13 移除 Pascal;CUDA 12.x + gcc ≤13;CUDA 12.8 的 cuBLAS fp16 长上下文 GEMM attention 出 NaN(12.4 正常) | SB README、QR issue #4 |

## 1. 供体项目总览

| 代号 | 本地路径(绝对) | 上游 | 基线 | 许可证 | 规模 |
|---|---|---|---|---|---|
| **QR** | `C:\Users\yaoml\Desktop\code\llama_cpp\p100-llama-cpp-main` | qr0n/p100-llama-cpp | llama.cpp **b10660**(commit 6c84c7d)+ patches 0001–0018,已应用在 `QR\llama.cpp\` 内 | MIT(© 2026 qr0n;底层 ggml MIT) | 18 patch + 10 篇 docs,逐 patch 实测 |
| **SB** | `C:\Users\yaoml\Desktop\code\llama_cpp\llama-cpp-p100-patches-main` | shinbunbun/llama-cpp-p100-patches | llama.cpp **v0.5.0**(历经 b10133→v0.2.0→v0.4.0→v0.5.0 rebase) | MIT(© 2026 shinbunbun) | 45 patch(01–48,缺 03/43),`docs\patches.md` 全索引含被否决尝试 |
| **Q36** | `C:\Users\yaoml\Desktop\code\llama_cpp\qwen36-p100-patches-main` | (Mrinohk,Qwen3.6 P100 patches) | 上游 llama.cpp `c21284cdf` + SB 的 17 个 patch 作基线 | MIT | 35 patch:fp16 matvec、persistent megakernel、MoE GEMM、GDN prefill |
| **ST** | `C:\Users\yaoml\Desktop\code\llama_cpp\Strata-for-p100-0.1.41` | Niko1221/Strata 0.1.41 | 独立 CUDA/HIP/SYCL 引擎(Qwen3.8-Flash-Next MoE),`-DSTRATA_EXPERIMENTAL_SM60=ON` 可跑 sm_60 | MIT(© 2026 Niko1221;llama.cpp 衍生文件带 ggml MIT 头,pin 在 commit 3cf03257) | `src\kernels\cuda\*.cu` 43 文件 + `src\prefill\*`,含 parity/基准 harness |
| **SP** | `C:\Users\yaoml\Desktop\code\llama_cpp\p100-strata-main` | qr0n,对 Strata 上游 `99f3dbd` 的 1 个 patch | 同 ST | MIT | MXFP4 dot/dequant + Q5_1 + PLE 驻留;2×P100 实机数据 |

表内路径缩写:**QR/SB/Q36/ST/SP** = 上表绝对路径;`llama.cpp/ggml/src/ggml-cuda/…` 一律相对各仓库根;QR 的应用后源码在 `QR\llama.cpp\…`(可直接读),其余项目只给 `.patch` 文件。

---

## 2. 算子族 ①:decode 线性层 matvec(T=1..8,含 MTP 多列)

> NInfer 需求:Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS/IQ3_* 权重 × fp16/fp32 激活,T=1 单序列
> decode、T=2..8 MTP/DFlash2 验证与并发批。

### 2.1 核心 fp16/HFMA2 matvec(最强供体群)

| 条目 | 源文件与关键 kernel | 数据类型/形状 | 技术要点 | 实测收益 | 移植障碍 |
|---|---|---|---|---|---|
| **QR-0007 fp16 GP100 matvec**(首选) | `QR\patches\0007-…fp16-matrix-vector-path-for-batch-1-8.patch`;应用后 `QR\llama.cpp\ggml\src\ggml-cuda\mmvq-gp100.cu`:`gp100_mmvq_kq(_nc)`、`gp100_mmvq_iq4_xs(_nc)`、`gp100_mmvq_q6_K(_nc)`、`gp100_mmvq_q3_K`、`gp100_mmvq_b32`(Q8_0/IQ4_NL)、`gp100_prep_act`、`gp100_rms_norm_mul_prep` | Q4_K/Q5_K/Q6_K/Q3_K/IQ4_XS/IQ4_NL/Q8_0;T=1(K 量化 4 行/warp)与 2..8 列(`_nc` 多变体) | ① 激活一次性转 fp16 并按 32-block 2 的幂预缩放(不可能溢出,块和精确 fp32 保留);② 权重用 **PRMT 直接拼出次正规 half 位型** `q*2^-24`,HFMA2 对次正规满速(实测 14.98 vs 14.96 TFLOP/s),免 magic-number 减法;③ 单 sub-block 内 fp16 累加(16 项/lane 有界)后升 fp32;偏移(−32/−4/+128)用块和在 fp32 里剥除;④ src1 激活缓存 + 融合单行 RMS_NORM×w 直写缓存 | 2×P100 Qwen3.8-27B:tg128 **25.04→33.22**;MTP(draft3,真实 prompt 集)**26.15→40.91 tok/s**;1×P100 Llama-3.1-8B Q4_K_M tg **39.3→79.6(2.03x)**;Q4_K 517 GB/s、Q5_K 427、IQ4_XS 346;KLD 反而**优于**被替换的 q8_1 路径(RMS 误差 1.1e-3 vs 1.2e-2) | 整文件仅 `__CUDA_ARCH__==600` 编 device code,其他 arch `NO_DEVICE_CODE`;依赖 PRMT(半速,已计入);`GGML_CUDA_GP100_MMVQ_DISABLE` 等开关 |
| **QR-0008 split-K 短矩阵** | `QR\patches\0008-…split-K-over-the-block-s.patch`,同文件:`gp100_store_ksplit`、`GP100_LAUNCH_B32(type,KS=4)` | Q8_0/IQ4_NL,nrows<2048(ssm_alpha 24×5120 等),T=1 | 块内 4 warp 各算 1/4 条 K 切片(串行链 160 块→40 块深度),`__shared__ float red[4][RT]` 合并;2048+ 行仍 KS=1 字节不变 | 24 行 alpha 投影 5.7→3.0 us(190 GB/s→360);30.1ms/token 里约 −0.28ms | 仅 sm_60 编译;smem 用量小(4×R float) |
| **QR-0011 批处理 decode** | `QR\patches\0011-…batched-decode-several-sequences-one-toke.patch`(mmvq-gp100.cu 多列、gated_norm_prep ≤8 行) | 2..8 列(并发序列或 MTP);Q6_K 2–4 列、Q4/Q5 2–8 列 | 把 `[K,1,n_seqs]` 激活当 n 列喂多列 kernel;Q6_K 列 epi 简化为 2 FMUL+4 FFMA;**无条件(clamp)激活载入**让编译器跨行共享列激活(104→60 loads/iter,单步最大收益) | 4 并发 agent 聚合 58.7→**72.9 tok/s**(4 seq: 68.0→86.8,+28%);单序列不变 | 同 QR-0007;R=8 实测 −10%(寄存器 >250) |
| **QR-0016 多列 Q3_K + 大矩阵 Q8_0** | `QR\patches\0016-…multi-column-Q3_K-and-large-matrix-Q8_0-M.patch`:`gp100_mmvq_q3_K_nc`、`gp100_mmvq_b32_nc` | Q3_K 2..8 列(MTP draft head 124160 行)、Q8_0 ≥2048 行 2..8 列 | Q3_K 每行 32 权重解包一次、全列复用;Q8_0 走 b32 staging | Q3_K MTP head 每步 0.83→0.48 ms;4 seq 聚合 86.8→88.3(+1.7%);KLD −ub5 0.011278 | 同 QR-0007 |
| **Q36-0001 fp16 decode matvec + 融合 norm** | `Q36\patches\0001-…fp16-decode-matvec…patch`:`ggml/src/ggml-cuda/mmvq-p60.cu`(`mmvq_p60`、`mmvq_p60_prep_x`、p60_x 激活缓存) | Q4_K/Q5_K/Q8_0,T=1 | half2 数学;融合 RMS_NORM 直接产出 fp16 激活布局(省 prep kernel);opt-in `GGML_CUDA_P60_MMVQ=1` | 基线数据见 results/stock-vs-patched:35B-A3B decode tg128 68→125.7(配合后续 patch) | 与 QR-0007 同思路的独立实现,可交叉验证 |
| **Q36-0013 fp16 多列长行** | `Q36\patches\0013-…fp16-multi-column-matvec…patch`:`mmvq_p60_mc`、`mmvq_p60_prep_x_wide`(两遍 prep) | Q4_K/Q5_K,**2..3 列**且 K≥5120、K%1024==0(MTP 验证) | 0x6400 魔数解 nibble 成 half2;K 切成 16-slot 段、每段 2 lane,列激活放寄存器,warp 走 8 行,段间 smem 合并 | Q5_K T3 27.7→20.6 ms、Q4_K T3 21.4→14.9;2×P100 MTP **24.03→26.27 t/s(+9.3%)**;KLD 0.00046 | 与 QR `_nc` 系列互为替代实现 |
| **Q36-0002/0003/0004** | 同仓库 patches:swiglu 融合 prep、残差 ADD 折进 fused rms_norm、fused norm 直写 q8_1 | T=1 为主;SMoE 分支 | 激活准备链融合(Q36 侧对 QR-0007 prep-cache 思路的独立扩展) | 汇入 2x 总收益 | — |

### 2.2 通用 MMVQ 参数/调度修正(不给新 kernel,但收益大且全类型适用)

| 条目 | 源文件与关键符号 | 覆盖 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0001 q8_1 块和复用** | `QR\patches\0001-…q8_1-block-sum….patch`,`vec_dot_q4_K_q8_1_impl_vmmq` | 全部 q8_1 激活点积,所有 arch | 直接读存储的 `ds.y`,省 4 dp4a/调用,148→138 指令/16 权重 | +4% tg(GP100);非 bit-identical 但与 MMQ 同款近似 | 通用;若走 fp16 路径则被 0007 取代 |
| **QR-0002 GP100 参数表** | `QR\patches\0002-…GP100-MMVQ-parameter-table….patch`,`MMVQ_PARAMETERS_PASCAL`,`calc_rows_per_block(type,…)` | `__CUDA_ARCH__==600` 专属 | K 量化 4 行/block、其余 2;行/block 扫描 1/2/4/8 = 40.8/48.3/**52.2**/51.3 | 1×P100 8B **+32.9%**;2×P100 27B +17% | sm_61 必须保持原表(半速 F16),别照搬 |
| **SB-02 行/块(pre-Turing)** | `SB\patches\02-mmvq-rows-per-block-sm60.patch`(mmvq.cu) | pre-Turing,ncols=1 | 同上思路,瓶颈是激活行 L2 重读 | llama-bench tg32 **+23.0%**(Q4_0 9B) | 与 QR-0002 二选一 |
| **SB-09 小 K nwarps** | `SB\patches\09-mmvq-nwarps-small-K-sm60…`;`calc_nwarps` 由 host 传 | IQ3_XXS/Q3_K/IQ2_XS 小 K(K=512 时 128 线程只用到 16) | 按 K 精确给 warp 数:IQ3_XXS 64.0→36.6us、Q3_K 106.6→48.3us | MoE decode +1.29%,dense 噪声内 | need=3 时非 bit-identical(2 warp 换序) |
| **SB-10 q8_1 激活缓存** | `SB\patches\10-mmvq-q8-1-activation-cache.patch` | ncols=1 全类型 | 同一 src1 一个图内只 quantize 一次;key 含 data 指针/stream/通道 | MoE +1.17%、dense +0.88%;但缓存改变分配布局会让融合决策翻转→输出漂移,须按下游要求决定去留 | **非 bit-identical** 风险记录充分 |
| **SB-36 大批次分块走 MMVQ** | `SB\patches\36-mmvq-chunk-large-batch.patch` | 8<ne11≤32 的量化 mul_mat | 不落入 cuBLAS 整权重 f16 展开(248320×2560 Q6_K LM head 要 1212.5 MiB 常驻池) | 省 ~1.1 GiB 常驻;速度 ±0.5% | 价值在显存而非速度;对 NInfer 27B 词表头直接适用 |
| **SB-29/30 码本上共享内存**(通用技巧) | `SB\patches\29-mmvq-iq3xxs-grid-smem.patch`、`30-mmvq-ksigns-smem.patch`:`iq3xxs_grid_smem_init`、`ksigns_smem_init/apply_ksigns` | IQ3_XXS/IQ2_XS/IQ2_S(dot 路径),T=1..3 | 发散索引的 `__constant__` 码本 32 路序列化:**1KiB 表搬 smem,隔离基准 2.965→1.824 us/Mw(nc=1,−38%),constant 反而 6.486(2.2x 慢)**;ksigns 用 smem 查表替 `__vcmpne4/__vsub4` 模拟(每 4 权重 8.6→2.1 条 SASS) | 27B IQ3_XXS 模型 decode +3.1–7.6%(短上下文),IQ2_XS 模型 +1.5–1.8%;全 bit-identical | 每 block +1KiB smem(48KB 预算内);`*_smem_init` 含 `__syncthreads` 不可放 early-return 后;2KiB–8KiB 的其他表没做(占用率权衡不同) |

### 2.3 精确 int8 语义路线(bit-exact,给 EXACT/MTP 一致性用)

| 条目 | 源文件与 kernel | 覆盖 | 要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **SB-01 vmad 模拟 dp4a** | `SB\patches\01-vmad-dp4a-sm60.patch`(`common.cuh` 的 `ggml_cuda_dp4a` 600 分支) | 一切 int8×int8 点积,所有 shape | 4 条 PTX `vmad.s32.s32.s32` 带字节选择子,无字节抽取;15→6.75 条 SASS | decode **+6.5–6.8%**,数值逐位相同 | sm_61 有真 dp4a 时勿用(4 VMAD+IMAD vs 1 DP4A);GP100 上 SIMT 点积仍只 ~8 TOP/s |
| **ST `include\strata\kernels\dp4a.hpp`** | `strata_dp4a`(同上 vmad,带 1.07e9 case 位精确校验)、`strata_spin_pause`(pre-Volta 无 `__nanosleep` 的空转) | 全部 i-quant/K-quant dot 调用点 | **一个头文件同时解决 sm_60 两大指令缺口**,所有 Strata kernel 经它调用 | 隔离测 2.2x 提速 | 无;直接可抄(MIT) |
| **Q36-0010/0014 staged-load Q8_0/Q6_K matvec** | `Q36\patches\0010-…Q8_0-matvec…`、`0014-…Q6_K-matvec…`:`mul_mat_vec_q8_0_staged`、`mul_mat_vec_q6_K_staged` | Q8_0(T=1,K=2048/4096)、Q6_K(T=1..3) | 2 字节对齐块(34B/210B)导致逐 16 位碎读:保持 stock 线程映射/累加序,把 4 行块经 **smem 暂存 + 合并 16B/8B 载入 + 下一块与计算重叠**;位精确 | Q8_0 17.8MB 47.2→36.3us(→463-490 GB/s);Q6_K 输出头 4.08→3.02 ms;tg256 **+7.3%**、MTP +5.3%,输出逐字节相同 | **指令缓存教训**:同 kernel 的 fp16 版 17KB 码体让邻接 kernel 慢 0.43ms/token(5.4/9.4KB 无此害);staged 版码体小是设计目标 |
| **Q36-0032/0033/0034/0035 EXACT 路线** | patches 0032–0035:`p60-mega-x.cu`(`mul_mat_vec_q4_K_megax` 等 warp 模拟 stock dot)、chunk4 MMVQ、FA pad-to-256 | T=1..6 全部生成步 | 用 persistent kernel 复刻 stock 逐位语义(位相同于单 token decode),MTP 输出==非投机;v2 共享权重行跨 token,用等价但更廉价的整数部分绕开 I2F | MoE half T=1 113us(stock ~135);T=6 ~496us;**投机输出与不开投机逐 token 相同** | 受模拟 dp4a 指令速率约束;价值在正确性而非极致速度 |

### 2.4 persistent megakernel(launch 风暴的另一解)

| 条目 | 源文件与 kernel | 覆盖 | 要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **Q36-0018/0019 层半 megakernel** | `Q36\patches\0018/0019…`:`ggml/src/ggml-cuda/p60-mega.cu`、`p60-mega-t.cu`、`p60-mt.cuh` | MoE/GDN/SwiGLU 每半个层一次 cooperative launch,T=1 与 T=2..6;fp16 激活 smem 预置换 | 软件 grid barrier(计数器跨 launch 递增,非 cooperative API);匹配失败自动回退;**思路源自 open-jet(AGPL)——只可参考行为,不可抄码** | 35B-A3B 无投机 85.6→**116.7 t/s**;MTP greedy 122→157.6(+29%) | fp32/fp16 激活路径非位精确(KLD≈0.004);0035 INVAR 开关统一 T=1..6 kernel 族消除"验证≠单解码"漂移 |
| **Q36-0020 T=2..6 输出层** | `p60-mega-t.cu` q8_0_t(K=2048 稠密 q8_0 多列,x smem 2 幂 fp16,行按 1088B 流水) | Q8_0 词表头 2..6 列 | 输出层 248k 行 @6 列 2829→1686 us | — | 与 QR-0016 b32_nc 重叠,选一并存 |
| **ST `native_mmvq.cu`(122KB)** | `native_quantize_q8_1_kernel`、`native_swiglu_quantize_q8_1_kernel`(SwiGLU×q8_1 融合)、`native_{q4_k,q5_k,q6_k,q3_k,q2_0,iq4_xs}_mmvq_kernel`、`native_small_mmvq_kernel`、`native_mmvq_multi_kernel`(多列)、`native_mmvq_il_kernel`+`native_q8_1_interleave_kernel`(交错布局位精确多列,T=2..4)、`native_mmvq_wave_kernel`、`native_q8_0_packed_kernel/native_q6_k_packed_kernel`(预打包 SoA 权重、16B 对齐路径)、`s26_tsum.cuh`(IQ 点积的 `sumu=sum(quant)` 寄存器内前缀和技巧) | 全部 K-quant/IQ4_XS/IQ4_NL/Q8_0/Q2_0/Q4_0..Q5_1;T=1..4 | llama.cpp MMVQ 的独立移植(带 MIT 头,pin commit 3cf03257),经 `STRATA_DP4A`;IL/wave/packed 是超出 llama.cpp 的多列布局实验;`docs/MMVQ_IL_TABLE.md` 给出**按卡型测表方法**(sm_75 及更旧默认关,P100 需自测 `mmvq_il_parity --emit-table`) | 全 bit-identical 设计(parity 测试把关) | IL 表未含 sm_60 数据(NInfer 要自测);打包格式是私有布局,需自带转换 |
| **ST `iq_kernels.cu`(192KB)** | `vec_dot_q2_0/q3_*/q4_*/q5_*/q6_k/q8_0/iq1_m/iq2_*s/iq3_*s/iq4_nl/iq4_xs_q8_1` 全家 + `dequant_flat_kernel`/`dequant_gu_kernel` + `mmvq_kernel/mmvq_multi_kernel`、`native_gu/down(_multi)_kernel`(分组专家)、`embed_rows_kernel`(IQ4_NL/IQ2_S/Q4_0/Q8_0 表行 gather) | 所有 i-quant 点积 + 混合精度 MoE(gu 与 down 类型不同,`dgu/dd` 运行期参数) | 逐 dot 转录自 llama.cpp vecdotq;`s26_tsum`、`unpack_ksigns`、table_16 的 PRMT 路径;AMD/NVIDIA 分支干净 | 为 ST 引擎整体 31–38 tok/s@40k(2×P100)供能 | NVIDIA 段含 sm_80+ `cp.async` 预取分支(用宏隔离,sm_60 走非预取路径,可移植) |

### 2.5 次选/专用 matvec

| 条目 | 源文件与 kernel | 覆盖 | 要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **SB-12 Q4_1 fp16 GEMV** | `SB\patches\12-mmvq-f16-sm60.patch`:新文件 `ggml/src/ggml-cuda/mmvq-f16-sm60.cu`(`quantize_a16`、a16 kernel、`a16_pick_nwarps`) | **仅 Q4_1**(全类型可用需扩);T=1..8(`ub∈{1,5,8}` 实例) | 与 QR 相反路线:**LOP3 魔数(0x64006400/0x54005400)直接拼 half2 再减 1024/64 基**(利用 10 位尾数恰容 4bit 值),fp16 累加(11 位,块和用 fp32 修 d·min 项);含 `quantize_a16` 预量化激活防溢出 | Q4_1 模型 decode **+9.5%**(dense 9B)+6.5%;ub=8 时 +25%;非 bit-exact 但 PPL/KLD 略优 | 只实测 Q4_1;累加顺序改变需按 NInfer 的位精确要求评估;寄存器压力(2 变体 launch bounds)使 3 列反而回退 |
| **ST `s_gemv.cu` / `s2_gemv{,_fast,_quads,_q8}.cu` / `s2_expert_grouped.cu`** | 一行一线程 naive → warp-per-row(`s2_gemv_quads`)→ 4 warp/block(`s2_gemv_fast`) | S/S2 家族 2–6 bit 码 + f16/fp32 scale 激活 | `decode_tbl` 共享内存码本;`row_dot_s2_q8` 纯 dp4a 双整数和;`s2_gemv_q8` 注释明确"速度收益应来自 __dp4a —— **sm_60 上没有,所以留 naive**"(反面教材) | naive 5600→590 GB/s(quads);fast 受 6700 GB/s 载入上限 | 依赖 dp4a 性能的位置在 sm_60 退化;`s26_tsum.cuh` 的"每 64 元素省一次 32 项 warp 和"是关键可抄技巧 |
| **ST `bf16_gemv.cu`** | `bf16_gemv_naive_kernel`(一行一线程 bf16 位扩展 fp32 FMA)/ split 版 | 稠密 bf16 权重 × bf16 激活(T=1) | `f32_from_bf16` 位技巧;fp32 累加 | 无 P100 数字(P100 走"fp32 投影"路径时用它) | 纯 fp32 FMA,未用 HFMA2;NInfer 若 bf16→fp16 预转换后应改用 fp16 GEMV |
| **ST `native_moe.cu` / `shared_expert.cu` / `router_top10.cu`** | topk_moe、shared-expert 流水 | MoE 路由(NInfer dense,**不需要**,列此仅作对照) | — | — | — |

---

## 3. 算子族 ②:prefill GEMM(dequant→fp16 + cuBLAS)

> 结论:**cuBLAS hgemm 在 GP100 上就是最快路线**(约 80% 真 fp16 峰值,19.0 TF 见 1266MHz 时
> 13798/17408/5120 例程 15.3 TF/s),自写 fp16 SIMT GEMM 只在特定形状赢;真正的收益在
> ①喂给 cuBLAS 前的格式搬运,②算法选择覆盖,③bf16 权重展开。

| 条目 | 源文件与 kernel | 覆盖/形状 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0005 convert_unary 向量化** | `QR\patches\0005-…vectorise-the-contiguous-convert_unary-path.patch`:`convert_unary_cont_vec` | f32↔f16 连续对齐段,prefill 全部(占 prefill GPU 时间 6.1%) | 4 元素/线程:16B 读+8B 写;对齐/计数不足回退通用 kernel | 2×P100 pp **427.0→448.7(+5.1%)**;32k 大请求 380→404.9(+6.5%);164 GB/s→~209 稳态 | 无——纯通用改进,bit-identical |
| **Q36-0015/0016 cuBLAS ALGO6** | `Q36\patches\0015/0016…`(ggml-cuda.cu `ggml_cuda_cublas_gemm_…`) | f16 GEMM:K≥5120 全列数;K<5120 列数≥256 | GP100 默认启发式在 512–1024 列极差;ALGO6 在 K≥5120 的 64..4096 列全胜(ffn_down 17408×5120 32.0→9.0 ms);cuBLAS 拒绝则回退 | Qwen3.8-27B pp512 **139→200 t/s(+44%)**;35B pp512 +3.0%;附带 prefill 精度更近 decode(KLD 0.0071→0.00055) | 与 cuBLAS 版本相关(12.9 实测);NInfer 需在自己的形状网格重扫 |
| **ST prefill `gemm.cu` bf16 路径** | `ST\src\prefill\gemm.cu`:`bf16_to_f16_kernel`(7.0–7.4)、`bf16_to_f32_kernel`(Pascal 分支,`STRATA_EXPERIMENTAL_SM60` 编译) | bf16 权重/激活的稠密 GEMM(prefill) | sm_60 无 BF16 路径:cublasGemmEx(16BF) 报 NOT_SUPPORTED(#395 P40 实测);改为**精确移位展宽 f32 + cublasSgemm**(fp32 累加语义不变);7.0–7.4 走 bf16→fp16(尾数 7→10 位无损,±65504 钳制) | V100 上 bf16 展开是"第五个 4K prompt";P100 fp32 展宽无量化损失 | fp32 Sgemm 相对 fp16 路径吞吐折半——NInfer 若有 bf16 checkpoint,先离线转 fp16 更划算(见族⑦) |
| **ST prefill `prefill.cpp` + `kernels.cu` 展开路径** | `dequant_flat_kernel`(int 展开 `K_DEQUANT_FLAT`)、`dequant_bf16.cu` `K_Q_ROW_TO_BF16`(`STRATA_DEQUANT_BF16=0` 退回 f32 scratch) | 量化权重 → bf16/f16/f32 scratch → cuBLAS | 一次性展开整层进复用 scratch,`cublasGemmEx` CUDA_R_16BF→32F;scratch 分配失败优雅降级 | P100 整链在跑(Q4_0 KV 8k 下 decode 11.74 / 无 KV 15.85 tok/s,老社区基准) | bf16 scratch 在 P100 上仍被 fp32 GEMM 消费 |
| **Q36-0026 fp16 SIMT GEMM(dense tile 参考)** | `Q36\patches\0026-…fp16-expert-GEMM…patch`:`ggml/src/ggml-cuda/p60-moe-gemm.cu`(`q4_K_down`/`q5_K_down` 特化) | MoE 专家,但 kernel 结构=**通用 fp16 SIMT tile GEMM**:64 行 × 64 对,每 64 宽 K 步把权重解包进 smem half2 对,128-bit 载入喂 4 行/4 对 | 权重只解包一次(1024 次 HFMA2);fp32 累加;设备端 tile list 生成无 host 同步 | q4_K gate 9833→3588 us(**7.2 TFLOPS**,vs MMQ 2.5);0030/0031 窄/宽列数 tile 调整(1081→1454 t/s 合成) | MoE 包装(ids 排序 gather)剥掉后即 dense tile-GEMM 供体,与 QR 0007 的 PRMT/次正规解码组合可自造 sm_60 GEMM |
| **QR 0010 的 cuBLAS 教训** | 见族③:GEMM-attention 用 cublasGemmStridedBatchedEx fp16;**cuBLAS 12.8 fp16 NaN(12.4 正常)** | — | 版本门控 + `FA_GEMM_PREC` 逃生门 | — | NInfer prefill GEMM 也要版本回归门 |
| **SB-05 mmvf f32 vec 上限** | `SB\patches\05-mmvf-f32-pascal.patch`(`mmvf.cu` `get_max_cols_vec_kernel` 的 pre-Turing 分支 3→8) | **fp32×fp32** mul_mat,T=4(MTP draft+1 恰好超上限落进 sgemm) | 无矩阵核的分支统一给 8;sgemm 占 decode 5.9% 时的逃逸口 | +3.7–4.3% | 也抬高 Volta(未测);dense fp32 形状若 NInfer 走 fp16 可不要 |
| **SB-40 MMQ 线程表(no-DP4A)** | `SB\patches\40-mmq-iq4-nl-threads.patch`:`mmq-config-pascal-older.cuh`(新文件) | MMQ 批量路径,IQ4_NL/IQ3_S 的 tile | IQ4_NL 用 TQ16_8 要 320 线程×3 轮→80 线程 1 轮;Turing+ 与 HIP 表不动 | MoE down 投影 +3–4%,bit-identical | 若 NInfer prefill 完全绕开 MMQ(改 dequant→hgemm)则仅参考 |

---

## 4. 算子族 ③:attention(decode GQA、prefill flash 非 MMA)

### 4.1 decode GQA(T=1 与 T=2..8 验证)

| 条目 | 源文件与 kernel | 覆盖 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR f16-KV 策略 + vec kernel 算术链** | `QR\docs\kv-cache-type.md`(策略) + 上游 `fattn-vec.cuh/fattn-common.cuh` 路径 | T=1..2 decode GQA;f16 KV | 关键洞见:**量化 K 的点积在 sm_60 走 `vec_dot_q8_0_q8_1_impl→模拟 dp4a`,而 F16 K 走 `ggml_cuda_mad(half2)`(FAST_FP16 对 600 成立)**;HFMA2 15.8 对模拟 dp4a 3.95 TOP/s → f16 KV 反而快 | tg@75k 深度 **+24.1%**,pp 不变;多卡 23.8→29.9(+25.6%) | MTP(T>2)走 tile kernel(反正先转 f16),该收益不叠加;`-ctk f16 -ctv q8_0` 混型在 -sm tensor 下断言崩溃(未修) |
| **QR-0010 GQA-6 tile 折叠** | `QR\patches\0010-…patch`(fattn-tile.cuh `ggml_cuda_fattn_tile_get_config_nv` 新档位、launch_fattn_tile_switch) | 12Q/2KV per GPU(`-sm tensor`)2..8 列验证 | 每 KV head 折 6 个 query head(KV 读 1 次而非 3 次);`<256,8,6>`;0012 把档位限到 GP100 | 4 序列 24k 深度 prefill +27%;tg@32k +5.9%;`test-backend-ops FA` 2936/2936 | **sm_61 上 12/148 非整数→零驻留→−34%**(QR issue #3);P100 的 2 块/SM 上限被 0012 用作硬门控——NInfer TP2 的 head 分配先验 occupancy |
| **Q36-0020 decode FA(T=2..6)** | `Q36\patches\0020…`:`ggml/src/ggml-cuda/p60-fa.cu` | head 256、GQA 8、f16 K/V,2..6 token | 一块持该 KV head 全部 T×8 query 行(K/V 读一次);HFMA2 32 项链回 fp32;fp32 softmax/累加;二段合并 | 6 token@28k KV:710→382 us;@4k:140→79 us | `GGML_CUDA_P60_FA=0` 回退 stock |
| **Q36-0022/0023/0025 persistent attention 半层** | patches 0022/0023/0025 | MoE 35B:D 128/256,GQA 4/8,T=1;Q4_0 KV;含 RoPE、快照拷贝、残差 | 半层一次 cooperative launch;0023 激活 16B 向量化 | 汇入总表 | 绑定 qwen35moe 图形状;思路对 dense GQA 适用 |
| **ST `native_flash_attn.cu`** | `attend()`(D=256、ncols=1、F16 K/V、128 线程、`qreg/vkq` 寄存器分片、`__fmaf_rn/__fmul_rn` 锁 FMA 顺序、显式 online-softmax、status 码回退) | decode T=1;head_dim 256、kv=head/12(私有 GQA 布局) | 从 llama.cpp fattn-vec 特化(MIT 头,3cf03257);**不依赖 tensor core/dp4a/PRMT,half2float+FFMA 纯 SIMT**,sm_60 直接可编 | — | `tile[4*4*256]`=16KB smem;掩码/步表语义绑定 Strata 缓存;作为"NInfer 自有 KV 布局的 decode FA 模板"最佳 |
| **ST `qsa_decode_attn.cu`** | pre-sm_75 变体(PR #540,`STRATA_ATTN_PRE75_BUILT`,SM60 实验构建启用):`reduce12/warp_sum_x` 蝶形 reduce、更少 shuffle/smem、寄存器 80 vs 38 | 稀疏 decode attention(选中 KV 子集),KV 模式 0/1/2/3 = f16 / Q8_0 / **旋转 Q4_0(内联 nibble+scale 解旋)** | 逐位同基础 kernel;V100 +7% | `srow` 间接寻址;掩码行 weight=0 | 基础 kernel 是稀疏 top-k 选择后的 gather-KV;dense 路径需去掉 indexer/select |

### 4.2 prefill flash(非 MMA)

| 条目 | 源文件与 kernel | 覆盖 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0010 cuBLAS-GEMM flash attention**(Kmic-68 码,MIT fork) | `QR\patches\0010…` + 应用版 `QR\llama.cpp\ggml\src\ggml-cuda\fattn-gemm.cu`:`cublas_gemm_f32`、`fattn_gemm_softmax/mask_first_nz/fill/accum_O/finalize/q_to_f16`、`launch_fattn_gemm` | **fp32 Q + f16 KV**(q4_0/q8_0 cache 另加 `dequant_q4_0`);长 prefill:nq≥128 且 KV≥4096 | 标准 online-softmax,两 matmul 交 cuBLAS(P100 attention 形状 13–15 TFLOPS);K/V 逐块解量化省整缓存 f16 缓冲(262k 上下文 512MiB/卡);QK^T 默认 fp16 累加(`FA_GEMM_PREC=16`),fp32 可选 | pp2048@32k 深度 **285.6→352.9(+23.6%)**;长上下文 PPL 2.5031→2.5016;decode KLD 逐位不变;16 卡 NCCL 版 639 t/s 追平 2080 Ti | **cuBLAS 12.8 fp16→NaN**(#4):非 12.4 强制 fp32(收益缩到 +3.4%);q4_0 解量化段代码量大但独立;上游 Kmic-68/llama.cpp(MIT)可直溯 |
| **QR-0015 fp16/fp32 门控** | `QR\patches\0015…`(2 行) | — | `GGML_CUDA_CC_CUBLAS_SUPPORTS_F16_GEMM` 式版本判定模板 | — | — |
| **SB-31 f16 KV 分块 staging** | `SB\patches\31-fattn-f16-kv-chunk.patch`(`fattn-common.cuh` `FATTN_CHUNK_SIZE 16384`、`fattn.cu launch_fattn` 分块循环 + 每块重读 mres/lres 在线合并) | pre-Volta/pre-Turing 整缓存转 f16 路径(tile kernel);多块时 grid.y=kchunk 独立 softmax,末尾 `flash_attn_merge_softmax` | 量化 KV 缓存下 tile kernel 先全量转 f16:262k 上下文每卡 960 MiB compute buffer + 10 ms/遍转换(占 pp 6.9%);分 16k 块 → scratch 64 MiB | 960→64 MiB;pp 908@32k 430.5→533.5(**+24%**,9 轮);decode@32k −1.2%;>65k KV 因重扫缓存 −2~−8%(净赢) | 块内 pad 到 128;与 QR-0010 的 GEMM-FA 属两条不同路线(tile 路线省实现);tensor-core 路径不受影响(NInfer 无 tc,天然适用) |
| **Q36-0028 prefill FA(D256/GQA8)** | `p60-fa.cu` 长版(>6 token) | head 256 GQA 8,4 token×8 head=32 行/block,32-key tile | warp=4 行、lane=key;Q/K k-pair smem 广播;S/softmax fp32,P 存 key-pair;O+=PV HFMA2;下一 tile K 与当前 V 寄存器预取;全掩 tile 跳过 | 512 token@4k/16k:5.3/5.5→**6.9 TFLOPS**;pp1536@24k 634→694 | head_dim≠256 或 GQA≠8 回退;`FATTN_KQ_STRIDE` 127 vs 255 的 64 位除法陷阱(#831)注释详尽 |
| **ST `qsa_prompt_attn.cu`** | m8n8k4 f16 路径(实验构建):`m8n8k4_ldmatrix`、`qsa_attn_mma_accum` | 12 Q-head 打包、40 tile | — | — | **m8n8k4 是 Volta(sm_70)指令,sm_60 不可用——列入不可移植**;其非-MMA 对照 kernel(逐位等价)才是 P100 用的:lane=key、16 寄存器 K 片段+16 寄存器 12-head Q、`qsa_attn_bfly_sum12` 蝶形 12 步、`qsa_attn_rescale8` 折 rescale;V100 +7% |
| **ST `qsa_select.cu`/`qsa.cu`/`native_qsa_score.cu`/`native_qsa_indexer.cu`** | 稀疏 top-k 选择(radix select 6 轮/块、`select_finalize_kernel` 全序保证)、index head 投影(≤8 head 全寄存器)、amax→scale 链 | 深上下文选择性注意力(256k) | `native_qsa_score.cu`:`__CUDA_ARCH__>=800` 走 mma.sync tf32,**pre-sm_80 的标量回退用 bf16 中间值→amax 定标→f16 乘积**,带 0x00ff007f/0x007f00ff LOP3 掩码技巧 | 146k prompt 只读 20% KV;P100 走标量回退(慢但省 80% 读) | 整套依赖 NInfer 不需要(QSA 是 qwen4exp 特性);但 **bf16 amax→f16 SIMT 化**正是任务清单里"bf16-amax-f16-mma"的本地对应实现 |

---

## 5. 算子族 ④:GDN(gated delta net:decode 递推、prefill、MTP 验证状态)

### 5.1 decode 递推 + 状态搬运融合

| 条目 | 源文件与 kernel | 形状 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0009 状态管线折叠**(NInfer 直接命中 qwen35 图) | `QR\patches\0009…`(62KB):`gated_delta_net_cuda`/`_gather_cuda` 原位读状态、`gp100_softplus_bias_mul`(ADD→SOFTPLUS→MUL 一 kernel)、`gp100_gated_norm_prep`(silu(z)·rms_norm·w 直写 fp16 激活缓存)、`ssm_conv_state_f32`(conv 窗原位读写) | 单序列 decode;S_v=128 | 每线程先装载自己将写的状态元点到寄存器再写回→行可同进同出;l2norm 保留 `l2_norm_f32<32>` 的 lane 分片和蝶形序(逐位同);gated norm 的跨 warp 归约按 stock 256 线程块补零(逐位同) | 1786→1306 launches/token;真实 decode **32.3→35.25(+9.1%)**;KLD 逐位不变 | 一个**分配器陷阱**:融合节点延后读张量=读 ggml-alloc 认为已死的内存;`ggml_cuda_gp100_not_clobbered()` 的证明方法必读;仅匹配 qwen35 图;单序列 |
| **QR-0011 GDN 多序列** | 同上 patch:`gated_delta_net_gather_ms_cuda`([K,1,nseq] 列即状态行,零拷贝 gather)、ssm_conv 原位多序列、softplus 融合接受 per-head 广播 | 2..8 并发 | 一块覆盖所有序列的列(warp=(列,序列)),load 与 store 之间 `__syncthreads`(recurrent 缓存可重排 cell→跨块竞态);extra-states 拷贝非空时证明后才走原位 | 4-agent 状态 gather 1.15→0.54ms/层 | 同 QR-0009 |
| **Q36-0005/0006/0007/0008/0009/0012** | patches:跳 gather 原位态(0005)、q/k norm+scale 融合(0006)、gated out-norm 融合(0007)、alpha·dt-bias 跨 binbcast 融合(0008)、conv 状态洗牌折进 ssm_conv(0009)、dense qwen35 图节点重排(0012) | decode T=1(qwen35moe + dense qwen35) | QR-0009 的独立复刻+扩展(norm-scale 融合对应 llama.cpp v0.5.0 的 `scale(rms_norm(x),1/√n)` 新写法) | 汇入 2x 总收益 | 与 SB-48/23/24/25/27 同一族,任选实现路线 |
| **SB-23/24/25/27/48(GDN 融合,arch 无关)** | `SB\patches\23/24/25/27/48…`:beta sigmoid 折进 gdn(4080 launch 消失)、2MiB state gather 折进 gdn(defer-node 机制)、快照索引每图一次(−7.3ms)、conv GET_ROWS 折进 CONCAT(GP100 **64 位除法**是瓶颈,fastdiv 重写)、`fuse-rms-norm-scale` | qwen3-5/6 系模型 | 全 bit-identical(18/48 的替代关系有实测:48 开 tg64 73.72,关 72.93) | MoE 模型整链 +12.4%(17–22 合测) | 非 sm_60 专属,NInfer 可自由选择移植 |
| **ST `fused_gdn.cu` / `gdn.cu` / `native_gdn.cu`+`native_gdn_preprocess.cu`** | `gdn_step_norm_kernel`(512 线程,每列一线程组走 S=128 全程、状态驻留寄存器、conv+l2norm 折入)、`gdn_conv_l2_kernel`、`gdn_ab_kernel`;`gdn.cu` 的列所有权无 barrier 设计与 `MAX_H` 占用率教训(24 块/48SM 全空 vs 192 块);`native_gdn.cu`=llama.cpp gated_delta_net 的寻址适配 | decode T=1;S=128 状态 | 纯 fp32 FFMA + `__ldg`;`RG=4/RPG=32` 分片 | — | ST 私有状态布局 (S,h_v,S) 快轴 j;移植时对齐 NInfer 布局 |
| **ST `verify_kernels.cu`(MTP/DFlash2 语义金矿)** | `gdn_step_norm_multi_kernel`(多 token 验证步,conv+l2norm 折入)、`gdn_step_commit_kernel`、`gdn_step_split_kernel`(S25:每头 128 列拆 4×32 列块)、`gdn_state_replay_kernel`(S26:仅回写状态的 commit + gate/beta/v 预取)、`gdn_out_norm_kernel`、`mtp_select_kernel`、`row_top_prob_kernel`、`gdn_conv_commit/multi` | 投机窗口 T=1..6,带**逐 token 状态快照与回滚**(reject→replay) | 每 token 算术从单 token 原件逐序转录(验证位等价于顺序 decode);q8_1 输出像由持有该 32 列的 warp 顺手写(S26 `STRATA_QFUSE`) | 2×P100 实机 MTP 30–38 tok/s 的验证链 | 绑定 Strata 的 cell/step 环形索引语义;移植前读 `include/strata/kernels/verify_kernels.hpp` |

### 5.2 GDN prefill(chunked 的 matmul 部分)

> **缺口预警:本地没有任何 sm_60 可用的 chunked(WY 形式)GDN prefill kernel。**
> Strata 的 chunked 实现显式绑死 sm_80+(见 §9),sm_60 的 chunked 需自行推导——现有最近参考如下。

| 条目 | 源文件与 kernel | 形状 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0018 4 状态列/warp** | `QR\patches\0018-…gated-delta-net-prefill-at-4-state-column.patch`:`gated_delta_net_body<…,CPW=4>`、`GDN_LAUNCH_BOUNDS` | prefill 2048 token;S_v=128,单序列非 KDA 路径 | 每 warp 4 列共享 k/q 载入与 l2 归约(算术逐列不变,位精确);块数 768→192,单波驻留;`GP100_GDN_CPW=1` 回退 | pp2048 **446.7→463.8(+3.8%)**、pp8192 +3.8%;KLD -ub512 不变 | CPW>1 有 static_assert 限制(KDA/多序列路径不给);仍是 token-串行递推 |
| **Q36-0027 prefill GDN 4col×8lane** | `Q36\patches\0027-…prompt-processing-gated-delta-net…patch`(gated_delta_net.cu) | S_v=128、无 KDA、≥32 token(`GGML_CUDA_P60_GDN_CHUNK`) | warp=4 列(8 lane×16 行),3 步归约;q/k 每 4 列读一次 smem 广播;每 token 输入按 8-token 块合并预载入 | 1536 token 32 头:18.5→8.7 ms;pp4096 1462→1540 | 归约序改变(非位精确);与 QR-0018 同族可选 |
| **ST `prefill\kernels.cu` gdn_rec 家族** | `gdn_rec_kernel`(一列一线程)、`gdn_rec_cols_kernel`(256 线程 128 列)、**`gdn_rec_cols_pipe_kernel`**(下一 token 的 q/k/v/gate/beta 入寄存器流水——**纯 sm_60 可移植的流水版**)、`gdn_rec_kh_kernel`(3 个共 key 的 value head 同线程,q/k 摊销;`STRATA_GDN_CP_ASYNC` 宏对 <sm_80 有**普通拷贝回退**,但 host 侧 `gdn_keyhead_ok` 只在 sm_80+ 且 64 块全驻留才启用——NInfer 可评估在 56SM 上的取舍)、`gdn_rec_quad*`(AMD 4-head 分组,LDS 广播思路)、`gdn_l2_kernel`、`gdn_conv_kernel/tiled/hist`、`gdn_out_norm_16_kernel`、`gdn_states_gather_kernel` | prefill 全 token 扫描;H=32 K-heads×4 V、S=128、C=10240 | 每变体输出 fp16(16B/线程向量化),T 维 128-token tile 入 smem | 8k prompt GDN 1.71→1.63 s、128k 16.6→12.7s(`--gdn-prefill 2` vs 0;老社区数据);kh vs pipe 计时在 `gdn_rec_parity.cu --bench` | pipe/kh 位精确;chunked 变体不可用(下行) |
| **ST chunked WY(仅 sm_80+,反面参考)** | `gdn_chunk_prep_kernel`/`gdn_chunk_scan_kernel`(`STRATA_GDN_CHUNKED=1`,GCH=32) | 与 18 个 chunked matmul 合并省 107 启动/块 | `s_D 128KB + s_Mm 2×64KB + s_A 64KB`=**93.7KB smem**、cp.async 链、`__syncthreads` 依赖——**sm_60 三重不可行** | 5090 上 5.68→1.99ms(2.85x) | 列入 §9 不可移植清单;NInfer 的 chunked-on-sm_60 需要自己按 WY 形式重写 ≤48KB 的 tiling,这是全目录最大的空白点之一 |
| **SB-39 lanes-per-column** | `SB\patches\39-gdn-lanes-per-column.patch` | S_v=128、≥32 token | 归约宽度做成模板参数(32→16 lane,FMA/shuffle 比翻倍);`GGML_GDN_LPC=16` | 40k prefill −4.3%,decode 构造性不动 | 归约序改变 |

---

## 6. 算子族 ⑤:KV cache 读写 / 量化点积

| 条目 | 源文件与 kernel | 覆盖 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0010 q4_0 直进 tile smem** | patch 0010(fattn-tile 内解量分支) | f16/q4_0 KV,prefill tile | 无整缓存预转换 | 深上下文 +27% | — |
| **SB-31** | 见族③(同一条目,双重命中:KV 展开的内存策略) | 量化 KV tile 路径 | 分块展开+在线 softmax 合并 | +24%@32k | — |
| **ST `kv_q8.cu`** | K/KV append/gather:每 block 一个 64 值组一个 head(64 线程),组 scale 单值;`MULTI` 模板多 token 单启动 | Q8 KV cache 读写;head_dim%64 | 确定性组最大值(跨版本可复现) | — | 需与 QSA 读端(`qsa_decode_attn` KV_MODE=1)配套 |
| | `ST\src\kernels\cuda\kv_q8.cu`;关键:每 block = 一个 64 值组 × 一个 KV head(K/z=0,V/z=1),64 线程每线程一值;`MULTI` 模板 = 一次启动写多 token 的 K/V(STRATA_LFUSE) | Q8_0 KV append/gather,paged;head_dim%64 | 确定性组最大值(scale 可复现),逐值位精确 | P100 生产配置"int8 KV"跑 2×P100 社区基准 | gather 端(注意力读+反量化)在 `qsa_decode_attn.cu`,配套移植 |
| **ST `kv_q4.cu`** | `fwht256_kernel`(**Fast Walsh–Hadamard 旋转 256 维:一 warp 一行、每 lane 8 个寄存器值、正交自逆**),Q4_0 K/V 量化(带确定性组最大值),gather 逆旋转 | Q4_0 KV;head 256 或 128(2×128 折叠) | 旋转把离群值摊平→4bit 存 K/V 精度损失可控;PR #21(code-martin) | 2×P100 社区基准 Q4_0 KV@8k:11.74 vs 无 KV 15.85 decode 差距可接受 | FWHT 本身纯 f32 FMA/add,sm_60 直接可编;K≠256 需改 butterfly 层数 |
| **ST `kv_stream.cu`** | 池/分页 KV 流式解析:`Runs`(至多 4 池数组的逐块字节程)、`RT=1024` resolve 块、按页 gather/回写 | 长上下文 KV 流式(类 PagedAttention) | 与 kv_q8/kv_q4 配套的驻留/换入换出 | — | Strata 语义重,NInfer 若自管 KV 只取 kernel 形状 |
| **ST `quantize_act.cu`** | `quantize_q8_0_kernel`、`quantize_q8_0_scaled_kernel`(外部 scale)、`dequant_q8_0_kernel`、`quantize_q8_K/dequant_q8_K` | 激活/中间量 q8_0 往返 | 确定性 scale | — | — |
| **SB-38 getrows q4_0 块读** | `SB\patches\38-getrows-q4-0-block.patch`(getrows.cu):32 元素按 2 字节对齐的 q4_0 行改块读 | q4_0 表行 gather(`emb` 等) | 与 Q36-0010/0014 同款"对齐碎读→smem/宽读"修复 | 60 chunk × 8k prefill −10.1%(125.7→113.0 s),bit-identical | opt-in `GGML_CUDA_GETROWS_Q4_0_BLK=1` |
| **SB-14/37 getrows 窄行** | `14-getrows-narrow-rows.patch`(每行 16B→一个 warp 一行)、`37-getrows-narrow-batched.patch`(一个 block 一 ubatch 行 + warp 内 fastdiv) | MoE/emb gather;dense 12.5% MoE 3.1% | 278us→可忽略;MoE −9.7%、dense −9.4% prefill,bit-identical | 37 默认开 | 64 位除法规避技巧(GP100 无硬件)注释详尽 |
| **QR f16-KV 政策** | `QR\docs\kv-cache-type.md`(§4.1 已列) | — | 对 NInfer:sm_60 上 **f16 KV 是默认正确答案**,量化 KV 只在 VRAM 逼不得已是选项,且必须走"读端 HFMA2"设计 | +24%@75k | — |

---

## 7. 算子族 ⑥:TP2 allreduce 传输

> 这条线 QR 是唯一完整实测供体;Strata 的 peer/doorbell kernel 提供 mapped-memory 传输的底层原语。

| 条目 | 源文件与 kernel | 范围 | 技术要点 | 实测 | 障碍 |
|---|---|---|---|---|---|
| **QR-0003 Pascal internal AllReduce** | `QR\patches\0003…` + `QR\llama.cpp\ggml\src\ggml-cuda\allreduce.cu`(`ggml_cuda_ar_pipeline_init` 门控) | `-sm tensor` cc≥600 | 上游唯一 Volta-only 构件是 `__nanosleep`——换成 `clock64()` 延迟环(128 周期≈100ns@1328MHz)即解锁;传输=mapped pinned host memory,arrival flag=`volatile int`+`__threadfence_system()`(本来就是为无 NVLink 的 PCIe 卡设计);pre-Volta 向量宽 8B 自适应;20 行改动 | pp512 +3.7%、pp8192 **+17.0%**、tg **+9.6%**、真实请求 TTFT 23.4→21.3s;消除每 2048-token prefill 20.5GB 的 f32 绕道主机(23.1% GPU 时间) | spin 安全性需审计:**只有每块 lane0 自旋、flag 由对端 GPU 写、BAR.SYNC 全收敛**才无死锁(GP100 无独立线程调度);bf16 wire 非位精确(`GGML_CUDA_AR_BF16_THRESHOLD=0` 可位精确并保留 tg 收益) |
| **QR-0004 双拷贝引擎重叠** | `QR\patches\0004…`(`allreduce.cu` stream/event) | 所有架构 copy-engine 路径,≥1MB 张量 | P100 `asyncEngineCount=2`:D2H/H2D 各占一流;**必须给"计算流等自己 D2H"补显式 event**(少这条:速度相同、PPL 7.68≠3.53——fast-wrong 教材) | pp2048 +5.7%、35min soak +4.7%;2MB pinned 对拷 100→55ms | 需要评审 ordering;数值经 PPL 门验证 |
| **QR-0006 N(>2)卡 internal AllReduce** | `QR\patches\0006…`(allreduce.cu +27KB) | 3+ GPU | 上游仅支持 2;扩展为 N 设备 chunked/copy-engine/混合三路径 | 正确性验证过(虚拟 3/4 设备 PPL 匹配);4×P100 用户报告稠密 27B **+70–90% tg**;`GGML_CUDA_P2P=1` 在该类平台 **−40%**(禁);`GGML_CUDA_ALLREDUCE=none` −25% | NInfer 只 TP2,可作背景;chunked kernel 无 warp 同步依赖(注释明示) |
| **ST `prefill\kernels.cu` peer 系列** | `moe_combine_peer_kernel`、`moe_combine_peer16_kernel`、`peer_scatter_add_kernel`、`peer_gather_add_kernel`(带 `head[]/count[]` 溢出链表)、`sums_to_f16_kernel`/`f16_to_f32_wide_kernel`(f16 线宽) | 跨设备(mapped 内存)合并/散布/聚集加法 | 用**单条向量化 kernel 对每行求和**替代 `atomicAdd` 链;读入 16B 写出 8B 保 128-bit 事务;16 项溢出用链 | 第二 GPU 路径在 P100/V100 实跑 | 语义是"专家合并"不是层内 allreduce,改造为 NInfer TP2 第二路径需自加 flag 同步(用 QR-0003 方式) |
| **ST `elementwise.cu` doorbell 家族** | `doorbell_ring/wait/publish/publish_value_kernel`、`copy_from_mapped(_multi)`、`copy_rows_from_mapped_kernel`(verify 窗口跳算行写 +0.0) | GPU↔主机 mapped pinned 传输原语 | volatile store 规则(GFX1201 plain store 被重排进 barrier 前的教训)、读改写 vs 纯 store(`STRATA_DOORBELL_STORE=1`)、**克隆字面量陷阱**:ring 的 `+1` 表达式在 host 求值,克隆时要同步改 | — | 面向 CPU-GPU;NInfer 主机中转 TP2 可直接复用其 float4 拷贝+门铃模式 |

---

## 8. 算子族 ⑦:bf16 → fp16/fp32 反量化

| 条目 | 源文件与 kernel | 覆盖 | 要点 | 实测/校验 | 障碍 |
|---|---|---|---|---|---|
| **ST `dequant_bf16.cu`** | `K_Q_ROW_TO_BF16`(18 格式→bf16:q4_0/q5_0/q8_0/q2_k/q3_k/q4_k/q5_k/q6_k/iq1_s/iq2_xxs/iq3_xxs/iq2_s/iq3_s/iq4_nl/iq4_xs/mxfp4…;`ggml_cuda_fp32_to_bf16` 单值、`fp32x2_to_bf16x2`(PRMT)双值)、`K_QF32_TOBF16_ROW`、`K_Q_BF16_TO_F32`、`K_BF16_ROW_TO_F16/F32`、`k_q_to_f32<IL=true>`(顺带产 q8_1-IL 表) | 全量化格式 ↔ {f16,bf16,f32} 三向 | 统一宏调度 `STRATA_DEQUANT_BF16=0` 退回 f32 scratch 模板;f16 舍入用 `__float2half_rn`(与 ggml 逐位一致) | ST 的 dequant_bf16_parity.cu 体系 | bf16≠fp16 的警告注释(gr.cu:46)要带进 NInfer 文档 |
| **ST prefill `gemm.cu` 三 kernel** | `bf16_to_f16_kernel`(尾数 7→10 无损;±65504 钳制;`__float2half_rn`)、`bf16_to_f32_kernel`(`(uint32)x<<16` 精确位展宽) | 稠密 bf16 checkpoint→可用 GEMM | Pascal:bf16 GEMM NOT_SUPPORTED 的两条出路——f32 展宽(cublasSgemm)或离线转 f16 | 7.0–7.4 走 f16 路径省"第五个 4K prompt";Pascal f32 路径数值无损 | sm_60 上 bf16 消费端永远要落到 f16 或 f32,没有第三条 |
| **Q36-0004 fused norm 写 q8_1;ST `native_quantize_q8_1_kernel`** | 见族①/② 源文件 | 激活 bf16/f32→q8_1 往返 | q8_1 finite clamp(`q8_1_finite.hpp`) | — | — |
| **SB-42 qwen4exp-hc-exact** | `SB\patches\42-qwen4exp-hc-exact.patch` | bf16 逐位语义:HC fold 的 `rn_bf16`、down matmul 逐位等价 | bit-identical 默认;fold 默认关(`LLAMA_QWEN4EXP_HC_FOLD=1`) | v0.5.0 实测链 | 绑定 hyper-connections 模型;若 NInfer 需 bit 级 bf16 往返,`rn_bf16` 写法可直接抄 |
| **QR-0003/0006 的 bf16 wire** | allreduce.cu 的 f32↔bf16 转换 kernel(`GGML_CUDA_AR_BF16_THRESHOLD`) | TP2 数值策略参考 | 位精确模式=纯 f32 | 见 §7 | — |

## 9. 算子族 ⑧:NVFP4(fp4 码 + fp8 scale)反量化

> **本地供体最薄的一族。** 唯一实测代码是 MXFP4(e2m1 + **E8M0** 32 元素块 scale),NVFP4 是
> e2m1 + **E4M3(FP8)** 16 元素块 scale + fp32 全局 scale——码本解码可复用,scale 解码链需自行补
> E4M3→f32 一步(256 项查表或位构造;sm_60 无 `__nv_fp8` 硬件)。

| 条目 | 源文件与 kernel | 覆盖 | 要点 | 实测/校验 | 移植障碍(NVFP4 视角) |
|---|---|---|---|---|---|
| **SP-0001 MXFP4 全套** | `SP\patches\0001-p100-mxfp4-q5_1-ple-residency.patch` → `ST` 树:`vec_dot_mxfp4_q8_1`、`get_int_b1`(17B 块字节对齐 nibble,来自 llama.cpp 同名函数)、`e8m0_to_fp32`(`x==0?0x00400000:(uint)x<<23`,CUDA12.8 前形态,位精确)、`Split<39>`(权重预载/点积两段式)、`dq_mxfp4`(→f32/f16/bf16)、`Fmt<39>` 注入 MMVQ/GU/D 分派表;`gguf_reader.hpp` 块几何(32 元素/17B) | MXFP4 权重 × q8_1 激活,decode MMVQ 与专家路径 | e2m1 码本=**kvalues_mxfp4 = IQ4_NL 码本×2**,`get_int_from_table_16`(PRMT 表 16)直接复用;`*0.5f` 吸收码本加倍 | 2×P100 生产部署(AtomicChat down 投影);parity 门 | ① E8M0→E4M3:需新增 e4m3_to_f32(SM60 无 cvt;查表 256 项或位拆分)且块宽 16≠32、②点积仍吃 vmad 模拟 dp4a 速率(3.95 TOP/s),fp4→fp16 直喂 HFMA2 的路(次正规/魔数解码)**无人做过**——正是 NInfer 要自研的核心 |
| **ST `ngram.cpp` `fp8_e4m3_dequant_row`**(主机侧) | `include\strata\kernels\ngram.hpp:132`;NaN 规则 `e==15&&m==7→0` | E4M3 字节码→f32(scale 外乘) | 主机实现的解码语义参考 | — | 需 GPU 化;逐位语义可作 device kernel 的 parity 基线 |
| **ST iq_kernels 表解码基元** | `get_int_from_table_16`、`unpack_ksigns`、`q4_0_centred`(PRMT 链) | fp4 码→int8 值 | NVFP4 的低 4bit e2m1→half 映射可仿 QR-0007 PRMT→次正规 half 或 SB-12 LOP3 魔数(e2m1 值域 {0,.5,1,…,6} 能塞进 10 位尾数 → **LOP3 魔数法对 fp4 同样成立**,SB-12 注释的推导可重跑) | — | 无人把 fp4×fp8scale 接到 HFMA2 路径;这是缺口,亦是机会 |
| **线索(未本地验证)** | srose69/nvfp4effma(任务点名):README 级线索,五个本地项目中**均未出现引用** | 名称推测 NVFP4 权重做 E×F×MA(反量化×fp16 FMA)路线 | — | 许可证/内容均未核实,**不得移植,先联网核实** |

---

## 10. 交叉辅助供体(norm/swiglu/elementwise/launch 策略)

| 条目 | 文件与符号 | 作用 | 数字 | 备注 |
|---|---|---|---|---|
| QR-0007 `gp100_rms_norm_mul_prep` | mmvq-gp100.cu | 单行 RMS_NORM×w 直写 fp16 激活缓存(5120 行原本 1 块 latency-bound) | 折进 0007 收益 | NInfer 的 RMSNorm+prep 融合模板 |
| QR-0013(反向) | `0013-…GLU-Q6_K-MMVQ…` | 8 行 GLU Q6_K MMVQ 静态 smem **56KB>48KB** → CUDA 12.4 起编不过;删实例化恢复 48KB | — | smem 预算硬证据 |
| QR-0017 | `0017-…uncached-activation-buffer…` | 256 线程×8 列 fp16 激活缓冲 3.05MB→0.55MB(每块 ≤10KB smem,2 块/SM) | 1536-token pp 8192→2673us | 占用率修复模板 |
| SB-16/26 cpy fastdiv | `cpy.cu` | GP100 无硬件 64 位除法:`k_get_rows_float` 640 万条 SASS 里 **289 万是 MUFU 序列**;warp 内算一次商/倒数 | −56% 核函数、+0.82%;26 = 1.9× | 所有 gather/concat 类 kernel 通用修法 |
| SB-17/18 norm 缓存与兄弟融合 | norm.cu、ggml-cuda.cu | rms_norm 寄存器缓存、ADD+RMS_NORM(+MUL+Mul) 一启动 | +0.71%/+0.95%/+1.06% | 与 QR-0009 同类但 arch 无关 |
| SB-19/20 | `fuse-pre-add-rms-norm`、`fuse-add-unary-mul` | 默认关:dense 位精确,MoE 非确定(实测记录) | +0.70%/+0.72%;双开 MoE +2.36% | 风险记录值得整篇读 |
| SB-11/13/15/21/22/28/41 host 侧 | penalties-direct、sampler-prefilter、mtp-draft-vocab、sched、top_k | CPU 发射瓶颈:惩罚表直扫(0.78ms/token=5.3%)、prefilter 2.2%、top-k 部分选择 7.6×、radix-select 133→74.8us | 各 +1~+5% | decode 到 ~2ms/步后 host 占比升高——NInfer CPU 规划参考 |
| SB-15 / Q36-0017 MTP 草案词表 | `mtp-draft-vocab`、`draftvocab/dv32k.bin`、`scripts/build-draft-vocab.sh` | 输出头只算高频 32k 词表(接受率换带宽) | SB:+14–20%(接受 0.87);Q36:8-prompt 135.5→180.0(+33%)、28k 102→136;覆盖不足反而 −17% | open-jet 思路(AGPL,行为参考);Q36 版每 token 精确率不降 |
| ST `cvec.cu`/`ple.cu`/`native_rope.cu`/`rope.cu`/`native_router.cu` | CVec 表(GPU 常驻/主机门铃两模式,`__constant__` 16B 结构)、PLE 行 gather(IQ4_NL/IQ2_S/Q4_0/Q8_0 表) | Qwen4exp 特性,与 NInfer 的 PLE/MTP 输入侧形似可借鉴 | — | `__constant__` 4KB 限制注释完整 |
| ST `gr.cu`/`fused_gr.cu`/`native_gr_*` | hyper-connections(gated residual):`gr_up_fused_kernel`、bf16 激活语义(`lo=silu((bf16(xn)@w)/hc)`)、cp.async 流水 | 若 NInfer Qwen3.8-27B 含 HC 则必读;`fused_gr.cu` 的 **cp.async 段有 `__CUDA_ARCH__>=800` 宏门**,非预取路径完整 | 68→72/78 tok/s(5090);V100 −48% 主机拷贝 | **fused 版含 cp.async(§9 清单)**;gr.cu 的 naive 版纯 SIMT 可编 sm_60 |
| SB-44/45/46/47 + Q36-0021/0024 | qwen4exp 专家缓存/稀疏 gather/moe cache;persistent MoE | MoE 专属,NInfer(dense)仅背景价值 | 40k decode −58.8%/−52% 组合收益 | 架构绑定;`LLAMA_MOE_CACHE` 等开关语义清晰 |

---

## 11. 许可证警示

| 项目/条目 | 许可证 | 结论 |
|---|---|---|
| QR(p100-llama-cpp-main) | **MIT**(© 2026 qr0n;底层 llama.cpp/ggml MIT,patch 同照) | 可直接移植。注意 0010 的代码作者另有其人 |
| SB(llama-cpp-p100-patches-main) | **MIT**(© 2026 shinbunbun) | 可直接移植 |
| Q36(qwen36-p100-patches-main) | **MIT**(README 明示;基线 SB 亦 MIT) | 可直接移植 |
| ST(Strata-for-p100-0.1.41) | **MIT**(© 2026 Niko1221 及贡献者);llama.cpp 衍生文件保留 ggml MIT 头并 pin commit | 可直接移植(保留衍生文件版权头义务) |
| SP(p100-strata-main) | **MIT**(© qr0n,基于 Strata) | 可直接移植 |
| **L-Forster/open-jet(megakernel)** | **AGPL** | **只能作行为参考,不可抄码**;Q36 已示范正确做法("No code from it is used here——it is AGPL and built on sm_80 features");persistent 层半 kernel、草案词表两思路引用时须注明思路出处 |
| **Kmic-68/llama.cpp** | llama.cpp fork → **MIT** | QR-0010(fattn-gemm、GQA 折叠、q4_0 smem 解量)实际作者;移植时保留出处(QR 的 patch 头已注明,随码带走) |
| **sirCamp/flash-attention-legacy** | 未核实 | 五个本地项目均未引用;名称推测为前 MMA 时代 FA。先核许可证 |
| **poisonxa16/pxa** | 未核实 | 同上,本地无引用,仅记录 |
| **srose69/nvfp4effma** | 未核实 | 同上,且恰是 NVFP4 族最需要的线索——优先联网核实,勿在核实前移植 |
| NVIDIA cccl、cpp-httplib、miniaudio、stb、nlohmann/json、subprocess.h、zod-to-json-schema(ST third_party) | 各自 BSD/MIT/zlib | 与 kernel 无关,列出以防打包误带 |

## 12. 不可移植清单(sm_60 硬拒:wmma/tc/cp.async/dp4a/大 smem/`__nanosleep`)

| 条目 | 位置 | 不可移植原因 | sm_60 对应替代 |
|---|---|---|---|
| ST `prefill\wmma_gemm.cu`/`wmma_gemm.h` | 整个文件 | **AMD RDNA3(gfx1100)WMMA**,hip 头编译;NVIDIA 上永不启用 | 不适用(NVIDIA 走 cuBLAS/自 SIMT) |
| 一切 NVIDIA wmma/mma.sync | 上游 llama.cpp MMQ/FA 的 tensor-core 路径;ST `native_qsa_score.cu` 的 `mma.sync.aligned…tf32` 分支(宏门 ≥800)、`native_qsa_indexer.cu` | sm_70/80 指令 | 各文件都留了 pre-sm_80 标量回退(QSA score 的 f16-amax 链是最佳参考) |
| ST `qsa_prompt_attn.cu` 的 m8n8k4 f16 kernel(PR #600) | `__CUDA_ARCH__ >= 700 && < 750` 内联 PTX | **Volta 专属 mma**;sm_60 切片里编译成 trap(文件其余部分可用) | 同文件 fp32 SIMT 对照 kernel(+7% 于 V100,逐位同) |
| ST `gdn_chunk_prep_kernel`/`gdn_chunk_scan_kernel` | `prefill\kernels.cu` | ① `cp.async.cg…16` PTX(宏 `STRATA_GDN_CP_ASYNC`,Turing 编普通拷贝但**host 只放行 sm_80+**);② 93.7KB smem>48KB;③ 依赖独立线程调度的 spin 假设 | 用 `gdn_rec_cols_pipe_kernel`(寄存器流水)或 QR-0018/Q36-0027 的列折叠;chunked-WY-on-sm_60 = NInfer 自研项 |
| ST `fused_gr.cu` 的 cp.async 半流流水 | `#if __CUDA_ARCH__ >= 800` 包 cp.async 原语 | 8.0+ | naive 路径(gr.cu)纯 SIMT 可用 |
| ST `iq_kernels.cu`/`native_mmvq.cu` 中 sm_80+ 预取分支 | `cp_async16/cp_async_commit/cp_async_wait<0/1>`(宏隔离) | 8.0+ | 非预取模板本体编译 sm_60;注意 ST 的 i-quant 链假定 `__dp4a` 速率,须带 `STRATA_DP4A` vmad 头一起移植 |
| **`__dp4a` 本体** | 任何按 6.1 假设计时的 int8 路径(SB-01 注释、ST dp4a.hpp) | GP100=6.0 无此指令 | vmad×4(SB-01/ST,位精确)或干脆 fp16/HFMA2(QR-0007/SB-12) |
| `__nanosleep` | 上游 allreduce.cu 门控(=QR-0003 的全部故事)、ST spin_pause(<700 空转) | 7.0+ | clock64() 延迟环 / 空转 spin |
| 静态 smem >48KB | QR-0013 的 8 行 GLU Q6_K(56KB,CUDA12.4 起编不过)、ST chunked GDN(93.7KB) | GP100 48KB/block 上限 | 降实例化/换列折叠方案(QR-0017 的 buffer 收缩是同类思路) |
| GPU 特性表错配 | SB-35 的 MMID 上限按架构表(32/8/16);`GGML_CUDA_CC_DP4A`=610 的各处门 | sm_60 会静默走非预期路径 | 照抄表并把 600 单列一行,别信 610 值 |
| CUDA graphs / MMVQ 融合 | QR `docs/negative-results.md` | 非硬件禁止,但实测 −1.4%/−2.1% | 别投入;persistent kernel(Q36)才是削减 launch 的正道——CPU 发射 floor 约 17.8ms/token(2×P100) |
| `GGML_CUDA_P2P=1` | QR `docs/prefill-and-multi-gpu.md` | 双路 UPI P100 上 357→33 t/s | 用 QR-0003 主机中转;NInfer TP2 默认禁 P2P |
| cuBLAS 12.8 的 fp16 长上下文 GEMM | QR issue #4(0015) | NaN;12.4 正常 | fp32 默认 + 版本白名单 |
| CUDA 13 | SB README | 移除 Pascal 代码生成 | 固定 CUDA 12.x + gcc≤13(12.4 时 nvcc 需 gcc≤13) |

## 13. 覆盖度结论

**覆盖最强族:** ① decode matvec——三套独立 fp16/HFMA2 实现(QR-0007/0008/0011/0016 全谱 1..8 列、SB-12 的 LOP3 魔数路线、Q36-0001/0010/0013/0014 的位精确与多列路线)+ ST 的 MMVQ 全家桶与 vmad-dp4a/共享码本通用件;叠加 MTP 语义(0032-0035 的位精确 MTP、ST verify_kernels)与草案词表,**几乎每块拼图都有两个以上来源且带 P100 实数**。attention、TP2 allreduce、bf16 转换、通用 MMVQ 参数学也都有完整实测供体。

**最缺参考族:** ⑧ **NVFP4→fp16**——本地唯一可编译代码是 MXFP4(SP: E8M0 缩放 + e2m1 表解码),它缺 E4M3 scale 解码、缺 fp4 直喂 HFMA2 的位构造、缺 16 元素块几何;srose69/nvfp4effma 是名义线索但本地零引用。④ **GDN chunked prefill 的 sm_60 版**——唯一 chunked(WY)实现(ST)三重绑定 sm_80+(cp.async、93.7KB smem、host 门控),sm_60 可用的只有逐 token 递推的列折叠改良(QR-0018、Q36-0027、pipe/kh 家族),没有任何"sm_60 预算内的 chunked matmul"参考。其次薄的是 fp4×fp8 的注意力/KV 侧(完全没有)。

**其他重要提醒(移植前读):** GP100 上 `PRMT/SHR` 半速、无 64 位除法、无独立线程调度(spin 必须逐处审计)、f16 KV 优先于量化 KV、`test-backend-ops` 之外必须用 `-ub 1/5` + KLD 对 Q8_0 参考门控质量(QR 全程如此)、`bench/sweep.cu`(QR)与 `sweep.cu`(ST tools)可复用做本机指令吞吐标定。
