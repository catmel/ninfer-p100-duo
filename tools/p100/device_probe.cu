// P100 (sm_60) device capability probe for the NInfer P100 migration.
//
// Records, on the actual target card, the hardware facts every sm_60 route decision depends
// on: compute capability, SM count/clocks, HBM2 bandwidth inputs, the 48 KiB shared-memory
// ceiling, L2 size, copy-engine count, cooperative launch, CUDA-graph stream capture, and
// peer access between every device pair (the TP2 transport decision matrix:
// NVLink-P2P / PCIe-P2P / host-staged). Run it on every candidate dual-card host and record
// the output in docs/p100-migration.md before choosing a transport mode.
//
// Build (Linux, CUDA 12.x):  nvcc -O2 -arch=sm_60 tools/p100/device_probe.cu -o device_probe
// Build (Windows): run inside VsDevCmd (or "cmake --build" of this project's tools target):
//   nvcc -O2 -arch=sm_60 tools\p100\device_probe.cu -o device_probe.exe
// Usage: ./device_probe [dev_a dev_b]   # peer matrix covers all visible pairs by default

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

#define CHECK(expr)                                                        \
    do {                                                                   \
        cudaError_t e_ = (expr);                                           \
        if (e_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__,       \
                         __LINE__, #expr, cudaGetErrorString(e_));         \
        }                                                                  \
    } while (0)

namespace {

__global__ void hfma2_kernel(const half2* __restrict__ a, const half2* __restrict__ b,
                             half2* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __hfma2(a[i], b[i], c[i]);
}

__global__ void coop_kernel(int* out) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    if (threadIdx.x == 0) atomicAdd(out, 1);
    g.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = g.size();
}

// The exact cross-device form the TP2 captured graph uses: cudaMemcpyAsync between devices
// with peer access enabled (NOT cudaMemcpyPeerAsync, which is not stream-capturable).
void probe_graph_and_copies(int dev) {
    cudaStream_t s = nullptr;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float* buf = nullptr;
    CHECK(cudaMalloc(&buf, sizeof(float) * 256));

    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(buf, buf + 128, sizeof(float) * 128, cudaMemcpyDeviceToDevice, s);
    hfma2_kernel<<<2, 64, 0, s>>>(reinterpret_cast<half2*>(buf),
                                  reinterpret_cast<half2*>(buf),
                                  reinterpret_cast<half2*>(buf), 128);
    cudaGraph_t graph = nullptr;
    cudaError_t end = cudaStreamEndCapture(s, &graph);
    std::printf("  graph capture(d2d memcpy+hfma2)=%s\n", cudaGetErrorString(end));
    if (end == cudaSuccess && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        cudaError_t inst = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        std::printf("  graph instantiate=%s\n", cudaGetErrorString(inst));
        if (inst == cudaSuccess) {
            cudaError_t launch = cudaGraphLaunch(exec, s);
            cudaStreamSynchronize(s);
            std::printf("  graph launch=%s\n", cudaGetErrorString(launch));
            cudaGraphExecDestroy(exec);
        }
        cudaGraphDestroy(graph);
    }
    CHECK(cudaFree(buf));
    CHECK(cudaStreamDestroy(s));
}

void probe_cooperative(int dev) {
    int attr = 0;
    CHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, dev));
    int* buf = nullptr;
    CHECK(cudaMalloc(&buf, sizeof(int) * 2));
    CHECK(cudaMemset(buf, 0, sizeof(int) * 2));
    void* args[] = {&buf};
    cudaError_t e = cudaLaunchCooperativeKernel(
        static_cast<void*>(coop_kernel), dim3(2), dim3(64), args, 0, 0);
    CHECK(cudaDeviceSynchronize());
    int host[2] = {-1, -1};
    CHECK(cudaMemcpy(host, buf, sizeof(host), cudaMemcpyDeviceToHost));
    std::printf("  cooperativeLaunch attr=%d launch=%s blocks=%d\n", attr,
                cudaGetErrorString(e), host[0]);
    CHECK(cudaFree(buf));
}

void probe_device(int d) {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, d));
    std::printf("\n[dev %d] %s  cc=%d.%d\n", d, p.name, p.major, p.minor);
    std::printf("  SMs=%d clock=%d MHz  smem/block=%zuKB (optin=%dKB, perSM=%zuKB)  regs/block=%d\n",
                p.multiProcessorCount, p.clockRate / 1000, p.sharedMemPerBlock / 1024,
                p.sharedMemPerBlockOptin / 1024, p.sharedMemPerMultiprocessor / 1024,
                p.regsPerBlock);
    std::printf("  mem=%zuMB memclk=%d MHz bus=%d-bit -> ~%.0f GB/s  L2=%dKB\n",
                p.totalGlobalMem >> 20, p.memoryClockRate / 1000, p.memoryBusWidth,
                p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) * 2.0 / 1e9, p.l2CacheSize >> 10);
    std::printf("  asyncEngines=%d concurrentKernels=%d\n", p.asyncEngineCount,
                p.concurrentKernels ? 1 : 0);
    int hostNativeAtomic = 0, pageable = 0, umap = 0;
    CHECK(cudaDeviceGetAttribute(&hostNativeAtomic, cudaDevAttrHostNativeAtomicSupported, d));
    CHECK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, d));
    CHECK(cudaDeviceGetAttribute(&umap, cudaDevAttrDirectManagedMemAccessFromHost, d));
    std::printf("  hostNativeAtomic=%d pageableMemAccess=%d umapFromHost=%d\n",
                hostNativeAtomic, pageable, umap);
    probe_cooperative(d);
    probe_graph_and_copies(d);
}

}  // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "no CUDA devices\n");
        return 1;
    }
    std::printf("device_count=%d\n", count);
    for (int d = 0; d < count; ++d) {
        CHECK(cudaSetDevice(d));
        probe_device(d);
    }
    // TP2 transport decision matrix: every ordered pair. canAccess=1 with NVLink bridges the
    // SXM2/PCIe+bridge hosts; =0 means host-staged copies are the route (the engine probes
    // the same predicate at startup, so this output predicts startup transport selection).
    for (int a = 0; a < count; ++a) {
        for (int b = 0; b < count; ++b) {
            if (a == b) continue;
            int can = 0;
            CHECK(cudaDeviceCanAccessPeer(&can, a, b));
            std::printf("peer canAccess %d->%d = %d\n", a, b, can);
        }
    }
    if (argc >= 3) {
        const int a = std::atoi(argv[1]), b = std::atoi(argv[2]);
        if (a < count && b < count) {
            CHECK(cudaSetDevice(a));
            if (cudaDeviceEnablePeerAccess(b, 0) == cudaSuccess) {
                std::printf("enablePeerAccess %d->%d = OK (direct P2P active)\n", a, b);
                cudaDeviceDisablePeerAccess(b);
            } else {
                cudaGetLastError();
                std::printf("enablePeerAccess %d->%d = FAILED (host-staged required)\n", a, b);
            }
        }
    }
    return 0;
}
