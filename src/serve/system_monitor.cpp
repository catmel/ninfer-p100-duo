#include "serve/system_monitor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace ninfer::serve {
namespace {

#if defined(__linux__)

// Minimal NVML surface, resolved at runtime. Keeping local struct/typedef declarations (instead
// of including nvml.h) avoids a build-time dependency on the NVIDIA management library.
using NvmlDevice_t = struct nvmlDevice_st*;

constexpr unsigned kNvmlSuccess        = 0;
constexpr unsigned kNvmlNameBufferSize = 64; // NVML_DEVICE_NAME_BUFFER_SIZE

struct NvmlApi {
    unsigned (*init)(void)                                        = nullptr;
    unsigned (*shutdown)(void)                                    = nullptr;
    unsigned (*device_get_count)(unsigned*)                       = nullptr;
    unsigned (*device_get_handle)(unsigned, NvmlDevice_t*)        = nullptr;
    unsigned (*device_get_name)(NvmlDevice_t, char*, unsigned)    = nullptr;
    unsigned (*device_get_memory_v2)(NvmlDevice_t, void*)         = nullptr;
    unsigned (*device_get_memory)(NvmlDevice_t, void*)            = nullptr;
    unsigned (*device_get_utilization)(NvmlDevice_t, void*)       = nullptr;
    unsigned (*device_get_temperature)(NvmlDevice_t, int, unsigned*) = nullptr;
    // Power is optional: not every device reports it, so a missing symbol or a NOT_SUPPORTED
    // answer leaves the reading unavailable instead of failing the sample.
    unsigned (*device_get_power_usage)(NvmlDevice_t, unsigned*)       = nullptr;
    unsigned (*device_get_enforced_power_limit)(NvmlDevice_t, unsigned*) = nullptr;
    unsigned (*device_get_power_management_limit)(NvmlDevice_t, unsigned*) = nullptr;
    void* library = nullptr;
};

// nvmlMemory_v2_t. Its `version` field must carry NVML's struct-version encoding, or the call
// is rejected with INVALID_ARGUMENT and the memory reading silently stays zero.
struct NvmlMemoryV2 {
    unsigned int version        = 0;
    unsigned long long total    = 0;
    unsigned long long reserved = 0;
    unsigned long long free     = 0;
    unsigned long long used     = 0;
};

// nvmlMemory_t: the unversioned original, still served by current drivers.
struct NvmlMemoryV1 {
    unsigned long long total = 0;
    unsigned long long free  = 0;
    unsigned long long used  = 0;
};

struct NvmlUtilization {
    std::uint32_t gpu    = 0;
    std::uint32_t memory = 0;
};

constexpr unsigned kNvmlMemoryV2Version =
    static_cast<unsigned>(sizeof(NvmlMemoryV2)) | (2U << 24U);

bool load_nvml(NvmlApi& api) {
    for (const char* name : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
        api.library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (api.library != nullptr) { break; }
    }
    if (api.library == nullptr) { return false; }
    auto sym = [&](const char* name) -> void* { return dlsym(api.library, name); };
    api.init                   = reinterpret_cast<decltype(api.init)>(sym("nvmlInit_v2"));
    api.shutdown               = reinterpret_cast<decltype(api.shutdown)>(sym("nvmlShutdown"));
    api.device_get_count       = reinterpret_cast<decltype(api.device_get_count)>(sym("nvmlDeviceGetCount_v2"));
    api.device_get_handle      = reinterpret_cast<decltype(api.device_get_handle)>(sym("nvmlDeviceGetHandleByIndex_v2"));
    api.device_get_name        = reinterpret_cast<decltype(api.device_get_name)>(sym("nvmlDeviceGetName"));
    api.device_get_memory_v2   = reinterpret_cast<decltype(api.device_get_memory_v2)>(sym("nvmlDeviceGetMemoryInfo_v2"));
    api.device_get_memory      = reinterpret_cast<decltype(api.device_get_memory)>(sym("nvmlDeviceGetMemoryInfo"));
    api.device_get_utilization = reinterpret_cast<decltype(api.device_get_utilization)>(sym("nvmlDeviceGetUtilizationRates"));
    api.device_get_temperature = reinterpret_cast<decltype(api.device_get_temperature)>(sym("nvmlDeviceGetTemperature"));
    api.device_get_power_usage = reinterpret_cast<decltype(api.device_get_power_usage)>(sym("nvmlDeviceGetPowerUsage"));
    api.device_get_enforced_power_limit =
        reinterpret_cast<decltype(api.device_get_enforced_power_limit)>(sym("nvmlDeviceGetEnforcedPowerLimit"));
    api.device_get_power_management_limit = reinterpret_cast<decltype(api.device_get_power_management_limit)>(
        sym("nvmlDeviceGetPowerManagementLimit"));
    if (api.init == nullptr || api.shutdown == nullptr || api.device_get_count == nullptr ||
        api.device_get_handle == nullptr || api.device_get_name == nullptr ||
        (api.device_get_memory_v2 == nullptr && api.device_get_memory == nullptr) ||
        api.device_get_utilization == nullptr || api.device_get_temperature == nullptr) {
        return false;
    }
    return api.init() == kNvmlSuccess;
}

std::string read_first_line(const char* path) {
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) { return {}; }
    std::string line;
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), file) != nullptr) { line.append(buffer); }
    std::fclose(file);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
    return line;
}

std::string cpu_model_name() {
    std::FILE* file = std::fopen("/proc/cpuinfo", "r");
    if (file == nullptr) { return {}; }
    char line[512];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::strncmp(line, "model name", 10) == 0) {
            const char* colon = std::strchr(line, ':');
            std::fclose(file);
            if (colon == nullptr) { return {}; }
            std::string name(colon + 1);
            while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) { name.pop_back(); }
            while (!name.empty() && name.front() == ' ') { name.erase(name.begin()); }
            return name;
        }
    }
    std::fclose(file);
    return {};
}

// CPU jiffies from /proc/stat's aggregate `cpu` line: total work and the idle share (idle plus
// iowait, which is time the CPU had nothing to run).
struct CpuJiffies {
    bool valid    = false;
    std::uint64_t total = 0;
    std::uint64_t idle  = 0;
};

CpuJiffies read_cpu_jiffies() {
    CpuJiffies out;
    std::FILE* file = std::fopen("/proc/stat", "r");
    if (file == nullptr) { return out; }
    char line[512];
    if (std::fgets(line, sizeof(line), file) == nullptr || std::strncmp(line, "cpu ", 4) != 0) {
        std::fclose(file);
        return out;
    }
    std::fclose(file);
    const char* cursor = line + 4;
    for (int field = 0; field < 8; ++field) {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end == cursor) { break; }
        out.total += value;
        // 4th and 5th fields are idle and iowait.
        if (field == 3 || field == 4) { out.idle += value; }
        cursor = end;
    }
    out.valid = out.total != 0;
    return out;
}

// MemTotal and MemAvailable from /proc/meminfo, in bytes. Available (not free) is what a user
// reads as "used": it already excludes reclaimable page cache.
bool read_memory(std::uint64_t& total, std::uint64_t& used) {
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) { return false; }
    std::uint64_t total_kb = 0, available_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        unsigned long long value = 0;
        if (std::sscanf(line, "MemTotal: %llu kB", &value) == 1) {
            total_kb = value;
        } else if (std::sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
            available_kb = value;
        }
    }
    std::fclose(file);
    if (total_kb == 0 || available_kb > total_kb) { return false; }
    total = total_kb * 1024ULL;
    used  = (total_kb - available_kb) * 1024ULL;
    return true;
}

// The package temperature: hwmon drivers first (coretemp on Intel, k10temp/zenpower on AMD),
// then the ACPI x86_pkg_temp thermal zone. Only drivers whose temp1_input is the package sensor
// are used -- on coretemp temp1 is "Package id 0" and temp2..N are individual cores.
struct CpuTempSensor {
    std::string path;  // file holding millidegrees Celsius
    std::string label; // human-readable source
};

CpuTempSensor find_cpu_temp_sensor() {
    for (int index = 0; index < 16; ++index) {
        const std::string dir = "/sys/class/hwmon/hwmon" + std::to_string(index);
        const std::string name = read_first_line((dir + "/name").c_str());
        if (name != "coretemp" && name != "k10temp" && name != "zenpower") { continue; }
        const std::string input = dir + "/temp1_input";
        if (std::FILE* probe = std::fopen(input.c_str(), "r")) {
            std::fclose(probe);
            return CpuTempSensor{.path = input, .label = name + " package"};
        }
    }
    for (int index = 0; index < 16; ++index) {
        const std::string dir = "/sys/class/thermal/thermal_zone" + std::to_string(index);
        const std::string type = read_first_line((dir + "/type").c_str());
        if (type.find("pkg") == std::string::npos && type.find("cpu") == std::string::npos) {
            continue;
        }
        const std::string input = dir + "/temp";
        if (std::FILE* probe = std::fopen(input.c_str(), "r")) {
            std::fclose(probe);
            return CpuTempSensor{.path = input, .label = type};
        }
    }
    return {};
}

#endif // __linux__

} // namespace

struct SystemMonitor::Impl {
#if defined(__linux__)
    NvmlApi api;
    unsigned device_count = 0;
    CpuJiffies previous_cpu;
    CpuTempSensor cpu_temp;
#endif
};

SystemMonitor::SystemMonitor() {
    host_.threads = std::thread::hardware_concurrency();
#if defined(__linux__)
    impl_          = new Impl();
    host_.cpu_name = cpu_model_name();
    impl_->previous_cpu   = read_cpu_jiffies();
    impl_->cpu_temp       = find_cpu_temp_sensor();
    host_.cpu_temp_source = impl_->cpu_temp.label;
    if (load_nvml(impl_->api) && impl_->api.device_get_count(&impl_->device_count) == kNvmlSuccess) {
        host_.nvml_available = true;
        NvmlDevice_t device  = nullptr;
        char name[kNvmlNameBufferSize]{};
        if (impl_->device_count > 0 &&
            impl_->api.device_get_handle(0, &device) == kNvmlSuccess &&
            impl_->api.device_get_name(device, name, sizeof(name)) == kNvmlSuccess) {
            host_.gpu_name = name;
        }
    }
#endif
}

SystemMonitor::~SystemMonitor() {
#if defined(__linux__)
    if (impl_ != nullptr) {
        if (impl_->api.shutdown != nullptr) { impl_->api.shutdown(); }
        if (impl_->api.library != nullptr) { dlclose(impl_->api.library); }
        delete impl_;
    }
#else
    (void)impl_;
#endif
}

SystemSample SystemMonitor::sample(const std::vector<int>& devices) {
    SystemSample out;
    out.gpus.reserve(devices.size());
#if defined(__linux__)
    if (impl_ == nullptr) {
        for (const int index : devices) { out.gpus.push_back(GpuReading{.index = index}); }
        return out;
    }

    // Host memory.
    std::uint64_t ram_total = 0, ram_used = 0;
    if (read_memory(ram_total, ram_used)) {
        out.host.ram_total = ram_total;
        out.host.ram_used  = ram_used;
    }
    // CPU load, averaged over the interval since the previous sample.
    const CpuJiffies now = read_cpu_jiffies();
    if (now.valid && impl_->previous_cpu.valid) {
        const std::uint64_t total_delta = now.total - impl_->previous_cpu.total;
        const std::uint64_t idle_delta  = now.idle - impl_->previous_cpu.idle;
        if (total_delta > 0 && idle_delta <= total_delta) {
            out.host.cpu_util =
                100.0 * static_cast<double>(total_delta - idle_delta) / static_cast<double>(total_delta);
            out.host.cpu_util_available = true;
        }
    }
    if (now.valid) { impl_->previous_cpu = now; }
    // CPU package temperature, in millidegrees on both sources.
    if (!impl_->cpu_temp.path.empty()) {
        const std::string raw = read_first_line(impl_->cpu_temp.path.c_str());
        if (!raw.empty()) {
            const long millidegrees = std::strtol(raw.c_str(), nullptr, 10);
            if (millidegrees > 0) {
                out.host.cpu_temp           = static_cast<double>(millidegrees) / 1000.0;
                out.host.cpu_temp_available = true;
            }
        }
    }

    if (!host_.nvml_available) {
        for (const int index : devices) { out.gpus.push_back(GpuReading{.index = index}); }
        return out;
    }
    for (const int index : devices) {
        GpuReading reading;
        reading.index = index;
        do {
            if (index < 0 || static_cast<unsigned>(index) >= impl_->device_count) { break; }
            NvmlDevice_t device = nullptr;
            if (impl_->api.device_get_handle(static_cast<unsigned>(index), &device) != kNvmlSuccess) {
                break;
            }
            char name[kNvmlNameBufferSize]{};
            if (impl_->api.device_get_name(device, name, sizeof(name)) == kNvmlSuccess) {
                reading.name = name;
            }
            NvmlMemoryV2 memory_v2{};
            memory_v2.version = kNvmlMemoryV2Version;
            bool have_memory  = impl_->api.device_get_memory_v2 != nullptr &&
                               impl_->api.device_get_memory_v2(device, &memory_v2) == kNvmlSuccess;
            if (have_memory) {
                reading.mem_total = memory_v2.total;
                reading.mem_used  = memory_v2.used;
            } else if (impl_->api.device_get_memory != nullptr) {
                NvmlMemoryV1 memory_v1{};
                if (impl_->api.device_get_memory(device, &memory_v1) == kNvmlSuccess) {
                    reading.mem_total = memory_v1.total;
                    reading.mem_used  = memory_v1.used;
                }
            }
            NvmlUtilization utilization{};
            if (impl_->api.device_get_utilization(device, &utilization) == kNvmlSuccess) {
                reading.util = utilization.gpu;
            }
            unsigned temp = 0;
            if (impl_->api.device_get_temperature(device, 0 /* NVML_TEMPERATURE_GPU */, &temp) ==
                kNvmlSuccess) {
                reading.temp = temp;
            }
            unsigned milliwatts = 0;
            if (impl_->api.device_get_power_usage != nullptr &&
                impl_->api.device_get_power_usage(device, &milliwatts) == kNvmlSuccess) {
                reading.power_mw        = milliwatts;
                reading.power_available = true;
            }
            unsigned limit_mw = 0;
            if (impl_->api.device_get_enforced_power_limit != nullptr &&
                impl_->api.device_get_enforced_power_limit(device, &limit_mw) == kNvmlSuccess) {
                reading.power_limit_mw = limit_mw;
            } else if (impl_->api.device_get_power_management_limit != nullptr &&
                       impl_->api.device_get_power_management_limit(device, &limit_mw) == kNvmlSuccess) {
                reading.power_limit_mw = limit_mw;
            }
            reading.available = true;
        } while (false);
        out.gpus.push_back(reading);
    }
#else
    for (const int index : devices) { out.gpus.push_back(GpuReading{.index = index}); }
#endif
    return out;
}

} // namespace ninfer::serve
