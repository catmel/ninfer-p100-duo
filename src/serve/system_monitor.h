#pragma once

// Host telemetry for the web panel's Monitor and About views: per-GPU NVML readings
// (utilization, memory, temperature) plus CPU load, CPU temperature and system memory.
// NVML is loaded lazily with dlopen so serving never links against the driver, and the
// Linux-only sources degrade to "not available" rows when a file or sensor is missing.

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// One device row. `available` is false when NVML could not read this device; individual figures
// carry their own flag, because a device may report temperature but no power (or the reverse).
struct GpuReading {
    int index               = 0;
    bool available          = false;
    std::string name;
    std::uint64_t mem_total = 0; // bytes
    std::uint64_t mem_used  = 0; // bytes
    std::uint32_t util      = 0; // percent
    std::uint32_t temp      = 0; // degrees Celsius
    bool power_available    = false;
    std::uint32_t power_mw  = 0; // milliwatts, as NVML reports it
    std::uint32_t power_limit_mw = 0; // 0 when the device exposes no limit
};

// Current host readings. Each optional figure carries its own flag, so a missing sensor is
// reported as missing rather than as a zero.
struct HostReading {
    bool cpu_util_available = false;
    double cpu_util         = 0.0; // percent, busy over the interval since the previous sample
    bool cpu_temp_available = false;
    double cpu_temp         = 0.0; // degrees Celsius
    std::uint64_t ram_total = 0;   // bytes
    std::uint64_t ram_used  = 0;   // bytes
};

// One sample of everything the panel shows.
struct SystemSample {
    std::vector<GpuReading> gpus;
    HostReading host;
};

// What does not change while the server runs.
struct HostStatic {
    bool nvml_available = false;
    std::string gpu_name;        // first readable device's name, for single-GPU display
    std::string cpu_name;
    unsigned threads = 0;
    std::string cpu_temp_source; // sensor the CPU temperature comes from; empty when none
};

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    SystemMonitor(const SystemMonitor&)            = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    // One reading per requested physical device index (NVML device indices: the server's
    // resolved device ids) plus the host figures. CPU load is the average over the interval
    // since the previous call, so the first call after construction reports it as unavailable.
    [[nodiscard]] SystemSample sample(const std::vector<int>& devices);

    [[nodiscard]] const HostStatic& host() const noexcept { return host_; }

private:
    struct Impl;
    Impl* impl_ = nullptr; // opaque NVML and /proc state; null outside Linux
    HostStatic host_;
};

} // namespace ninfer::serve
