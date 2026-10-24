#pragma once

#include "serve/generation_service.h"
#include "serve/response_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"
#include "serve/system_monitor.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_metrics(const httplib::Request& req, httplib::Response& res);
    void handle_web_asset(const httplib::Request& req, httplib::Response& res) const;

    // The process-wide console logger serializes lines from request and reporter threads.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_rejected(const RequestRejectionLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);
    void log_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    ResponseStore response_store_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;

    // Web panel: engine snapshots captured at attach() plus the /metrics sampling state.
    ninfer::LoadSummary load_summary_{};
    ninfer::MemorySummary memory_summary_{};
    bool has_load_summary_ = false;
    SystemMonitor system_monitor_;
    std::mutex monitor_mutex_;
    bool monitor_has_baseline_ = false;
    std::chrono::steady_clock::time_point monitor_baseline_time_{};
    ninfer::RuntimeStats monitor_baseline_{};
    double monitor_tok_s_         = 0.0;
    double monitor_prefill_tok_s_ = 0.0;
    // Last completed request's measured rates, so the panel still shows a prefill speed for a
    // prompt the sub-second sampling interval never caught.
    RequestRates last_rates_{};
    bool has_last_rates_ = false;
    // Fixed-size rings of the last kMonitorHistorySamples /metrics observations, served back as
    // the panel's sparkline series.
    static constexpr std::size_t kMonitorHistorySamples = 120;
    struct MonitorRing {
        std::deque<double> values;
        void push(double value) {
            values.push_back(value);
            if (values.size() > kMonitorHistorySamples) { values.pop_front(); }
        }
    };
    MonitorRing ring_tok_s_;
    MonitorRing ring_prefill_tok_s_;
    MonitorRing ring_cpu_util_;
    MonitorRing ring_cpu_temp_;
    MonitorRing ring_ram_used_;
};

} // namespace ninfer::serve
