// ctx_decay_run: one-load, continuously-generated context-decay sweep.
//
// Loads the Engine ONCE, then generates (prompt + max-new) tokens in a single run with EOS
// ignored, so the context grows continuously from the seed prompt to the requested ceiling.
// Every streamed OutputDelta is stamped with wall-clock; at completion the exact generated
// token count calibrates this run's bytes-per-token, and the delta timeline is binned into
// fixed token windows (default 2048) to report decode speed versus context length.
// This is the "one pass, record every +2K" methodology: no per-point reload, no restarts.
//
// Usage: ctx_decay_run ARTIFACT.csv-ish args:
//   ctx_decay_run <artifact.ninfer> <out.csv> <max_context> [step=2048] [tp=2] [seed_tokens=256]
// Flags fixed to production decode contract: --kv-capacity auto, prefill-chunk 2048,
// mtp spec draft 3, greedy, devices 0..tp-1, bf16 KV.
#include "product/prompt_input/prompt_input.h"
#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Stamp {
    std::size_t cum_bytes = 0;
    double t_seconds      = 0.0;
};

class TimingSink final : public ninfer::OutputSink {
public:
    void publish(ninfer::OutputDelta delta) override {
        cum_bytes_ += delta.text.size();
        stamps_.push_back(Stamp{cum_bytes_, std::chrono::duration<double>(Clock::now() - start_).count()});
    }
    void set_start(Clock::time_point s) { start_ = s; }
    std::size_t bytes() const { return cum_bytes_; }
    const std::vector<Stamp>& stamps() const { return stamps_; }
private:
    Clock::time_point start_{};
    std::size_t cum_bytes_ = 0;
    std::vector<Stamp> stamps_;
};

std::string build_seed_prompt(std::size_t approx_tokens) {
    // Repetitive-safe essay text; the model continues coherently for tens of thousands of
    // tokens without drifting into degenerate single-character loops.
    static const char* kPara =
        "Explain in detail, section by section, how a modern language server keeps code "
        "analysis correct while a document is edited: the incremental parse tree, the "
        "diagnostic pipeline, the semantic cache, indexing, position mapping, and the "
        "consistency guarantees at each boundary. Continue writing the next section. ";
    std::string out;
    const std::size_t para = 58;
    while (out.size() < approx_tokens * para / 10) { out += kPara; }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ctx_decay_run <artifact.ninfer> <out.csv> <max_context> [step=2048] [tp=2] [seed_tokens=256]\n");
        return 2;
    }
    const std::string artifact   = argv[1];
    const std::string out_csv    = argv[2];
    const std::uint32_t max_ctx  = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const std::size_t step       = argc > 4 ? static_cast<std::size_t>(std::stoul(argv[4])) : 2048;
    const int tp                 = argc > 5 ? std::stoi(argv[5]) : 2;
    const std::size_t seed_tokens= argc > 6 ? static_cast<std::size_t>(std::stoul(argv[6])) : 256;

    std::vector<int> devices;
    for (int i = 0; i < tp; ++i) { devices.push_back(i); }

    ninfer::EngineOptions eo;
    eo.artifact_path   = artifact;
    eo.device          = 0;
    eo.tp              = tp;
    eo.devices         = devices;
    eo.max_context     = max_ctx;
    eo.kv_capacity     = ninfer::KvCapacityPolicy::automatic();
    eo.prefill_chunk   = 2048;
    eo.speculative.backend        = ninfer::SpeculativeBackend::Mtp;
    eo.speculative.draft_tokens   = 3;
    // Full-vocabulary MTP artifacts (this fork's Q4/Q6 models) require the full proposal head.
    eo.speculative.proposal_head  = ninfer::ProposalHead::Full;
    eo.use_cuda_graph  = true;

    ninfer::Engine engine(std::move(eo));
    const auto& ls = engine.load_summary();
    std::fprintf(stderr, "loaded: target=%s weights=%s\n", ls.target.c_str(), ls.weights_id.c_str());
    std::fflush(stderr);

    ninfer::PromptInput input = ninfer::product::prompt_from_text(build_seed_prompt(seed_tokens), /*thinking=*/false);
    ninfer::RequestOptions ro;
    ro.stop.include_model_defaults = false;   // run to output budget, not EOS
    // max_new sized to drive context to (near) the ceiling; Engine clamps to capacity.
    ro.execution.requested_output_tokens = max_ctx > seed_tokens + 64 ? max_ctx - static_cast<std::uint32_t>(seed_tokens) - 64 : 16;

    ninfer::PreparedPrompt prompt = engine.prepare(std::move(input));
    const std::size_t prompt_tokens = prompt.summary().prompt_tokens;
    std::fprintf(stderr, "prompt tokens: %zu, generating %u tokens continuously...\n",
                 prompt_tokens, ro.execution.requested_output_tokens);
    std::fflush(stderr);

    TimingSink sink;
    const auto run_start = Clock::now();
    sink.set_start(run_start);
    ninfer::GenerationHandle handle = engine.submit(std::move(prompt), std::move(ro));
    const ninfer::GenerationResult result = handle.wait(&sink);
    const double wall = std::chrono::duration<double>(Clock::now() - run_start).count();

    const std::size_t n_tok = result.generated_token_ids.size();
    const std::size_t n_bytes = sink.bytes();
    const double bpt = n_tok > 0 ? static_cast<double>(n_bytes) / static_cast<double>(n_tok) : 5.8;
    std::fprintf(stderr, "generated=%zu bytes=%zu bpt=%.3f finish=%d decode_avg=%.2f tok/s\n",
                 n_tok, n_bytes, bpt, static_cast<int>(result.finish_reason),
                 result.timings.decode_seconds > 0 ? static_cast<double>(n_tok) / result.timings.decode_seconds : 0.0);

    // Bin: window k covers generated tokens [k*step, (k+1)*step). Map to delta stamps via
    // this-run bytes-per-token, take first stamp time entering each boundary.
    const auto& stamps = sink.stamps();
    std::ofstream csv(out_csv);
    csv << "window,ctx_start,ctx_end,decode_tps,cum_seconds,generated_at_window_end\n";
    std::size_t s_idx = 0;
    double prev_t = 0.0;
    std::size_t prev_tok = 0;
    for (std::size_t k = 1; k * step <= n_tok; ++k) {
        const std::size_t want_bytes = static_cast<std::size_t>(static_cast<double>(k * step) * bpt);
        while (s_idx + 1 < stamps.size() && stamps[s_idx + 1].cum_bytes <= want_bytes) { ++s_idx; }
        const double t = stamps[s_idx].t_seconds;
        const double dt = t - prev_t;
        const std::size_t dtok = k * step - prev_tok;
        const std::size_t ctx_start = prompt_tokens + prev_tok;
        if (dt > 0.0) {
            csv << k << ',' << ctx_start << ',' << (prompt_tokens + k * step) << ','
                << (static_cast<double>(dtok) / dt) << ',' << t << ',' << (k * step) << '\n';
        }
        prev_t = t; prev_tok = k * step;
    }
    // Partial tail window (keeps the max-ctx point even when the run stops early).
    if (n_tok > prev_tok) {
        csv << "tail," << (prompt_tokens + prev_tok) << ',' << (prompt_tokens + n_tok) << ",NA,NA," << n_tok << '\n';
    }
    csv.close();
    std::fprintf(stderr, "wrote %s\n", out_csv.c_str());
    return 0;
}
