#!/bin/bash
# Full baseline qualification for ninfer-p100-duo on 2x Tesla P100 NVLink.
# Runs: ctest suite, GGUF native oracle, decode regression, per-model perf A/B,
# serve HTTP smoke, then 2K..128K context decay for each model. GPU-exclusive.
set -u
SRC=/vol1/1000/code/ninfer-p100-duo
BUILD=$SRC/build-p100-duo
MODELS_DIR=/vol1/1000/docker/ninfer-p100x2/models
OUT=/vol1/1000/bench/p100-duo
TAG=qual-$(date +%Y%m%d)
D=$OUT/$TAG
mkdir -p "$D"
COMPOSE=/vol1/1000/docker/ninfer-p100x2
say() { echo "[qual] $(date +%H:%M:%S) $*" | tee -a "$D/summary.txt"; }
BR() { docker run --rm --gpus all --shm-size 8g -v "$SRC":/work -v "$MODELS_DIR":/workspace/models:ro \
       --entrypoint bash ninfer:p100-builder -c "$1"; }

gpu_stop()  { cd "$COMPOSE" && docker compose stop >/dev/null 2>&1; sleep 20; }
gpu_start() { cd "$COMPOSE" && docker compose up -d >/dev/null 2>&1; }

say "=== E0 hardware ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv > "$D/hardware.csv"
nvidia-smi topo -m > "$D/topology.txt" 2>&1
nvidia-smi nvlink -s >> "$D/topology.txt" 2>&1
docker run --rm --entrypoint nvcc ninfer:p100-builder --version | tail -1 >> "$D/hardware.csv"
say "$(tr '\n' ';' < "$D/hardware.csv")"
say "nvlink: $(tr '\n' ';' < "$D/topology.txt" | head -c 300)"

say "=== E1 ctest ==="
gpu_stop
BR 'cd /work/build-p100-duo && ctest --output-on-failure -j 4' > "$D/ctest.log" 2>&1
rc=$?; tail -3 "$D/ctest.log" | head -2 >> "$D/summary.txt"
[ $rc -eq 0 ] && say "ctest PASS (rc0)" || say "ctest rc=$rc — see ctest.log failures"

say "=== E2 GGUF native oracle (FP64 cases + valid-column bounds) ==="
BR 'set -e; cd /work; mkdir -p tools_bin
for t in oracle_gguf_vector_check oracle_decode_check; do
  nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -DNINFER_PASCAL_BUILD -I src -I include \
    tools/p100/$t.cu build-p100-duo/src/libninfer_ops.a build-p100-duo/src/libninfer_core.a \
    -o tools_bin/$t 2>&1 | grep -E "error" | head -5 || true
done' > "$D/oracle_build.log" 2>&1
say "oracle build rc=$? ($(grep -ci error "$D/oracle_build.log" || true) errors)"
BR '/work/tools_bin/oracle_gguf_vector_check' > "$D/oracle_gguf_vector.log" 2>&1
say "oracle_gguf_vector rc=$? PASS=$(grep -c PASS "$D/oracle_gguf_vector.log") FAIL=$(grep -c FAIL "$D/oracle_gguf_vector.log")"

say "=== E3 decode regression ==="
BR '/work/tools_bin/oracle_decode_check' > "$D/oracle_decode.log" 2>&1
say "oracle_decode rc=$? PASS=$(grep -c PASS "$D/oracle_decode.log") FAIL=$(grep -c FAIL "$D/oracle_decode.log")"

say "=== E4 perf A/B x3 per model (draft-3, 160 tok) ==="
for M in "$MODELS_DIR"/*.ninfer; do
  NAME=$(basename "$M" .ninfer)
  for i in 1 2 3; do
    BR "/work/build-p100-duo/apps/ninfer /workspace/models/$NAME.ninfer --tp 2 --devices 0,1 --no-thinking --spec mtp --draft-tokens 3 --max-new 160 --prompt 'Explain the tradeoffs of a power-of-two ring buffer, then write C code.'" > "$D/cli_${NAME}_$i.txt" 2>&1
    dec=$(grep -oE 'decode speed[[:space:]]+[0-9.]+' "$D/cli_${NAME}_$i.txt" | grep -oE '[0-9.]+$' | tail -1)
    pre=$(grep -oE 'prefill speed[[:space:]]+[0-9.]+' "$D/cli_${NAME}_$i.txt" | grep -oE '[0-9.]+$' | tail -1)
    acc=$(grep -oE 'accepted [0-9.]+%' "$D/cli_${NAME}_$i.txt" | grep -oE '[0-9.]+' | tail -1)
    say "perf $NAME run$i decode=${dec:-NA} prefill=${pre:-NA} accept=${acc:-NA}%"
    echo "$NAME,$i,${pre:-NA},${dec:-NA},${acc:-NA}" >> "$D/perf_ab.csv"
  done
done

say "=== E5 serve HTTP smoke (OpenAI route) ==="
gpu_start
for i in $(seq 1 40); do sleep 8; curl -s --max-time 4 http://127.0.0.1:8088/v1/models | grep -q qwen && break; done
curl -s http://127.0.0.1:8088/v1/models > "$D/serve_models.json"
curl -s --max-time 400 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17*23? Reply with the number only."}],"max_tokens":24,"temperature":0}' > "$D/serve_chat.json"
grep -q '"content"' "$D/serve_chat.json" && say "serve chat OK: $(grep -oE '[0-9]+' "$D/serve_chat.json" | tail -1)" || say "serve chat FAIL"

say "=== E6 context decay 2K..128K per model ==="
gpu_stop
for M in "$MODELS_DIR"/*.ninfer; do
  NAME=$(basename "$M" .ninfer)
  bash /tmp/ctx_decay_bench.sh "$NAME" "$D/decay_${NAME}.csv" 128 2 96 > "$D/decay_${NAME}.log" 2>&1
  say "decay $NAME done rows=$(wc -l < "$D/decay_${NAME}.csv")"
done
say "=== ALL DONE $TAG ==="
