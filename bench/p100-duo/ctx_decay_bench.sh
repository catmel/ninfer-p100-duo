#!/bin/bash
# Context-length decay benchmark for NInfer P100 Duo (2x P100 TP2, containerized run).
# For each target prompt-token length 2048.. up to (MC - MAXNEW - margin), step STEPK:
# build a messages.json prompt of ~N tokens, run one CLI generation, record REAL prompt
# tokens + prefill/decode speed + MTP accept%. bytes-per-token calibrated from corpus.
# Usage: ctx_decay_bench.sh MODEL_NAME_WITHOUT_SUFFIX OUT.csv [MC=65536] [STEPK=2048] [MAXNEW=96] [START_K=STEPK]
# Assumes GPUs are free (service stopped) — caller's responsibility.
set -u
SRC=/vol1/1000/code/ninfer-p100-duo
MODELS=/vol1/1000/docker/ninfer-p100x2/models
MODEL_NAME=${1:?model name without .ninfer suffix}
OUT=${2:?output csv required}
MC=${3:-65536}
STEPK=${4:-2048}
MAXNEW=${5:-96}
START=${6:-$STEPK}
BPT=5.8   # measured bytes/token on this corpus (Q6: 21000B -> 3620 tok)
PD=$SRC/.ctxbench
mkdir -p "$PD"   # gitignored scratch (see .gitignore: .ctxbench/)
IMG=ninfer:p100-builder

CORPUS=$PD/corpus.txt
if [ ! -f "$CORPUS" ]; then
python3 - "$CORPUS" <<'PY'
import sys
text = ("the ring buffer index arithmetic wraps naturally because a power of two allows "
"masking instead of modulo and this property appears repeatedly in lock free queues "
"circular caches and sliding windows consider a buffer whose capacity is two to the k "
"then any position p maps to slot p bitwise and with k minus one which the compiler "
"turns into a single instruction the producer and consumer each maintain a monotonically "
"increasing counter and the distance between them never exceeds the capacity so the "
"masked slots never collide the memory ordering rules still require acquire release "
"semantics on the head and tail indices otherwise a reader could observe a stale payload "
"behind a published slot on architectures with weakly ordered stores an ordinary store "
"followed by a release store is cheaper than a full fence and the release publishes the "
"payload writes transitively padding the head and tail onto separate cache lines "
"prevents false sharing between the producer and consumer threads an idle spin should "
"back off with a nanosleep or pause instruction to avoid saturating the memory bus while "
"a reader waits for data that the writer has not produced yet finally the choice between "
"waiting with a condition variable and busy polling depends on the expected latency of "
"the producer relative to the cost of a context switch ")
with open(sys.argv[1], "w") as f:
    f.write(text * 4200)
PY
fi

echo "run,target_tokens,max_context,prompt_tokens,prefill_tps,decode_tps,generated,accept_n,accept_pct,wall_s,rc" > "$OUT"
i=0
LIMIT=$(( MC - MAXNEW - 256 ))
for t in $(seq "$START" "$STEPK" "$LIMIT"); do
  i=$((i+1))
  bytes=$(python3 -c "print(int($t * $BPT))")
  head -c "$bytes" "$CORPUS" > "$PD/p.txt"
  python3 - "$PD/p.txt" "$PD/m.json" <<'PY'
import json, sys
msg = [{"role": "user", "content": open(sys.argv[1]).read() + "\n\nSummarize the passage above in exactly five sentences."}]
json.dump(msg, open(sys.argv[2], "w"))
PY
  start=$(date +%s)
  log=$PD/run_${t}.log
  timeout 9000 docker run --rm --gpus all --shm-size 8g \
    -v "$SRC":/work -v "$MODELS":/workspace/models:ro --entrypoint bash "$IMG" -c \
    "/work/build-p100-duo/apps/ninfer /workspace/models/$MODEL_NAME.ninfer --tp 2 --devices 0,1 --no-thinking --spec mtp --draft-tokens 3 --max-new $MAXNEW --max-context $MC --kv-capacity auto --prefill-chunk 2048 --messages /work/.ctxbench/m.json" \
    > "$log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - start ))
  ptok=$(grep -oE 'prompt tokens[[:space:]]+[0-9]+' "$log" | head -1 | grep -oE '[0-9]+$')
  pre=$(grep -oE 'prefill speed[[:space:]]+[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
  dec=$(grep -oE 'decode speed[[:space:]]+[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
  gen=$(grep -oE 'generated tokens[[:space:]]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  accn=$(grep -oE 'accepted tokens[[:space:]]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  accp=$(grep 'accepted tokens' "$log" | tail -1 | grep -oE '\(([0-9.]+)%\)' | tr -d '()%')
  echo "$i,$t,$MC,${ptok:-NA},${pre:-NA},${dec:-NA},${gen:-NA},${accn:-NA},${accp:-NA},$wall,$rc" >> "$OUT"
  sync
  echo "[$i] tgt=$t prompt=${ptok:-NA} pre=${pre:-NA} dec=${dec:-NA} acc=${accn:-NA}/${accp:-NA} wall=${wall}s rc=$rc"
done
echo "DONE -> $OUT"
