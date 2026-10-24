#!/bin/bash
set -uo pipefail
SRC=/vol1/1000/code/ninfer-p100-duo
IMG=ninfer:p100-builder
M=/workspace/models/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer
P='Explain the tradeoffs of a power-of-two ring buffer, then write C code.'

echo "=== 0. stop serving container (free VRAM) ==="
docker stop ninfer-p100x2 >/dev/null 2>&1 || true
sleep 2

echo "=== 1. incremental rebuild (gguf pingpong + rows4 + a16) ==="
docker run --rm -v "$SRC":/work --entrypoint bash "$IMG" -c '
cd /work && cmake --build build-p100-duo -j 8 2>&1 | tail -3' | tee /tmp/v9_rebuild.txt
grep -E 'error|FAILED' /tmp/v9_rebuild.txt && { echo REBUILD_FAIL; docker start ninfer-p100x2 >/dev/null; exit 1; }

echo "=== 2. rebuild oracles, run GGUF-native + groupwise oracles on empty GPU ==="
docker run --rm -v "$SRC":/work --gpus all --entrypoint bash "$IMG" -c '
set -e
cd /work
for t in oracle_gguf_vector_check oracle_decode_check p100_speed_bench; do
  nvcc -O3 -std=c++17 -arch=sm_60 --expt-relaxed-constexpr -DNINFER_PASCAL_BUILD -I src -I include \
    tools/p100/$t.cu build-p100-duo/src/libninfer_ops.a build-p100-duo/src/libninfer_core.a \
    -o tools_bin/$t 2>&1 | grep -E "error" | head -5 || true
done
/work/tools_bin/oracle_gguf_vector_check > /work/tools_bin/oracle_gguf_v9.log 2>&1; echo "GGUF exit=$?"
tail -3 /work/tools_bin/oracle_gguf_v9.log
echo "gguf PASS_count=$(grep -c PASS /work/tools_bin/oracle_gguf_v9.log) FAIL=$(grep -c FAIL /work/tools_bin/oracle_gguf_v9.log)"
/work/tools_bin/oracle_decode_check > /work/tools_bin/oracle_v9.log 2>&1 || true
tail -1 /work/tools_bin/oracle_v9.log; echo "decode PASS_count=$(grep -c PASS /work/tools_bin/oracle_v9.log)"'

echo "=== 3. runtime image ==="
docker build -q -t ninfer:p100-duo -f - "$SRC/build-p100-duo/apps" <<'EOF'
FROM ninfer:p100-builder
COPY ninfer /usr/local/bin/ninfer
COPY ninfer-serve /usr/local/bin/ninfer-serve
EOF
echo IMAGE_EXIT=$?

echo "=== 4. draft-3 A/B x3 (compare vs 32.12 baseline) ==="
docker run --rm --gpus all --shm-size 8g \
  -v "$SRC":/work -v /vol1/1000/docker/ninfer-v100x2/models:/workspace/models:ro \
  --entrypoint bash ninfer:p100-builder -c "
for i in 1 2 3; do
  echo \"[run \$i]\"
  /work/build-p100-duo/apps/ninfer $M --tp 2 --devices 0,1 --no-thinking --spec mtp --draft-tokens 3 --max-new 160 --prompt \"$P\" 2>&1 | grep -E 'decode speed|prefill speed|generated tokens'
done"

echo "=== 5. compose recreate ==="
cd /vol1/1000/docker/ninfer-p100x2 && docker compose up -d --force-recreate 2>&1 | tail -2
for i in $(seq 1 40); do sleep 5; docker logs --since 3m ninfer-p100x2 2>&1 | grep -q 'listening on' && { echo SERVICE_READY; break; }; done
echo DEPLOY_V9_DONE
