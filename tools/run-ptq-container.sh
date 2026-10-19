#!/bin/sh
# Run the PTQ driver inside the quantibbler-ptq container.
#
# The sandbox's 5GiB cgroup killed the in-process runs (exit 137); the docker
# daemon lives OUTSIDE that cgroup, so this runs against host RAM with a sane
# explicit ceiling (--memory 32g of ~61g available; swap disabled so the
# failure mode is a clean in-container OOM kill, not host thrash).
#
# Mounts:
#   /work  = repo (ro)          /hf = HF cache (models + wikitext parquet)
#   /tmp/ptq = real-disk scratch (KD teacher-logit memmap lands here, 20GB class)
set -eu

REPO=/opt/data/quantibbler
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
OUT="${OUT:-/opt/data/quantibbler/artifacts/qwen3-0.6B-r8}"
RANK="${RANK:-8}"

mkdir -p /opt/data/tmp "$(dirname "$OUT")"
: > /opt/data/tmp/quantibbler-breadcrumbs.log

exec docker run --rm \
  --name quantibbler-ptq-run \
  --memory 32g --memory-swap 32g \
  --cpus 24 \
  -v "$REPO:/work:ro" \
  -v /opt/data/hf-cache:/hf \
  -v /opt/data/tmp:/tmp/ptq \
  -v "$OUT:/out" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log \
  -e OMP_NUM_THREADS=16 \
  -e MKL_NUM_THREADS=16 \
  -e OPENBLAS_NUM_THREADS=16 \
  quantibbler-ptq:latest \
  python tools/quantize_hf_model.py \
    --model "$MODEL" --rank "$RANK" \
    --seq-len 512 --calibration-seqs 16 --eval-seqs 5 \
    --max-admm-iter 400 \
    --checkpoint-dir /out/checkpoints \
    --out /out
