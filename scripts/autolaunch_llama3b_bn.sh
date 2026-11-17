#!/bin/bash
# Wait for the Llama download, run the smoke test, then launch full training if the smoke test produced a checkpoint.
# Usage: nohup setsid bash scripts/autolaunch_llama3b_bn.sh > logs/autolaunch.log 2>&1 &
set -uo pipefail
OPSD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP_GLOB="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for model download"
until compgen -G "${SNAP_GLOB}/model-00001-of-00002.safetensors" >/dev/null \
      && compgen -G "${SNAP_GLOB}/model-00002-of-00002.safetensors" >/dev/null \
      && ! pgrep -u "$USER" -f "hf download meta-llama/Llama-3.2-3B-Instruct" >/dev/null; do
  if ! pgrep -u "$USER" -f "hf download meta-llama/Llama-3.2-3B-Instruct" >/dev/null \
     && ! compgen -G "${SNAP_GLOB}/model-00001-of-00002.safetensors" >/dev/null; then
    log "ERROR: download process exited but shard 1 is missing; re-run the download"; exit 1
  fi
  sleep 60
done
log "model download complete"

wait_job() {
  local state
  while true; do
    state="$(sacct -j "$1" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')"
    [[ -n "$state" && ! "$state" =~ ^(PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|SUSPENDED)$ ]] && break
    sleep 60
  done
  echo "$state"
}

SMOKE_RUN="smoke_$(date +%Y%m%d_%H%M%S)"
SMOKE_OUT="$(cd "${OPSD_DIR}" && SMOKE=1 RUN_CONFIG="${SMOKE_RUN}" bash scripts/submit_opsd_llama3b_bn.sh)"
SMOKE_JOB="$(awk '{print $2}' <<< "${SMOKE_OUT}")"
log "smoke submitted: ${SMOKE_OUT}"
STATE="$(wait_job "${SMOKE_JOB}")"
log "smoke job ${SMOKE_JOB} finished: ${STATE}"

if [[ "${STATE}" != "COMPLETED" ]] || ! compgen -G "${OPSD_DIR}/ckpts/opsd/${SMOKE_RUN}/checkpoint-*/adapter_model.safetensors" >/dev/null; then
  log "ERROR: smoke test failed; not launching training. See ${OPSD_DIR}/logs/opsd_llama_bn_${SMOKE_JOB}.log"
  exit 1
fi

TRAIN_OUT="$(cd "${OPSD_DIR}" && bash scripts/submit_opsd_llama3b_bn.sh)"
log "training submitted: ${TRAIN_OUT}"
