#!/bin/bash
# Submit run_opsd_llama3b_bn.sh to SLURM (a40 QOS allows at most 2 GPUs/job).
# Smoke test: SMOKE=1 bash scripts/submit_opsd_llama3b_bn.sh
set -eo pipefail
OPSD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_NAME=opsd_llama_bn

if [[ "${SMOKE:-0}" == "1" ]]; then
  JOB_NAME=opsd_llama_bn_smoke
  export MAX_STEPS=2 SAVE_STEPS=2
  export RUN_CONFIG="${RUN_CONFIG:-smoke_${STAMP}}"
  TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${OPSD_DIR}/ckpts/opsd}"
mkdir -p "${OPSD_DIR}/logs"

J=$(sbatch --parsable --job-name="${JOB_NAME}" \
      --partition="${PARTITION:-a40}" --qos="${QOS:-a40}" \
      --gres=gpu:"${NUM_GPUS:-2}" --cpus-per-task=16 --mem=128G --time="${TIME_LIMIT:-2-00:00:00}" \
      --output="${OPSD_DIR}/logs/opsd_llama_bn_%j.log" --export=ALL \
      --wrap="bash ${OPSD_DIR}/scripts/run_opsd_llama3b_bn.sh")
echo "job ${J} | log ${OPSD_DIR}/logs/opsd_llama_bn_${J}.log"
