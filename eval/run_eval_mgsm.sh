#!/bin/bash
# MGSM eval: base Llama-3.2 first, then every OPSD checkpoint (4K new tokens = training budget).
# Submit: sbatch --partition=a40 --qos=a40 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 \
#   --output=../logs/eval_mgsm_%j.log run_eval_mgsm.sh [ckpt_root]
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
CONDA_BASE=/home/speech-nlp-cse/24m0756/anaconda3
KYOTO_OPSD=/home/speech-nlp-cse/24m0756/abhishek/kyoto/OPSD_official
export PATH="${CONDA_BASE}/envs/slm/bin:${PATH}"
export PYTHONPATH="${KYOTO_OPSD}/sandbox_extra_slm:${KYOTO_OPSD}/sandbox_extra_vllm"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}
LANG_CODE=${LANG_CODE:-bn}
CKPT_ROOT=${1:-../ckpts/opsd/llama32_3b_gsm8k_bn_nonthink_fixteacher_temp11_forwardbeta0_clip1e-6_gen4k_genbatch16}
TP=${TP:-1}
# Official non-thinking eval samples at T=1.0 (top_p auto 0.8); VAL_N samples per problem
VAL_N=${VAL_N:-5}
TEMP=${TEMP:-1.0}
OUT_DIR=${OUT_DIR:-eval_results/mgsm_${LANG_CODE}_valn${VAL_N}_t${TEMP}_$(basename "$CKPT_ROOT")}
mkdir -p "$OUT_DIR"

COMMON_ARGS="--base_model $BASE_MODEL --dataset mgsm --mgsm_lang $LANG_CODE --no_thinking \
    --temperature $TEMP --val_n $VAL_N --max_new_tokens 4096 --max_model_len 8192 --tensor_parallel_size $TP \
    --gpu_memory_utilization 0.85"

python evaluate_math.py $COMMON_ARGS --output_file "$OUT_DIR/base.json"

for ckpt in $(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sort -V); do
    python evaluate_math.py $COMMON_ARGS --checkpoint_dir "$ckpt" --output_file "$OUT_DIR/$(basename "$ckpt").json"
done

python - "$OUT_DIR" <<'PY'
import json, glob, os, re, sys
files = sorted(glob.glob(f"{sys.argv[1]}/*.json"), key=lambda f: -1 if "base" in f else int(re.findall(r"\d+", os.path.basename(f))[0]))
print(f"{'model':<18}{'avg@n':>8}{'maj@n':>8}{'pass@n':>8}{'boxed':>8}{'bengali':>9}")
for f in files:
    d = json.load(open(f))
    print(f"{os.path.basename(f)[:-5]:<18}{d['average_at_n_pct']:>7.1f}%{d['majority_vote_at_n_pct']:>7.1f}%{d['pass_at_n_pct']:>7.1f}%{d['format_rate']:>7.1f}%{(d['lang_fidelity_pct'] or 0):>8.1f}%")
PY
