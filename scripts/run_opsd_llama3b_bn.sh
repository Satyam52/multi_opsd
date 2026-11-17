#!/bin/bash
# OPSD on GSM8K (English) with "reason in Bengali"; eval target: MGSM-bn.
# Student = Llama-3.2 + LoRA, teacher = frozen base (LoRA disabled) seeing question + gold solution.
# Hyperparameters follow the official scripts/run_opsd_4b_nonthink.sh (vLLM colocate rollouts).
# Env: slm (torch 2.9, flash_attn) + kyoto sandbox_extra_slm (TRL 0.26) + sandbox_extra_vllm (real vLLM 0.11.2).
# Submit with scripts/submit_opsd_llama3b_bn.sh.
set -eo pipefail
OPSD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${OPSD_DIR}"

CONDA_BASE="${CONDA_BASE:-/home/speech-nlp-cse/24m0756/anaconda3}"
SLM_ENV="${SLM_ENV:-${CONDA_BASE}/envs/slm}"
# Compute nodes have no /usr/local/cuda; vllm-env only provides the nvcc DeepSpeed JIT needs
export CUDA_HOME="${CUDA_HOME:-${CONDA_BASE}/envs/vllm-env}"
export PATH="${SLM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${SLM_ENV}/lib:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
KYOTO_OPSD=/home/speech-nlp-cse/24m0756/abhishek/kyoto/OPSD_official
export PYTHONPATH="${KYOTO_OPSD}/sandbox_extra_slm:${KYOTO_OPSD}/sandbox_extra_vllm"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB__SERVICE_WAIT=300

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPSD_DIR}/ckpts/opsd}"
RUN_CONFIG="${RUN_CONFIG:-llama32_3b_gsm8k_bn_nonthink_fixteacher_temp11_forwardbeta0_clip1e-6_gen4k_genbatch16}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"  # 1 x 16 x 2 GPUs = 32 rollouts/step; batch 2 at 4K OOMs in the full-vocab loss on A40
# All 16 per-GPU rollouts of a step in one vLLM call (same weights), then backprop 1 at a time
GEN_BATCH="${GEN_BATCH:-16}"
MAX_STEPS="${MAX_STEPS:--1}"    # -1 = 1 full epoch of GSM8K train
SAVE_STEPS="${SAVE_STEPS:-25}"
# Official uses 1024; Bengali needs ~3-4x more Llama tokens than English
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
# Official uses 0.6 on 80GB H100s; on a 48GB A40 vLLM gets ~13GB (6.4GB weights + KV cache)
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.30}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "OPSD Llama-3.2 GSM8K->Bengali | model ${MODEL} | out ${OUTPUT_DIR}/${RUN_CONFIG}"
echo "gpus ${NUM_GPUS} | bs ${BATCH_SIZE} accum ${GRAD_ACCUM} gen_batch ${GEN_BATCH} | max_steps ${MAX_STEPS} save ${SAVE_STEPS} | completion ${MAX_COMPLETION_LENGTH}"
python -c "from transformers.utils import is_flash_attn_2_available as f; assert f(); import vllm; print('flash_attn ok, vllm', vllm.__version__)"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes $NUM_GPUS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --main_process_port "${MAIN_PROCESS_PORT:-$((12000 + RANDOM % 2000))}" \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --dataset_name openai/gsm8k \
    --target_language Bengali \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --generation_batch_size $GEN_BATCH \
    --gradient_checkpointing \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --output_dir $OUTPUT_DIR \
    --run_config $RUN_CONFIG \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --student_thinking False \
    --teacher_thinking False \
    --jsd_token_clip 1e-6 \
    --wandb_project OPSD-indic
