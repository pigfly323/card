#!/bin/bash
#SBATCH --job-name=local-bm25-sft-8b
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/local-bm25-8gpu-%j.out

set -e

# Set up Java 21 (required for pyserini - needs jdk.incubator.vector from Java 16+)
export JAVA_HOME=/gpfs/projects/kohlab/rulins/env/dev
export JVM_PATH=/gpfs/projects/kohlab/rulins/env/dev/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH

# Change to the working directory
cd /gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct

# Source environment variables (API keys)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
model_path=rl-research/DR-Tulu-SFT-8B
dataset_list="s42chen/wii 1.0"
exp_name="local-bm25-dr-tulu-sft-8b"
output_dir="/gpfs/scrubbed/rulins/dr-tulu/local-bm25-sft-8b"

# Environment variables
export CRAWL4AI_BLOCKLIST_PATH=/gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct/crawl4ai_block_list.txt
export MCP_MAX_CONCURRENT_CALLS=512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RUBRIC_JUDGE_MODEL=gpt-4.1-mini
export MCP_CACHE_DIR=.cache-${RANDOM}
export MCP_TRANSPORT_PORT=8003
export USE_LOCAL_SEARCH=true

# Verbose error reporting
export PYTHONFAULTHANDLER=1

# Run 8-GPU training with local BM25 search
uv run python -u open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --wandb_project_name rl-rag \
        --with_tracking \
        --beta 0.001 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 5e-7 \
        --per_device_train_batch_size 1 \
        --output_dir ${output_dir} \
        --kl_estimator kl3 \
        --dataset_mixer_list ${dataset_list} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list s42chen/wii 16 \
        --dataset_mixer_eval_list_splits train \
        --apply_adaptive_rubric_reward false \
        --normalize_rubric_scores false \
        --use_rubric_buffer false \
        --use_static_rubrics_as_persistent_rubrics true \
        --max_active_rubrics 5 \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 18500 \
        --model_name_or_path ${model_path} \
        --non_stop_penalty False \
        --non_stop_penalty_value 0.0 \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 4 \
        --vllm_num_engines 4 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --num_evals 500 \
        --save_freq 50 \
        --try_launch_beaker_eval_jobs_on_weka False \
        --gradient_checkpointing \
        --max_tool_calls 20 \
        --only_reward_good_outputs False \
        --tools mcp \
        --checkpoint_state_freq 10 \
        --checkpoint_state_dir ${output_dir}/checkpoints \
        --keep_last_n_checkpoints 2 \
        --mcp_parser_name v20250824 \
        --system_prompt_file open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml \
        --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
        --mcp_server_command "uv run python -m dr_agent.mcp_backend.main --transport http --port 8003 --host 0.0.0.0 --path /mcp --local-searcher-type bm25 --index-path 'data/bm25' --dataset-name data/corpus.jsonl"

