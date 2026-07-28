# `train_dr_tulu_8b_lff.sh` Training README

This document explains the purpose of `rl/open-instruct/train_dr_tulu_8b_lff.sh`, gives a brief project overview, describes the required environment, and shows how to run both the **LFF-based training script** and the **standard GRPO baseline script**.

---

## 1. Project Overview


This project is a reinforcement learning training project for deep-research / tool-using agents. The `open_instruct` code in this repository extends the original [Open-Instruct](https://github.com/allenai/open-instruct) training stack with support for:

- multi-tool interaction trajectories
- rubric-based and citation-based rewards
- `verify_answer`-aware training
- GRPO-based agent optimization
- Ray + DeepSpeed + vLLM distributed training

Typical tools used during training include:

- `google_search`
- `snippet_search`
- `browse_webpage`
- `verify_answer`

The goal is to train an agent that can search, reason, self-check with `verify_answer`, and improve final answer quality under rubric and citation rewards.

---

## 2. What `train_dr_tulu_8b_lff.sh` Does

The script:

1. configures temporary directories, Ray runtime variables, and `PYTHONPATH`
2. sets training-related environment variables
3. specifies:
   - the SFT initialization model
   - training / evaluation datasets
   - judge models and judge endpoints
   - MCP tool service settings
   - LFF and `verify_answer`-related options
4. launches:

```bash
python open_instruct/grpo_lff.py
```

5. runs a multi-node, multi-GPU RL training job

This is a **production-style experiment script**, not a minimal example. It assumes internal paths, a functioning Ray cluster, tool-serving infrastructure, and reward-judge connectivity.

---

## 3. LFF Training Features

The script enables **LFF** through:

```bash
--lff_enabled true
```

At a high level, LFF introduces `verify_answer`-aware training logic:

- it separates trajectory segments before and after `verify_answer`
- it uses draft-answer reward `r0` and final-answer reward `r1`
- it computes different advantages for different trajectory segments
- it can reweight / distill post-feedback reasoning on non-degenerate examples

This is the main difference from a plain GRPO baseline.

---

## 4. Rubric and Citation Reward

The script enables verifiable reward with:

```bash
--apply_verifiable_reward true
```

and uses judge backends configured via environment variables such as:

- `RUBRIC_JUDGE_MODEL`
- `CITATION_JUDGE_MODEL`
- `RUBRIC_JUDGE_API_BASE`
- `CITATION_JUDGE_API_BASE`

These judges are used to score:

- rubric satisfaction
- citation quality / evidence support
- answer format and tool-use behavior

---

## 5. `config.yaml` Judge Backend Notes

Training-time reward judges and `verify_answer` feedback judges are backed by:

```bash
open_instruct/common/config.yaml
```

In particular, the key backends are:

- `rubric_judge`
- `rubric_generation`
- `citation_judge`

Conceptually:

- `rubric_judge`
  - used by rubric reward computation
  - also reused by `verify_answer` / `verify_longform_answer`

- `citation_judge`
  - used by citation reward computation
  - also reused by `verify_answer` citation feedback

- `rubric_generation`
  - used when adaptive rubric generation is enabled

### Runtime override behavior

Although these backends are defined in `config.yaml`, this training script usually overrides the important fields via environment variables, for example:

```bash
export OPENAI_API_BASE=...
export RUBRIC_JUDGE_MODEL=...
export CITATION_JUDGE_MODEL=...
export RUBRIC_JUDGE_API_BASE=...
export CITATION_JUDGE_API_BASE=...
```

In practice, you can think of it as:

- `config.yaml` provides the default backend template
- the shell script provides the experiment-specific runtime override

### Sanitized example config

A secret-free example config is provided at:

```bash
open_instruct/common/config.example.yaml
```

Use that file as the shareable template when documenting or reproducing experiments.

---

## 6. Secret Handling

Do **not** hardcode real secrets in shared scripts or public documentation.

Prefer shell / scheduler / secret-manager injection, for example:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export OPENAI_API_KEY_POOL=<KEY1,KEY2,...>
export OPENAI_API_KEY_BYPASS_POOL=<BYPASS_KEY1,BYPASS_KEY2,...>
export SERPER_API_KEY=<YOUR_SERPER_API_KEY>
export S2_API_KEY=<YOUR_S2_API_KEY>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

The training script has been sanitized so that required secrets must now be provided externally.

---

## 7. Hardware / Cluster Requirements

The default script targets **2 nodes**, each with **8 GPUs**.

Important distributed settings in the LFF script include:

```bash
--num_learners_per_node 4 4
--vllm_num_engines 4
--vllm_tensor_parallel_size 2
```

This means:

- 2 nodes
- 4 learners on each node
- 4 vLLM engines total
- each vLLM engine uses tensor parallel size 2

If your hardware differs, you will likely need to adjust:

- `--num_learners_per_node`
- `--vllm_num_engines`
- `--vllm_tensor_parallel_size`
- `--vllm_gpu_memory_utilization`

---

## 8. Preparation Before Running

### 8.1 Code location

Training is launched from:

```bash
card/rl/open-instruct
```

### 8.2 Model path

The script currently points to a Qwen3-8B SFT checkpoint through the `model_name` variable. Replace that path if you want to use a different initialization model.

### 8.3 Dataset path

The script uses a local RL training dataset through the `dataset_list` variable. Replace it if needed.

### 8.4 Ray cluster

The script assumes a Ray cluster is already available.

For a manual setup, the pattern is typically:

#### Head node

```bash
ray start --head --port=8888 --dashboard-host=0.0.0.0
```

#### Worker node

```bash
ray start --address="<HEAD_IP>:8888" --block --dashboard-host=0.0.0.0
```

---

## 9. Important Environment Variables

The script depends on several groups of variables.

### 9.1 Import and path setup

The script sets values such as:

```bash
export DR_TULU_ROOT=...
export OPEN_INSTRUCT_ROOT=...
export PYTHONPATH=...
```

to ensure Ray actors and tool actors import the local code instead of globally installed packages.

### 9.2 Proxy settings

If your environment requires proxies, configure:

```bash
export http_proxy=...
export https_proxy=...
```

Otherwise, comment them out.

### 9.3 MCP tool server

The script expects an MCP service, configured through variables such as:

```bash
export MCP_HOST=...
export MCP_TRANSPORT=StreamableHttpTransport
export MCP_TRANSPORT_PORT=8003
```

Before training, verify that:

- the MCP service is running
- the required tools are reachable
- `NO_PROXY` is configured correctly


### 9.4 Judge backends

Judge-related variables include:

```bash
export RUBRIC_JUDGE_MODEL=...
export RUBRIC_GENERATION_MODEL=...
export CITATION_JUDGE_MODEL=...
export OPENAI_API_BASE=...
export RUBRIC_JUDGE_API_BASE=...
export CITATION_JUDGE_API_BASE=...
```

If you switch to locally deployed open-source judges (for example Qwen3-14B / 32B), update the corresponding model names and API base URLs consistently.



### 9.5 System prompt file location

The `system_prompt_file` / `NO_VA_SYSTEM_PROMPT_FILE` used by the training scripts are stored under:

```bash
rl/open-instruct/open_instruct/search_utils/system_prompts
```

For example, the scripts commonly reference:

```bash
${OPEN_INSTRUCT_ROOT}/open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml
${OPEN_INSTRUCT_ROOT}/open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907_no_draft_answer.yaml
```

If you need to modify the agent system prompt, enable/disable `verify_answer`, or switch tool-calling prompt templates, this is the directory you should edit first.

---

## 10. Running the LFF Training Script

From the training directory:

```bash
cd card/rl/open-instruct
bash train_card.sh
```

### Core command shape

The core command launched by the script is roughly:

```bash
python open_instruct/grpo_lff.py \
    --exp_name <experiment_name> \
    --wandb_project_name <wandb_project> \
    --beta 0.001 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --output_dir <output_dir> \
    --kl_estimator kl3 \
    --dataset_mixer_list <dataset_list> \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list <eval_dataset> 16 \
    --dataset_mixer_eval_list_splits train \
    --apply_adaptive_rubric_reward false \
    --normalize_rubric_scores false \
    --use_rubric_buffer true \
    --use_static_rubrics_as_persistent_rubrics true \
    --use_general_rubric false \
    --lff_enabled true \
    --lff_rs_weight 0.1 \
    --max_active_rubrics 5 \
    --llm_judge_timeout 300 \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18500 \
    --model_name_or_path <model_name> \
    --non_stop_penalty False \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --clip_higher 0.28 \
    --mask_truncated_completions true \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 50000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --lr_scheduler_type constant \
    --vllm_gpu_memory_utilization 0.7 \
    --apply_verifiable_reward true \
    --seed 42 \
    --num_evals 50 \
    --save_freq 50 \
    --gradient_checkpointing \
    --max_tool_calls 10 \
    --tools mcp \
    --checkpoint_state_freq 50 \
    --checkpoint_state_dir <checkpoint_dir> \
    --mcp_parser_name v20250824 \
    --system_prompt_file <system_prompt_file> \
    --mcp_port 8003 \
    --mcp_timeout 200 \
    --mcp_host <mcp_host> \
    --mcp_tool_names <tool_names> \
    --enable_draft_answer True \
    --save_traces true \
    --log_training_rollouts true \
    --hf_entity local \
    --log_training_rollouts_freq 1 \
    --num_training_rollouts_to_log 16 \
    --with_tracking \
    --lff_enable_va_reweight true \
    --lff_enable_rs true
```

---

## 11. Standard GRPO Baseline Script

In addition to the LFF script, this repository also contains a more standard GRPO-style training script:

```bash
rl/open-instruct/train_dr_tulu.sh
```

This script is closer to a plain **GRPO + rubric reward + tool use** setup.

### Key difference from the LFF script

- `train_dr_tulu_8b_lff.sh` runs:

```bash
python open_instruct/grpo_lff.py
```

- `train_dr_tulu.sh` runs:

```bash
python open_instruct/grpo_fast_rubric.py
```

In other words:

- `train_dr_tulu_8b_lff.sh` adds LFF / `verify_answer`-aware trajectory logic
- `train_dr_tulu.sh` is the simpler GRPO-style baseline

### When to use the standard GRPO script

Use `train_dr_tulu.sh` when you want to:

1. run a baseline without LFF
2. compare standard GRPO against LFF
3. validate the tool / reward / Ray / vLLM pipeline before enabling the more complex LFF logic

### How to run it

```bash
cd card/rl/open-instruct
bash train_grpo.sh
```

### Core command shape

The core command launched by the baseline script is roughly:

```bash
python open_instruct/grpo_fast_rubric.py \
    --exp_name <experiment_name> \
    --wandb_project_name <wandb_project> \
    --beta 0.001 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --output_dir <output_dir> \
    --kl_estimator kl3 \
    --dataset_mixer_list <dataset_list> \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list <eval_dataset> 16 \
    --dataset_mixer_eval_list_splits train \
    --apply_adaptive_rubric_reward true \
    --normalize_rubric_scores false \
    --use_rubric_buffer false \
    --use_static_rubrics_as_persistent_rubrics true \
    --use_general_rubric false \
    --max_active_rubrics 5 \
    --llm_judge_timeout 300 \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18500 \
    --model_name_or_path <model_name> \
    --non_stop_penalty False \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --clip_higher 0.28 \
    --mask_truncated_completions true \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 10000000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --lr_scheduler_type constant \
    --vllm_gpu_memory_utilization 0.7 \
    --apply_verifiable_reward true \
    --seed 42 \
    --num_evals 50 \
    --save_freq 50 \
    --gradient_checkpointing \
    --max_tool_calls 10 \
    --only_reward_good_outputs True \
    --tools mcp \
    --checkpoint_state_freq 50 \
    --checkpoint_state_dir <checkpoint_dir> \
    --mcp_parser_name v20250824 \
    --system_prompt_file <system_prompt_file> \
    --mcp_port 8003 \
    --mcp_timeout 200 \
    --mcp_host <mcp_host> \
    --mcp_tool_names <tool_names> \
    --enable_draft_answer True \
    --save_traces true \
    --log_training_rollouts true \
    --hf_entity local \
    --log_training_rollouts_freq 1 \
    --num_training_rollouts_to_log 1
```

---

## 12. Outputs

Training typically produces:

### 12.1 Run output directory

```bash
${OUTPUT_DIR}/<run_name>
```

This usually contains:

- saved model outputs
- traces
- rollout samples

### 12.2 Checkpoint state directory

```bash
${OUTPUT_DIR}/checkpoints/<exp_name>
```

used for resuming training.

### 12.3 Traces

If enabled, the scripts save training traces and rollout-level information useful for debugging and analysis.

---

## 13. Common Issues

### 13.1 Training exits immediately without steps

Typical reasons:

- the resumed checkpoint step already exceeds the requested total training steps
- the Ray cluster is not configured correctly
- learner / vLLM engine initialization failed

Check:

- `checkpoint_state_dir`
- `total_episodes`
- `num_learners_per_node`
- `vllm_num_engines`

### 13.2 MCP tool failures

Check:

- `MCP_HOST`
- `MCP_TRANSPORT_PORT`
- `NO_PROXY`
- MCP service logs

### 13.3 Judge timeout or slow reward computation

Check:

- `RUBRIC_JUDGE_TIMEOUT`
- `OPENAI_API_BASE`
- judge model throughput
- judge concurrency settings

### 13.4 Multi-`verify_answer` trajectory issues

If you are using the LFF script and encounter multi-VA errors, inspect:

- `traces_*.jsonl`
- `tool_error`
- `lff/multi_va_summary`

---

## 14. Recommended Cleanup / Sharing Practice

If you plan to share these scripts with collaborators or publish them:

1. keep secrets out of the scripts
2. inject secrets from your shell / scheduler
3. point readers to:

```bash
open_instruct/common/config.example.yaml
```

4. keep README examples secret-free

---

## 15. Quick Start

If you already have:

- a working Ray cluster
- valid model paths
- a running MCP tool service
- valid judge API access

then the shortest way to launch training is:

### LFF training

```bash
cd card/rl/open-instruct
bash train_cardf.sh
```

### Standard GRPO baseline

```bash
cd card/rl/open-instruct
bash train_grpo.sh
```

## 12.1 Evaluation code location

Agent evaluation code is located under:

```bash
card/agent/evaluation
```

If you want to:

- inspect evaluation task definitions
- run agent evaluation
- understand the benchmark / evaluation pipeline

this is the directory to start with.

> Note: the actual repository path is `agent/evaluation`, not `card/agent/evaluation`.

---

## 13. Common Issues

### 13.1 Training exits immediately without steps

Typical reasons:

- the resumed checkpoint step already exceeds the requested total training steps
- the Ray cluster is not configured correctly
- learner / vLLM engine initialization failed

Check:

- `checkpoint_state_dir`
- `total_episodes`
- `num_learners_per_node`
- `vllm_num_engines`

### 13.2 MCP tool failures

Check:

- `MCP_HOST`
- `MCP_TRANSPORT_PORT`
- `NO_PROXY`
- MCP service logs