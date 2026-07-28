#!/bin/bash
#SBATCH --job-name=eval-webshaper-base
#SBATCH --account=rulins
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/scrubbed/rulins/slurm_logs/dr-tulu/eval-webshaper-base-%j.out
#SBATCH --exclude=g001

# WebShaper evaluation for base 8B model with Google search + Jina browsing
# Dataset: rl-research/webshaper-heldout (246 examples)

set -e

# ============================================
# Configuration
# ============================================
MODEL_NAME="rl-research/DR-Tulu-SFT-8B"
MODEL_TAG="sft-8b-base-google"
DATASET="webshaper"
MAX_TOOL_CALLS=20
NUM_EXAMPLES="final_run"

EVAL_OUTPUT_DIR="/gpfs/scrubbed/rulins/dr-tulu/eval_output/webshaper-${MODEL_TAG}-t${MAX_TOOL_CALLS}"

# Fixed server ports (unique per script to avoid conflicts)
MODEL_PORT=30003
MCP_PORT=8003
MAX_CONCURRENT=5

echo "=============================================="
echo "Evaluating on $DATASET"
echo "=============================================="
echo "Model:        $MODEL_NAME"
echo "Tag:          $MODEL_TAG"
echo "Dataset:      $DATASET"
echo "Num examples: $NUM_EXAMPLES"
echo "Max tools:    $MAX_TOOL_CALLS"
echo "Output dir:   $EVAL_OUTPUT_DIR"
echo "=============================================="

# ============================================
# Setup environment
# ============================================
cd /gpfs/projects/kohlab/rulins/dr-tulu/agent

# Source environment variables (API keys)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Set MCP port
export MCP_TRANSPORT_PORT=${MCP_PORT}

# ============================================
# Kill existing servers and clean up screens
# ============================================
echo "Cleaning up existing servers..."
screen -wipe 2>/dev/null || true
pkill -f "vllm serve.*:${MODEL_PORT}" 2>/dev/null || true
pkill -f "mcp_backend.*:${MCP_PORT}" 2>/dev/null || true
screen -S vllm_main -X quit 2>/dev/null || true
screen -S mcp_server -X quit 2>/dev/null || true
sleep 2

# ============================================
# Launch VLLM server (main model on GPU 0)
# ============================================
echo "Starting main VLLM server on port $MODEL_PORT (GPU 0)..."
screen -dmS vllm_main bash -c "cd /gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct && CUDA_VISIBLE_DEVICES=0 uv run vllm serve $MODEL_NAME --dtype auto --port $MODEL_PORT --max-model-len 16384 --gpu-memory-utilization 0.8 2>&1 | tee /tmp/vllm_main.log"

# ============================================
# Launch MCP server with Google search (Serper API) + Jina browsing
# ============================================
echo "Starting MCP server with Google search on port $MCP_PORT..."
screen -dmS mcp_server bash -c "cd /gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct && uv run python -m dr_agent.mcp_backend.main --transport http --host 0.0.0.0 --port $MCP_PORT 2>&1 | tee /tmp/mcp_server.log"

# ============================================
# Wait for servers to be ready
# ============================================
echo "Waiting for main VLLM server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "Main VLLM server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Main VLLM server failed to start. Check /tmp/vllm_main.log"
        cat /tmp/vllm_main.log | tail -50
        exit 1
    fi
    sleep 5
done

echo "Waiting for MCP server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$MCP_PORT/health > /dev/null 2>&1; then
        echo "MCP server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: MCP server failed to start. Check /tmp/mcp_server.log"
        cat /tmp/mcp_server.log | tail -50
        exit 1
    fi
    sleep 5
done

# ============================================
# Run evaluation (generation)
# ============================================
echo "=============================================="
echo "Running $DATASET generation..."
echo "=============================================="

mkdir -p "$EVAL_OUTPUT_DIR"

cd /gpfs/projects/kohlab/rulins/dr-tulu/rl/open-instruct

uv run python /gpfs/projects/kohlab/rulins/dr-tulu/agent/workflows/auto_search_sft.py \
    generate-dataset $DATASET \
    --num-examples $NUM_EXAMPLES \
    --max-concurrent 10 \
    --batch-size 10 \
    --use-cache \
    --config /gpfs/projects/kohlab/rulins/dr-tulu/agent/workflows/auto_search_sft.yaml \
    --config-overrides "search_agent_model_name=$MODEL_NAME,search_agent_tokenizer_name=$MODEL_NAME,search_agent_base_url=http://localhost:$MODEL_PORT/v1,mcp_port=$MCP_PORT,use_browse_agent=false,search_agent_max_tool_calls=$MAX_TOOL_CALLS,browse_tool_name=jina,search_tool_name=serper,search_agent_max_tokens=16000,search_agent_temperature=0.0" \
    --output "$EVAL_OUTPUT_DIR/${DATASET}.jsonl"

# ============================================
# Run scoring
# ============================================
echo "=============================================="
echo "Running $DATASET evaluation..."
echo "=============================================="

uv run python /gpfs/projects/kohlab/rulins/dr-tulu/agent/scripts/evaluate.py $DATASET "$EVAL_OUTPUT_DIR/${DATASET}.jsonl" --grader-model gpt-4o

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $EVAL_OUTPUT_DIR"
echo "=============================================="

# Cleanup
screen -S vllm_main -X quit 2>/dev/null || true
screen -S mcp_server -X quit 2>/dev/null || true


