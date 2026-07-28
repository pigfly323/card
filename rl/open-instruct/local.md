# Local Training

## Setup

We assume that you have cloned the repository, and are in `dr-tulu/rl/open-instruct`.

First, install new dependencies with `uv`. Note that this environment points to your local installation of `dr-agent` (modified from pip) and some new dependencies like Tevatron and Pyserini. You should be able to 1-click sync with:
```bash
uv sync
```

Then, download the necessary data and indices:
```bash
uv run huggingface-cli download s42chen/wii-indexes --local-dir ./data --repo-type dataset
```
This contains the corpus and pre-built BM25, Qwen3-Embed-8B indices.
> Make sure that you are logged in with `huggingface-cli login`.

## Local Training

Training is similar as before, except we use a new dataset, and changed the tool registry in `rl/open-instruct/open_instruct/search_utils/mcp_tools.py` to use new local tools added to `dr-agent` in `agent/dr_agent/mcp_backend/local/search`.

To run the mini test training with BM25, run:
```bash
export OPENAI_API_KEY=xxx
export WANDB_API_KEY=xxx
bash ./train_local/train_dr_tulu_mini_base_local_bm25.sh
```

To run the mini test training with Qwen3-Embed-8B, run:
```bash
export OPENAI_API_KEY=xxx
export WANDB_API_KEY=xxx
bash ./train_local/train_dr_tulu_mini_base_local_qwen3-8.sh
```