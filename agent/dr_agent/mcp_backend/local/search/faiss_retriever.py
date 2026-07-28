import glob
import json
import logging
import os
import pickle
from itertools import chain
from typing import Optional, List

import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .base import BaseSearcher, LocalSearchResponse, LocalSearchResult

logger = logging.getLogger(__name__)


class FaissSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        super().parse_args(parser)
        parser.add_argument(
            "--index-path",
            required=True,
            help=(
                "Path to FAISS index. Two formats are supported:\n"
                "  1. Raw FAISS file   : /path/to/e5_Flat.index\n"
                "  2. Tevatron pkl glob: /path/to/corpus.*.pkl"
            ),
        )
        parser.add_argument(
            "--model-name",
            required=True,
            help="Encoder model for query embedding (e.g. 'intfloat/e5-large-v2').",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            default=False,
            help="L2-normalize query embeddings before search (needed for cosine / inner-product indexes).",
        )
        parser.add_argument(
            "--pooling",
            default="mean",
            choices=["mean", "cls", "eos"],
            help="Pooling strategy: mean (E5/BGE), cls (BERT), eos (Qwen-Embedding). Default: mean.",
        )
        parser.add_argument(
            "--torch-dtype",
            default="float16",
            choices=["float16", "bfloat16", "float32"],
            help="Torch dtype for encoder model. Default: float16.",
        )
        parser.add_argument(
            "--query-prefix",
            default="query: ",
            help=(
                "Prefix prepended to every query before encoding.\n"
                "  E5 models         : 'query: '\n"
                "  Qwen-Embedding    : long instruction string (see --task-prefix)\n"
                "  BGE models        : '' (empty)\n"
                "Default: 'query: '"
            ),
        )
        parser.add_argument(
            "--task-prefix",
            default="",
            help=(
                "If set, overrides --query-prefix with this full instruction string.\n"
                "Useful for Qwen-Embedding which expects a long Instruct: ... \\nQuery: prefix."
            ),
        )
        parser.add_argument(
            "--max-length",
            type=int,
            default=512,
            help="Maximum token length for query encoding. Default: 512.",
        )
        parser.add_argument(
            "--encoder-device",
            default=None,
            help=(
                "Device for the encoder model: 'cuda', 'cuda:1', 'cpu', etc. "
                "Defaults to 'cuda' if a GPU is available, otherwise 'cpu'. "
                "Set to 'cpu' to avoid OOM when the GPU is occupied by another process."
            ),
        )

    def __init__(self, args):
        super().__init__(args)
        if args.model_name == "bm25":
            raise ValueError("model_name cannot be 'bm25' for FAISS searcher")

        self.args = args
        self.faiss_index = None      # raw faiss.Index (for .index files)
        self.retriever   = None      # FaissFlatSearcher (for Tevatron .pkl files)
        self.model       = None
        self.tokenizer   = None
        self.lookup      = None      # list[docid]: position i → document id
        self.docid_to_metadata = None
        self.url_to_text       = None

        # Query prefix: --task-prefix takes precedence over --query-prefix
        self._query_prefix = (
            args.task_prefix if getattr(args, "task_prefix", "") else
            getattr(args, "query_prefix", "query: ")
        )

        logger.info("Initializing FAISS searcher...")
        self._load_faiss_index()
        self._load_encoder()
        self._load_dataset()
        logger.info("FAISS searcher initialized successfully.")

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_faiss_index(self) -> None:
        index_path = self.args.index_path

        if index_path.endswith(".index") and os.path.isfile(index_path):
            # ── Format 1: raw FAISS .index file ──────────────────────────
            logger.info("Loading raw FAISS index: %s", index_path)
            self.faiss_index = faiss.read_index(index_path)
            # Sequential lookup: position i → str(i).  Will be replaced by
            # dataset-derived IDs after _load_dataset() completes.
            self.lookup = list(range(self.faiss_index.ntotal))
            logger.info(
                "Loaded FAISS index: ntotal=%d, d=%d",
                self.faiss_index.ntotal, self.faiss_index.d,
            )
            self._setup_gpu_for_index()

        else:
            # ── Format 2: Tevatron pickle glob ────────────────────────────
            try:
                from tevatron.retriever.searcher import FaissFlatSearcher
            except ImportError as e:
                raise ImportError(
                    "Tevatron is required for .pkl index files. "
                    "Install it with: pip install tevatron"
                ) from e

            def pickle_load(path):
                with open(path, "rb") as f:
                    reps, lk = pickle.load(f)
                return np.array(reps), lk

            index_files = glob.glob(index_path)
            if not index_files:
                raise ValueError(f"No files found matching pattern: {index_path}")
            logger.info("Loading %d shard(s) from Tevatron pkl index.", len(index_files))

            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            self.retriever = FaissFlatSearcher(p_reps_0)
            self.lookup    = list(p_lookup_0)

            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc="Loading shards", total=len(index_files))
            for p_reps, p_lookup in shards:
                self.retriever.add(p_reps)
                self.lookup += list(p_lookup)

            self._setup_gpu_for_tevatron()

    def _setup_gpu_for_index(self) -> None:
        """Move raw FAISS index to GPU if available."""
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPU found — using CPU for FAISS.")
            return
        logger.info("Moving FAISS index to %d GPU(s).", num_gpus)
        if num_gpus == 1:
            co  = faiss.GpuClonerOptions(); co.useFloat16 = True
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index, co)
        else:
            co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
            self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index, co, ngpu=num_gpus)

    def _setup_gpu_for_tevatron(self) -> None:
        """Move Tevatron FaissFlatSearcher to GPU if available."""
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPU found — using CPU for FAISS.")
            return
        logger.info("Moving Tevatron FAISS index to %d GPU(s).", num_gpus)
        if num_gpus == 1:
            co  = faiss.GpuClonerOptions(); co.useFloat16 = True
            res = faiss.StandardGpuResources()
            self.retriever.index = faiss.index_cpu_to_gpu(res, 0, self.retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
            self.retriever.index = faiss.index_cpu_to_all_gpus(
                self.retriever.index, co, ngpu=num_gpus
            )

    # ------------------------------------------------------------------
    # Encoder loading
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        logger.info("Loading encoder model: %s", self.args.model_name)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map[self.args.torch_dtype]
        encoder_device = getattr(self.args, "encoder_device", None)
        if encoder_device:
            device = encoder_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        hf_home = os.getenv("HF_HOME")
        cache_dir = hf_home or None

        # Try Tevatron DenseModel first; fall back to plain AutoModel
        try:
            from tevatron.retriever.arguments import ModelArguments
            from tevatron.retriever.driver.encode import DenseModel
            model_args = ModelArguments(
                model_name_or_path=self.args.model_name,
                normalize=self.args.normalize,
                pooling=self.args.pooling,
                cache_dir=cache_dir,
            )
            self.model = DenseModel.load(
                model_args.model_name_or_path,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                cache_dir=model_args.cache_dir,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            self._use_tevatron_encoder = True
            logger.info("Using Tevatron DenseModel encoder.")
        except Exception:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.args.model_name,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            ).to(device).eval()
            self._use_tevatron_encoder = False
            logger.info("Using plain AutoModel encoder (pooling=%s).", self.args.pooling)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            cache_dir=cache_dir,
            padding_side="right",
        )
        logger.info("Encoder loaded on %s.", device)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """
        Load passage text from the dataset specified by --dataset-name.

        Supports:
          - Local JSONL file  : /path/to/wiki-18.jsonl
          - HuggingFace dataset: org/dataset-name

        Expected fields (flexible, first match wins):
          id       : "id"  | "_id"  | "docid"
          text     : "contents" | "text" | "passage"
          url      : "url" | "link"          (optional)
          title    : "title"                  (optional, prepended to text)
        """
        dataset_name = self.args.dataset_name
        logger.info("Loading dataset: %s", dataset_name)
        cache_dir = os.getenv("HF_DATASETS_CACHE") or None

        try:
            if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
                ds = load_dataset("json", data_files=dataset_name, split="train", cache_dir=cache_dir)
            else:
                ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}") from e

        self.docid_to_metadata = {}
        self.url_to_text       = {}

        for i, row in enumerate(ds):
            # ── document id ──────────────────────────────────────────────
            docid = str(
                row.get("id") or row.get("_id") or row.get("docid") or i
            )

            # ── text ─────────────────────────────────────────────────────
            title   = str(row.get("title") or "").strip()
            body    = str(
                row.get("contents") or row.get("text") or
                row.get("passage") or ""
            ).strip()
            snippet = f"{title}: {body}" if title and body else (title or body)

            # ── url ───────────────────────────────────────────────────────
            # source_url: the original article URL (for display / citation).
            # browse_key: unique per passage — used as the url returned in
            #   search results so local_browse can reliably retrieve the exact
            #   passage.  We use "doc:{docid}" because many passages share the
            #   same article URL (DPR-style split), and a dict keyed by URL
            #   would silently overwrite earlier passages from the same article.
            source_url = str(row.get("url") or row.get("link") or "").strip()
            browse_key = f"doc:{docid}"

            self.docid_to_metadata[docid] = {
                "snippet":    snippet,
                "url":        source_url or browse_key,  # display URL
                "browse_key": browse_key,
            }
            # Index by browse_key (always unique) so local_browse can find it.
            self.url_to_text[browse_key] = snippet
            # Also index by source_url for direct article URL lookups,
            # but do NOT overwrite an existing entry (first passage wins).
            if source_url and source_url not in self.url_to_text:
                self.url_to_text[source_url] = snippet

        n = len(self.docid_to_metadata)
        logger.info("Loaded %d passages from dataset.", n)

        # ── Align FAISS lookup with JSONL order ───────────────────────────
        # For raw .index files the initial lookup is [0, 1, ..., n-1].
        # Replace with dataset docids in row order so search results carry
        # meaningful IDs instead of bare integers.
        if self.faiss_index is not None:
            # Re-build lookup from dataset row order
            self.lookup = [
                str(row.get("id") or row.get("_id") or row.get("docid") or i)
                for i, row in enumerate(ds)
            ]
            expected = self.faiss_index.ntotal
            if len(self.lookup) != expected:
                logger.warning(
                    "Dataset has %d rows but FAISS index has %d vectors — "
                    "results may be misaligned.",
                    len(self.lookup), expected,
                )

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a numpy vector."""
        text = self._query_prefix + query
        device = next(self.model.parameters()).device

        batch = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            if "cuda" in str(device):
                with torch.amp.autocast(str(device)):
                    if self._use_tevatron_encoder:
                        q_reps = self.model.encode_query(batch)
                    else:
                        output = self.model(**batch)
                        q_reps = self._pool(output, batch["attention_mask"])
            else:
                if self._use_tevatron_encoder:
                    q_reps = self.model.encode_query(batch)
                else:
                    output = self.model(**batch)
                    q_reps = self._pool(output, batch["attention_mask"])

        vec = q_reps.cpu().float().numpy()
        if self.args.normalize:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            vec   = vec / np.maximum(norms, 1e-8)
        return vec

    def _pool(self, model_output, attention_mask) -> torch.Tensor:
        """Apply the configured pooling strategy to model output."""
        hidden = model_output.last_hidden_state   # (B, L, D)
        if self.args.pooling == "cls":
            return hidden[:, 0, :]
        if self.args.pooling == "eos":
            # Last non-padding token
            seq_len = attention_mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0)), seq_len, :]
        # mean pooling (default for E5 / BGE)
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> LocalSearchResponse:
        q_reps = self._encode_query(query)

        if self.faiss_index is not None:
            # ── raw .index path ──────────────────────────────────────────
            all_scores, psg_indices = self.faiss_index.search(q_reps, k)
        else:
            # ── Tevatron FaissFlatSearcher path ──────────────────────────
            all_scores, psg_indices = self.retriever.search(q_reps, k)

        results: List[LocalSearchResult] = []
        for score, idx in zip(all_scores[0], psg_indices[0]):
            if idx < 0 or idx >= len(self.lookup):
                continue
            docid    = str(self.lookup[idx])
            metadata = self.docid_to_metadata.get(docid)
            if metadata is None:
                continue
            results.append({
                "docid":   docid,
                "score":   float(score),
                "snippet": metadata["snippet"],
                # Return browse_key ("doc:{docid}") as the URL so that
                # local_browse can reliably retrieve this exact passage.
                # The original article URL is available in metadata["url"].
                "url":     metadata.get("browse_key", f"doc:{docid}"),
            })

        return {"results": results}

    def get_text_by_url(self, url: str) -> Optional[str]:
        return (self.url_to_text or {}).get(url)

    @property
    def search_type(self) -> str:
        return "FAISS"
