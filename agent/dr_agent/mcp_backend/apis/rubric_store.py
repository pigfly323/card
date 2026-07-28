import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    if not query:
        return ""
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _question_lookup_variants(question: str) -> List[str]:
    """
    Try a few string forms so snapshot keys from JSONL still match after minor
    punctuation / whitespace differences (common in HealthBench apostrophes).
    """
    raw = (question or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for v in (
        raw,
        raw.replace("\u2019", "'").replace("\u2018", "'"),  # curly single quotes
        raw.replace("\u201c", '"').replace("\u201d", '"'),  # curly double quotes
        re.sub(r"\s+", " ", raw),  # collapse internal whitespace (not lowercasing)
    ):
        if v and v not in out:
            out.append(v)
    return out


def normalize_rubric_item_for_judge(item: Any) -> Dict[str, Any]:
    """
    Map benchmark-native rubric shapes (HealthBench `criterion`, ResearchQA `rubric_item`)
    into `title` + `description` expected by verify_answer / draft_answer judges.
    """
    if not isinstance(item, dict):
        return {"title": "Criterion", "description": str(item).strip()}

    out = dict(item)
    desc = str(out.get("description", "")).strip()
    if not desc:
        if out.get("criterion") is not None:
            desc = str(out["criterion"]).strip()
        elif out.get("rubric_item") is not None:
            desc = str(out["rubric_item"]).strip()
    title = str(out.get("title", "")).strip()
    if not title:
        if desc:
            title = desc[:77] + "..." if len(desc) > 80 else desc
        else:
            title = "Criterion"
    out["title"] = title
    out["description"] = desc
    return out


def normalize_rubric_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize static/positive/negative rubric lists in place (returns a shallow copy)."""
    out = dict(result)
    for key in ("static_rubrics", "positive_rubrics", "negative_rubrics"):
        raw = out.get(key)
        if not isinstance(raw, list):
            continue
        out[key] = [normalize_rubric_item_for_judge(x) for x in raw]
    return out


class RubricStore:
    """
    Lightweight file-backed store for rubrics and ground-truth answers.

    Two independent indices are maintained:

    Rubric index  (RUBRIC_DATASET_PATH):
      query -> {"static_rubrics": [...], "positive_rubrics": [...], "negative_rubrics": [...]}
      Source: training JSONL where ground_truth JSON contains a "rubrics" key.

    Ground-truth index  (GT_DATASET_PATH):
      query -> list[str]  (preserves all acceptable answer candidates)
      Supports two file formats detected automatically:
        • JSON array  – e.g. HotpotQA: [{"Question": "...", "answer": ["yes"]}]
        • JSONL       – one object per line: {"question": "...", "answers": [...]}
      Also supports training JSONL where ground_truth JSON contains an "answer" key.
      Multiple files may be listed (space/newline/semicolon separated, same syntax as
      RUBRIC_DATASET_PATH); an optional weight token after the path is ignored.

    Both indices use _question_lookup_variants for robust question matching.

    Environment variables:
      RUBRIC_DATASET_PATH  – rubric source (path [weight] entries, JSONL format)
      GT_DATASET_PATH      – ground truth source (path [weight] entries, JSON array or JSONL)
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get(
            "RUBRIC_BUFFER_EXPORT_PATH",
            "/tmp/rubric_buffer_snapshot.json",
        )
        self._cache: Dict[str, Any] = {}
        self._last_mtime: Optional[float] = None
        self._last_load_ts: float = 0.0

        # Rubric index: query -> {"static_rubrics": [...]}
        self._dataset_index: Optional[Dict[str, Dict]] = None
        self._dataset_normalized_index: Optional[Dict[str, str]] = None
        if os.environ.get("RUBRIC_DATASET_PATH", "").strip():
            t0 = time.time()
            self._load_dataset_index()
            logger.info(
                "RubricStore: rubric index pre-loaded in %.3fs, %d queries",
                time.time() - t0,
                len(self._dataset_index or {}),
            )

        # Ground-truth index: query -> list[str] (preserves all answer candidates)
        self._gt_index: Optional[Dict[str, List[str]]] = None
        self._gt_normalized_index: Optional[Dict[str, str]] = None
        if os.environ.get("GT_DATASET_PATH", "").strip():
            t0 = time.time()
            self._load_gt_index()
            logger.info(
                "RubricStore: GT index pre-loaded in %.3fs, %d queries",
                time.time() - t0,
                len(self._gt_index or {}),
            )


    def _load_dataset_index(self) -> None:
        """
        Load rubrics from the training dataset JSONL into memory.

        RUBRIC_DATASET_PATH supports the same "path weight" space-separated format
        used by the RL training scripts, e.g.:
          /path/to/train.jsonl 1.0
        Multiple entries can be separated by newlines or semicolons.
        Only the path part is used; weight is ignored.
        """
        env_val = os.environ.get("RUBRIC_DATASET_PATH", "").strip()
        if not env_val:
            self._dataset_index = {}
            self._dataset_normalized_index = {}
            logger.debug("RubricStore: RUBRIC_DATASET_PATH not set, dataset fallback disabled")
            return

        self._dataset_index = {}
        self._dataset_normalized_index = {}
        total = 0

        # Parse one or more "path [weight]" entries
        for entry in re.split(r"[\n;]+", env_val):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split()
            jsonl_path = parts[0]
            if not os.path.exists(jsonl_path):
                logger.warning("RubricStore: dataset file not found: %s", jsonl_path)
                continue
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        gt_raw = obj.get("ground_truth")
                        if not gt_raw:
                            continue
                        # ground_truth may be a JSON string or already a dict/list
                        if isinstance(gt_raw, str):
                            try:
                                gt = json.loads(gt_raw)
                            except json.JSONDecodeError:
                                continue
                        elif isinstance(gt_raw, (dict, list)):
                            gt = gt_raw
                        else:
                            continue
                        # Support both single dict and list of dicts
                        entries = gt if isinstance(gt, list) else [gt]
                        for item in entries:
                            if not isinstance(item, dict):
                                continue
                            query = (item.get("query") or "").strip()
                            rubrics = item.get("rubrics", [])
                            if not query or not isinstance(rubrics, list):
                                continue
                            static = [r for r in rubrics if isinstance(r, dict)]
                            if not static:
                                continue
                            self._dataset_index[query] = {
                                "static_rubrics": static,
                                "positive_rubrics": [],
                                "negative_rubrics": [],
                            }
                            norm = _normalize_query(query)
                            self._dataset_normalized_index[norm] = query
                            total += 1
                logger.info(
                    "RubricStore: loaded %d rubric entries from dataset %s",
                    total, jsonl_path,
                )
            except Exception:
                logger.exception("RubricStore: failed loading dataset %s", jsonl_path)

        logger.info(
            "RubricStore: dataset index ready, total=%d queries", len(self._dataset_index)
        )

    def _load_gt_index(self) -> None:
        """
        Load ground-truth answers from GT_DATASET_PATH into memory.

        Supports two file formats (auto-detected by first character):
          • JSON array  – [{"Question": "...", "answer": ["yes"]}, ...]
          • JSONL       – {"question": "...", "answer": "..."} per line

        Also handles training JSONL where ground_truth is a JSON string
        containing a "query" + "answer" (or "answers") key.

        Field name priority:
          question field : "Question" > "question" > "problem" > gt.query
          answer field   : "answer"   > "answers"  > "Answer"  > gt.answer
        """
        env_val = os.environ.get("GT_DATASET_PATH", "").strip()
        if not env_val:
            self._gt_index = {}
            self._gt_normalized_index = {}
            return

        self._gt_index = {}
        self._gt_normalized_index = {}
        total = 0

        for entry in re.split(r"[\n;]+", env_val):
            entry = entry.strip()
            if not entry:
                continue
            file_path = entry.split()[0]
            if not os.path.exists(file_path):
                logger.warning("RubricStore GT: file not found: %s", file_path)
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == "[":
                        # JSON array format (e.g. HotpotQA)
                        items = json.load(f)
                        if not isinstance(items, list):
                            items = []
                    else:
                        # JSONL format — one object per line
                        items = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                items.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    # --- Extract question ---
                    q = str(
                        item.get("Question") or item.get("question")
                        or item.get("problem") or ""
                    ).strip()

                    # --- Extract answer ---
                    raw_ans = (
                        item.get("answer") or item.get("answers") or item.get("Answer")
                    )

                    # Fallback: training JSONL ground_truth JSON string
                    if not q or raw_ans is None:
                        gt_raw = item.get("ground_truth")
                        if gt_raw:
                            if isinstance(gt_raw, str):
                                try:
                                    gt_raw = json.loads(gt_raw)
                                except json.JSONDecodeError:
                                    gt_raw = None
                            for sub in (gt_raw if isinstance(gt_raw, list) else [gt_raw]):
                                if not isinstance(sub, dict):
                                    continue
                                q = q or str(sub.get("query") or "").strip()
                                raw_ans = raw_ans or (
                                    sub.get("answer") or sub.get("answers")
                                    or sub.get("answer_string")
                                )
                                if q and raw_ans is not None:
                                    break

                    if isinstance(raw_ans, list):
                        # 保留完整答案列表（多答案候选），不截断为第一个。
                        # 去重保序、去空，归一化为 list[str]。
                        seen = set()
                        ans_list = []
                        for a in raw_ans:
                            s = str(a).strip() if a is not None else ""
                            if s and s not in seen:
                                seen.add(s)
                                ans_list.append(s)
                        if not ans_list:
                            continue
                        self._gt_index[q] = ans_list
                        self._gt_normalized_index[_normalize_query(q)] = q
                        total += 1
                        continue

                    ans = str(raw_ans).strip() if raw_ans is not None else ""

                    if not q or not ans:
                        continue

                    self._gt_index[q] = [ans]
                    self._gt_normalized_index[_normalize_query(q)] = q
                    total += 1

                logger.info("RubricStore GT: loaded %d entries from %s", total, file_path)
                logger.info("First 3 GT entries: %s", list(self._gt_index.items())[:3])
                logger.info("Ground-truth for question 'who got the first nobel prize in physics?': %s", self._gt_index.get("who got the first nobel prize in physics?"))
            except Exception:
                logger.exception("RubricStore GT: failed loading %s", file_path)

        logger.info(
            "RubricStore GT: index ready, total=%d queries", len(self._gt_index)
        )
        logger.info("First 3 GT entries: %s", list(self._gt_index.items())[:3])
        logger.info("Ground-truth for question 'who got the first nobel prize in physics?': %s", self._gt_index.get("who got the first nobel prize in physics?"))


    def get_ground_truth(self, question: Optional[str]) -> Optional[List[str]]:
        """
        Look up the ground-truth answer(s) for a question.

        Returns a list of acceptable answer strings (preserves all candidates
        for multi-answer questions), or None if:
          - question is empty
          - GT_DATASET_PATH was not configured
          - no match found in the GT index
        """
        if not question:
            return None

        if self._gt_index is None:
            self._load_gt_index()

        if not self._gt_index:
            return None

        for q_try in _question_lookup_variants(question):
            if q_try in self._gt_index:
                logger.debug("RubricStore GT: exact match for %r", q_try[:60])
                return list(self._gt_index[q_try])

        for q_try in _question_lookup_variants(question):
            norm = _normalize_query(q_try)
            orig = self._gt_normalized_index.get(norm)
            if orig and orig in self._gt_index:
                logger.debug("RubricStore GT: normalized match for %r", q_try[:60])
                return list(self._gt_index[orig])

        logger.debug(
            "RubricStore GT: no match for %r", _normalize_query(question)[:80]
        )
        return None

    def _lookup_dataset(self, question: str) -> Optional[Dict[str, Any]]:
        """Look up rubrics from the dataset index. Returns None if not found."""
        if self._dataset_index is None:
            print(
                f"[RubricStore._lookup_dataset] dataset_index is None, triggering load. "
                f"RUBRIC_DATASET_PATH={os.environ.get('RUBRIC_DATASET_PATH', '(not set)')!r}",
                flush=True,
            )
            self._load_dataset_index()

        if not self._dataset_index:
            print(
                f"[RubricStore._lookup_dataset] dataset_index is empty after load. "
                f"RUBRIC_DATASET_PATH={os.environ.get('RUBRIC_DATASET_PATH', '(not set)')!r}",
                flush=True,
            )
            return None

        variants = _question_lookup_variants(question)
        print(
            f"[RubricStore._lookup_dataset] index_size={len(self._dataset_index)} "
            f"question_chars={len(question)} "
            f"variants={len(variants)} "
            f"question_prefix={question[:80]!r}",
            flush=True,
        )

        for q_try in variants:
            if q_try in self._dataset_index:
                entry = self._dataset_index[q_try]
                print(
                    f"[RubricStore._lookup_dataset] EXACT MATCH: static={len(entry.get('static_rubrics', []))} rubrics",
                    flush=True,
                )
                logger.info(
                    "RubricStore match: source=dataset_exact static=%d (lookup_chars=%d)",
                    len(entry.get("static_rubrics", [])),
                    len(q_try),
                )
                return {**entry, "source": "dataset_exact"}

        for q_try in variants:
            norm = _normalize_query(q_try)
            orig = self._dataset_normalized_index.get(norm)
            if orig and orig in self._dataset_index:
                entry = self._dataset_index[orig]
                print(
                    f"[RubricStore._lookup_dataset] NORMALIZED MATCH: static={len(entry.get('static_rubrics', []))} rubrics",
                    flush=True,
                )
                logger.info(
                    "RubricStore match: source=dataset_normalized static=%d",
                    len(entry.get("static_rubrics", [])),
                )
                return {**entry, "source": "dataset_normalized"}

        # No match — print a sample of index keys for diagnosis
        sample_keys = list(self._dataset_index.keys())[:3]
        print(
            f"[RubricStore._lookup_dataset] NO MATCH. "
            f"norm_question={_normalize_query(question)[:80]!r} "
            f"sample_index_keys={[k[:60] for k in sample_keys]}",
            flush=True,
        )
        return None

    def _load_if_needed(self) -> None:
        if not os.path.exists(self.path):
            self._cache = {}
            self._last_mtime = None
            logger.debug("RubricStore snapshot missing: path=%s", self.path)
            return
        try:
            mtime = os.path.getmtime(self.path)
            if self._last_mtime is not None and mtime == self._last_mtime:
                return
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            self._last_mtime = mtime
            self._last_load_ts = time.time()
            logger.info(
                "RubricStore loaded snapshot: path=%s mtime=%.3f load_ts=%.3f",
                self.path,
                mtime,
                self._last_load_ts,
            )
        except Exception:
            self._cache = {}
            logger.exception("RubricStore failed loading snapshot: path=%s", self.path)

    def get_rubrics(self, question: Optional[str]) -> Dict[str, Any]:
        """Look up rubrics for a question exclusively from the training dataset.

        The rubric_buffer_snapshot logic has been removed. Rubrics are loaded
        directly from RUBRIC_DATASET_PATH (training data JSONL) to ensure
        consistency with the RL training reward signal.
        """
        if not question:
            logger.debug("RubricStore match: source=default_static_without_question, returning empty rubrics")
            return {
                "static_rubrics": [],
                "positive_rubrics": [],
                "negative_rubrics": [],
                "source": "default_static_without_question",
            }

        dataset_result = self._lookup_dataset(question)
        if dataset_result is not None:
            return dataset_result

        # No match in dataset — return empty rubrics.
        logger.info(
            "RubricStore miss -> default_static_fallback: "
            "question_chars=%d norm_prefix=%r",
            len((question or "").strip()),
            (_normalize_query(question or "")[:160] + ("…" if len(_normalize_query(question or "")) > 160 else "")),
        )
        logger.debug("RubricStore match: source=default_static_fallback, returning empty rubrics")
        return {
            "static_rubrics": [],
            "positive_rubrics": [],
            "negative_rubrics": [],
            "source": "default_static_fallback",
        }
