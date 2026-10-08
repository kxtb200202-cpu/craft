"""M2: constraint pre-classifier (deterministic, no model inference).

Pulls the constraint corpus of the RAGFlow knowledge base and selects the target
constraint-type set T(t) of a task description by weighted term-hit rates computed
over it."""

import math
import re
from typing import List, Dict, Any, Optional, Set

try:
    from ..config import (
        CORPUS_TOP_K,
        RAGFLOW_API_KEY,
        RAGFLOW_BASE_URL,
        ragflow_dataset_name,
    )
    from ..ragflow.client import RAGFlowClient
except ImportError:
    from config import (
        CORPUS_TOP_K,
        RAGFLOW_API_KEY,
        RAGFLOW_BASE_URL,
        ragflow_dataset_name,
    )
    from ragflow.client import RAGFlowClient


CONSTRAINT_TYPES = ["resource", "capacity", "deadline"]


class ConstraintPreClassifier:
    ALPHA_WEIGHT = 1.0

    def __init__(self, client: Optional[RAGFlowClient] = None,
                 corpus_size: int = CORPUS_TOP_K):
        if client is None:
            client = RAGFlowClient(
                base_url=RAGFLOW_BASE_URL,
                api_key=RAGFLOW_API_KEY,
                dataset_name=ragflow_dataset_name(),
            )
        self.client = client
        self.corpus_size = corpus_size

        self.constraints: List[Dict[str, Any]] = []
        self._total_count = 0
        self._type_token_sets: Dict[str, Set[str]] = {}
        self._idf: Dict[str, float] = {}
        self._corpus_loaded = False

    def _ensure_corpus(self, probe_query: str = "constraint") -> None:
        if self._corpus_loaded:
            return
        self._corpus_loaded = True

        try:
            self.constraints = self.client.retrieve_as_constraints(
                probe_query, top_k=self.corpus_size,
                similarity_threshold=0.0, verbose=False)
        except Exception as exc:
            print(f"[M2] the constraint corpus could not be read from RAGFlow: {exc}")
            self.constraints = []

        self._total_count = len(self.constraints)
        self._build_index()
        print(f"[M2] constraint pre-classifier loaded {self._total_count} constraints "
              f"from the RAGFlow knowledge base '{self.client.dataset_name}'")

    def _build_index(self):
        docs = []
        type_tokens: Dict[str, Set[str]] = {t: set() for t in CONSTRAINT_TYPES}

        for c in self.constraints:
            ctype = c.get("type", "")
            text = f"{c.get('desc', '')} {c.get('slug', '')}"
            tokens = set(self._tokenize(text).keys())
            docs.append(tokens)
            if ctype in type_tokens:
                type_tokens[ctype].update(tokens)

        self._type_token_sets = type_tokens

        n = len(docs)
        df: Dict[str, int] = {}
        for tokens in docs:
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
        self._idf = {
            tok: math.log((n + 1) / (freq + 1)) + 1
            for tok, freq in df.items()
        }

    def _tokenize(self, text: str) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for token in re.findall(r'[a-zA-Z0-9_]+', text):
            tok = token.lower()
            weights[tok] = weights.get(tok, 0) + self.ALPHA_WEIGHT
        return weights

    def _weighted_tokens(self, text: str) -> Dict[str, float]:
        return {
            tok: base * self._idf.get(tok, 1.0)
            for tok, base in self._tokenize(text).items()
        }

    def select_types(self, task_description: str) -> List[str]:
        self._ensure_corpus(task_description)

        q = self._weighted_tokens(task_description)
        total_w = sum(q.values())
        if total_w <= 0:
            return list(CONSTRAINT_TYPES)

        sims: Dict[str, float] = {}
        for t in CONSTRAINT_TYPES:
            d_tokens = self._type_token_sets.get(t, set())
            hit_w = sum(w for tok, w in q.items() if tok in d_tokens)
            sims[t] = hit_w / total_w

        selected = [t for t in CONSTRAINT_TYPES if sims[t] > 0]

        if not selected:
            selected = list(CONSTRAINT_TYPES)

        detail = ", ".join(f"{t}={sims[t]:.3f}" for t in CONSTRAINT_TYPES)
        print(f"[M2] type similarity: {detail} → T(t)={selected}")
        return selected

    def classify(self, task_description: str) -> List[Dict[str, Any]]:
        types = set(self.select_types(task_description))
        candidate = [c for c in self.constraints if c.get("type", "") in types]
        kept_pct = len(candidate) / max(self._total_count, 1) * 100
        print(f"[M2] pre-classification done: {self._total_count} → {len(candidate)} "
              f"({kept_pct:.0f}%), type set={sorted(types)}")
        return candidate

    @property
    def total_count(self) -> int:
        self._ensure_corpus()
        return self._total_count
