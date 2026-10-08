"""M3: hybrid retrieval engine over the constraint knowledge base.

Queries the RAGFlow knowledge base and fuses the semantic-vector score with the
keyword score it returns, Score = alpha*s_sem + beta*s_kw, restricted to the
type-filtered candidate set, and returns the Top-K rules."""

import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

try:
    from ..ragflow.client import RAGFlowClient
    from ..config import (
        DEFAULT_TOP_K,
        HYBRID_WEIGHTS,
        RAGFLOW_BASE_URL,
        RAGFLOW_API_KEY,
        ragflow_dataset_name,
    )
except ImportError:
    from ragflow.client import RAGFlowClient
    from config import (
        DEFAULT_TOP_K,
        HYBRID_WEIGHTS,
        RAGFLOW_BASE_URL,
        RAGFLOW_API_KEY,
        ragflow_dataset_name,
    )


class RAGFlowRetriever:
    def __init__(self, client: Optional[RAGFlowClient] = None):
        if client:
            self.client = client
        else:
            self.client = RAGFlowClient(
                base_url=RAGFLOW_BASE_URL,
                api_key=RAGFLOW_API_KEY,
                dataset_name=ragflow_dataset_name(),
            )
        self._available = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self.client.is_healthy()
        return self._available

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K
                 ) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[int, float]]:
        if not self.is_available:
            raise RuntimeError("RAGFlow is unavailable")

        constraints = self.client.retrieve_as_constraints(
            query, top_k=top_k, verbose=False)

        vector_scores = {}
        keyword_scores = {}

        for i, c in enumerate(constraints):
            vector_scores[i] = c.get("_vector_similarity", 0.0)
            keyword_scores[i] = c.get("_term_similarity", 0.0)

        return constraints, vector_scores, keyword_scores


class HybridRetriever:
    def __init__(self,
                 ragflow_retriever: Optional[RAGFlowRetriever] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.ragflow_retriever = ragflow_retriever or RAGFlowRetriever()
        self.weights = weights or HYBRID_WEIGHTS

        self._use_ragflow: Optional[bool] = None
        self.stats = {
            "ragflow_mode": 0,
            "total_retrievals": 0,
        }

    @property
    def use_ragflow(self) -> bool:
        if self._use_ragflow is None:
            self._use_ragflow = self.ragflow_retriever.is_available
        return self._use_ragflow

    def retrieve(self, query: str, constraints: List[Dict[str, Any]] = None,
                 top_k: int = DEFAULT_TOP_K,
                 candidate_types: Optional[List[str]] = None,
                 verbose: bool = False) -> List[Dict[str, Any]]:
        self.stats["total_retrievals"] += 1

        if not self.use_ragflow:
            print(f"[M3] the RAGFlow knowledge base "
                  f"'{self.ragflow_retriever.client.dataset_name}' is unreachable, "
                  f"no constraint retrieved")
            return []

        return self._retrieve_with_ragflow(query, constraints, top_k,
                                            candidate_types, verbose)

    @staticmethod
    def _keep_candidate(c: Dict[str, Any],
                        candidate_types: Optional[List[str]],
                        candidate_ids: Optional[set],
                        candidate_slugs: Optional[set]) -> bool:
        if candidate_ids is not None:
            if c.get("id", "") not in candidate_ids and \
               c.get("slug", "") not in candidate_slugs:
                return False
        if candidate_types:
            if c.get("type", "") not in candidate_types:
                return False
        return True

    def _retrieve_with_ragflow(self, query: str,
                                constraints: Optional[List[Dict[str, Any]]],
                                top_k: int,
                                candidate_types: Optional[List[str]],
                                verbose: bool) -> List[Dict[str, Any]]:
        self.stats["ragflow_mode"] += 1

        if verbose:
            print(f"[M3] RAGFlow mode: retrieving top_k={top_k}, "
                  f"T(t)={candidate_types or 'all types'}...")

        ragflow_results, vector_scores, keyword_scores = \
            self.ragflow_retriever.retrieve(query, top_k=top_k * 3)

        n = len(ragflow_results)
        if n == 0:
            return []

        candidate_ids = None
        candidate_slugs = None
        if constraints is not None:
            candidate_ids = {c["id"] for c in constraints}
            candidate_slugs = {c.get("slug", "") for c in constraints}

        w_s = self.weights.get("semantic", 0.7)
        w_k = self.weights.get("keyword", 0.3)

        fused = []
        for i, c in enumerate(ragflow_results):
            if not self._keep_candidate(c, candidate_types,
                                        candidate_ids, candidate_slugs):
                continue
            score = (w_s * vector_scores.get(i, 0.0) +
                     w_k * keyword_scores.get(i, 0.0))
            fused.append((i, c, score))

        fused.sort(key=lambda x: x[2], reverse=True)

        top_k = min(top_k, len(fused))
        result = []
        for i, c, score in fused[:top_k]:
            c = dict(c)
            c["_retrieval_score"] = round(score, 4)
            c["_vector_score"] = round(vector_scores.get(i, 0.0), 4)
            c["_keyword_score"] = round(keyword_scores.get(i, 0.0), 4)
            result.append(c)

        avg_score = (sum(c["_retrieval_score"] for c in result) /
                     len(result)) if result else 0
        type_dist = defaultdict(int)
        for c in result:
            type_dist[c.get("type", "unknown")] += 1
        print(f"[M3] hybrid retrieval done (RAGFlow): "
              f"{n} → {len(result)} rules, avg_score={avg_score:.3f}, "
              f"types={dict(type_dist)}")

        return result

    def retrieve_as_text(self, query: str, constraints: List[Dict[str, Any]] = None,
                         top_k: int = DEFAULT_TOP_K,
                         candidate_types: Optional[List[str]] = None,
                         verbose: bool = False) -> str:
        top_constraints = self.retrieve(query, constraints, top_k,
                                        candidate_types=candidate_types,
                                        verbose=verbose)

        lines = []
        for c in top_constraints:
            cid = c.get("id", "")
            ctype = c.get("type", "")
            desc = c.get("desc", "")
            source = c.get("source", "")
            params = c.get("params", {})

            line = f"[{cid}] [{ctype}] {desc}\n"
            line += f"  params: {json.dumps(params, ensure_ascii=False)}\n"
            line += f"  source: {source}"
            lines.append(line)

        return "\n\n".join(lines)


def create_hybrid_retriever(
        ragflow_base_url: str = RAGFLOW_BASE_URL,
        ragflow_api_key: str = RAGFLOW_API_KEY,
        weights: Optional[Dict[str, float]] = None,
) -> HybridRetriever:
    client = RAGFlowClient(
        base_url=ragflow_base_url,
        api_key=ragflow_api_key,
        dataset_name=ragflow_dataset_name(),
    )
    ragflow_retriever = RAGFlowRetriever(client=client)
    return HybridRetriever(
        ragflow_retriever=ragflow_retriever,
        weights=weights,
    )
