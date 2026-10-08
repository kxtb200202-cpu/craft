"""RAGFlow API client (RAGFlow v0.24.0).

Wraps document management (upload, delete, parse status) and hybrid retrieval
(semantic vector + keyword) against the RAGFlow service."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


class RAGFlowClient:
    def __init__(self,
                 base_url: str = "http://127.0.0.1:9380",
                 api_key: str = "",
                 dataset_name: Optional[str] = None):
        if not dataset_name:
            try:
                from ..config import ragflow_dataset_name
            except ImportError:
                from config import ragflow_dataset_name
            dataset_name = ragflow_dataset_name()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dataset_name = dataset_name
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        self._dataset_id: Optional[str] = None

    def get_dataset_id(self) -> str:
        if self._dataset_id:
            return self._dataset_id

        url = f"{self.base_url}/api/v1/datasets"
        resp = requests.get(url, headers=self._headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data["code"] != 0:
            raise RuntimeError(f"listing the knowledge bases failed: {data}")

        for ds in data["data"]:
            if ds["name"] == self.dataset_name:
                self._dataset_id = ds["id"]
                chunk_count = ds.get("chunk_count", "?")
                doc_count = ds.get("document_count", "?")
                print(f"[RAGFlow] knowledge base '{self.dataset_name}' "
                      f"(id={self._dataset_id}, chunks={chunk_count}, docs={doc_count})")
                return self._dataset_id

        raise RuntimeError(f"knowledge base not found: '{self.dataset_name}', create it in RAGFlow first")

    def create_dataset(self, name: str = None) -> str:
        name = name or self.dataset_name
        url = f"{self.base_url}/api/v1/datasets"
        resp = requests.post(url, headers=self._headers,
                             json={"name": name}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"creating the knowledge base failed: {data}")
        self.dataset_name = name
        self._dataset_id = data["data"]["id"]
        print(f"[RAGFlow] created the knowledge base '{name}' (id={self._dataset_id})")
        return self._dataset_id

    def ensure_dataset(self) -> str:
        try:
            return self.get_dataset_id()
        except RuntimeError:
            return self.create_dataset()

    def list_documents(self) -> List[Dict[str, Any]]:
        ds_id = self.get_dataset_id()
        all_docs = []
        page = 1

        while True:
            url = f"{self.base_url}/api/v1/datasets/{ds_id}/documents"
            params = {"page": page, "page_size": 100}
            resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            data = resp.json()

            if data["code"] != 0:
                raise RuntimeError(f"listing the documents failed: {data}")

            docs = data["data"].get("docs", [])
            all_docs.extend(docs)
            if len(docs) < 100:
                break
            page += 1

        return all_docs

    def delete_document(self, doc_id: str) -> bool:
        ds_id = self.get_dataset_id()
        url = f"{self.base_url}/api/v1/datasets/{ds_id}/documents"
        payload = {"ids": [doc_id]}
        resp = requests.delete(url, headers=self._headers, json=payload, timeout=30)
        data = resp.json()
        return data.get("code") == 0

    def delete_all_documents(self) -> int:
        docs = self.list_documents()
        if not docs:
            print("[RAGFlow] the knowledge base is empty, nothing to delete")
            return 0

        doc_ids = [d["id"] for d in docs]
        print(f"[RAGFlow] deleting {len(doc_ids)} old documents...")

        ds_id = self.get_dataset_id()
        url = f"{self.base_url}/api/v1/datasets/{ds_id}/documents"
        total_deleted = 0
        batch_size = 32

        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i + batch_size]
            payload = {"ids": batch}
            resp = requests.delete(url, headers=self._headers, json=payload, timeout=30)
            data = resp.json()
            if data.get("code") == 0:
                total_deleted += len(batch)
            else:
                print(f"[RAGFlow] deleting a batch failed: {data}")

        print(f"[RAGFlow] deleted {total_deleted}/{len(doc_ids)} documents")
        return total_deleted

    def upload_document(self, file_path) -> Optional[str]:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        ds_id = self.get_dataset_id()
        url = f"{self.base_url}/api/v1/datasets/{ds_id}/documents"

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/json")}
            upload_headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.post(url, headers=upload_headers, files=files, timeout=30)

        data = resp.json()
        if data.get("code") == 0:
            doc_id = data["data"][0]["id"] if data.get("data") else None
            return doc_id
        else:
            print(f"[RAGFlow] upload failed {file_path.name}: {data}")
            return None

    def upload_constraints(self, constraints_dir: Path) -> int:
        json_files = sorted(constraints_dir.glob("C[0-9]*_*.json"))
        if not json_files:
            print(f"[RAGFlow] no constraint file found in: {constraints_dir}")
            return 0

        print(f"[RAGFlow] uploading {len(json_files)} constraint files...")
        uploaded = 0

        for fp in json_files:
            doc_id = self.upload_document(fp)
            if doc_id:
                uploaded += 1
            time.sleep(0.05)

        print(f"[RAGFlow] upload done: {uploaded}/{len(json_files)} files")
        return uploaded

    def start_parsing(self, doc_ids: List[str]) -> bool:
        ds_id = self.get_dataset_id()
        url = f"{self.base_url}/api/v1/datasets/{ds_id}/chunks"
        payload = {"document_ids": doc_ids}
        resp = requests.post(url, headers=self._headers, json=payload, timeout=30)
        data = resp.json()
        return data.get("code") == 0

    def start_parsing_all(self) -> bool:
        docs = self.list_documents()
        if not docs:
            return False
        doc_ids = [d["id"] for d in docs]
        print(f"[RAGFlow] starting the parsing of {len(doc_ids)} documents...")
        return self.start_parsing(doc_ids)

    def wait_for_parsing(self, timeout: int = 120) -> bool:
        ds_id = self.get_dataset_id()
        url = f"{self.base_url}/api/v1/datasets/{ds_id}"
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(url, headers=self._headers, timeout=10)
            data = resp.json()
            if data.get("code") != 0:
                return False

            ds_info = data.get("data", {})
            doc_count = ds_info.get("document_count", 0)
            chunk_count = ds_info.get("chunk_count", 0)
            status = ds_info.get("status", "")

            print(f"[RAGFlow] parsing status: status={status}, "
                  f"docs={doc_count}, chunks={chunk_count}")

            if status == "1" and chunk_count > 0:
                print(f"[RAGFlow] every document parsed (chunks={chunk_count})")
                return True

            time.sleep(3)

        print("[RAGFlow] parsing timed out")
        return False

    def retrieve(self, query: str, top_k: int = 20,
                 similarity_threshold: float = 0.1,
                 metadata_condition: dict = None,
                 rerank_id: str = "",
                 dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        ds_id = dataset_id or self.get_dataset_id()

        url = f"{self.base_url}/api/v1/retrieval"
        payload = {
            "question": query,
            "dataset_ids": [ds_id],
            "top_k": 1024,
            "page_size": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": 0.7,
            "keyword": True,
        }
        if metadata_condition:
            payload["metadata_condition"] = metadata_condition
        if rerank_id:
            payload["rerank_id"] = rerank_id

        try:
            resp = requests.post(url, headers=self._headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                print(f"[RAGFlow] retrieval failed: {data}")
                return []

            chunks = data.get("data", {}).get("chunks", [])
            return chunks

        except requests.exceptions.RequestException as e:
            print(f"[RAGFlow] retrieval request failed: {e}")
            return []

    def retrieve_as_constraints(self, query: str, top_k: int = 20,
                                 metadata_condition: dict = None,
                                 similarity_threshold: float = 0.1,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
        chunks = self.retrieve(query, top_k,
                               similarity_threshold=similarity_threshold,
                               metadata_condition=metadata_condition)

        if not chunks:
            return []

        constraints = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            try:
                constraint = json.loads(content)
                constraint["_ragflow_similarity"] = round(
                    chunk.get("similarity", 0), 4)
                constraint["_vector_similarity"] = round(
                    chunk.get("vector_similarity", 0), 4)
                constraint["_term_similarity"] = round(
                    chunk.get("term_similarity", 0), 4)
                constraint["_chunk_id"] = chunk.get("id", "")
                constraint["_document_keyword"] = chunk.get("document_keyword", "")
                constraints.append(constraint)
            except json.JSONDecodeError:
                constraints.append({
                    "id": chunk.get("document_keyword", "unknown"),
                    "type": "unknown",
                    "desc": content[:100],
                    "source": "RAGFlow retrieval",
                    "params": {},
                    "_ragflow_similarity": round(chunk.get("similarity", 0), 4),
                    "_vector_similarity": round(chunk.get("vector_similarity", 0), 4),
                    "_term_similarity": round(chunk.get("term_similarity", 0), 4),
                    "_raw_content": content,
                })

        if verbose:
            types = {}
            for c in constraints:
                t = c.get("type", "unknown")
                types[t] = types.get(t, 0) + 1
            avg_sim = (sum(c.get("_ragflow_similarity", 0) for c in constraints) /
                       len(constraints)) if constraints else 0
            print(f"[RAGFlow] retrieval done: top_k={top_k} → {len(constraints)} rules, "
                  f"avg_similarity={avg_sim:.3f}, types={types}")

        return constraints

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/v1/datasets",
                                headers=self._headers, timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False


def create_ragflow_client(
    base_url: str = "http://127.0.0.1:9380",
    api_key: str = "",
    dataset_name: Optional[str] = None,
) -> RAGFlowClient:
    return RAGFlowClient(
        base_url=base_url,
        api_key=api_key,
        dataset_name=dataset_name,
    )
