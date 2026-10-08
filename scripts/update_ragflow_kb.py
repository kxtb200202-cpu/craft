"""Rebuild the RAGFlow knowledge base from a directory of constraint rule files:
delete the rules currently stored there, then upload and parse the given set.

Usage: python update_ragflow_kb.py <constraints-directory>
       python update_ragflow_kb.py F:/myc1/REALM-Bench/datasets/P7/constraints"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

from config import (RAGFLOW_BASE_URL, RAGFLOW_API_KEY, ragflow_dataset_name)
from ragflow.client import RAGFlowClient

if len(sys.argv) < 2:
    raise SystemExit("usage: python update_ragflow_kb.py <constraints-directory>")

CONSTRAINTS_DIR = Path(sys.argv[1])

if not RAGFLOW_API_KEY:
    raise SystemExit(
        "RAGFLOW_API_KEY is not set - copy .env.example to .env and fill it in.")

if not CONSTRAINTS_DIR.exists():
    raise SystemExit(f"constraints directory not found: {CONSTRAINTS_DIR}")

client = RAGFlowClient(base_url=RAGFLOW_BASE_URL, api_key=RAGFLOW_API_KEY,
                       dataset_name=ragflow_dataset_name())

print(f"RAGFlow knowledge base '{ragflow_dataset_name()}'")
client.ensure_dataset()

print("=" * 50)
print("Step 1: delete the old documents")
print("=" * 50)
deleted = client.delete_all_documents()
print(f"deleted {deleted} old documents")

time.sleep(2)

print("\n" + "=" * 50)
print("Step 2: upload the new constraint files")
print("=" * 50)
json_files = sorted(CONSTRAINTS_DIR.glob("C[0-9]*_*.json"))
print(f"found {len(json_files)} constraint files")

uploaded = 0
doc_ids = []
for fp in json_files:
    doc_id = client.upload_document(fp)
    if doc_id:
        uploaded += 1
        doc_ids.append(doc_id)
    if uploaded % 20 == 0:
        print(f"  uploaded {uploaded}/{len(json_files)}...")

print(f"upload done: {uploaded}/{len(json_files)}")

print("\n" + "=" * 50)
print("Step 3: start parsing")
print("=" * 50)

docs = client.list_documents()
all_doc_ids = [d["id"] for d in docs]
print(f"total {len(all_doc_ids)} documents to parse")

if all_doc_ids:
    success = client.start_parsing(all_doc_ids)
    print(f"parsing started: {'ok' if success else 'failed'}")

print("\n" + "=" * 50)
print("Step 4: wait for the parsing to finish")
print("=" * 50)
client.wait_for_parsing(timeout=180)

print("\n" + "=" * 50)
print("Knowledge base rebuilt!")
print("=" * 50)
