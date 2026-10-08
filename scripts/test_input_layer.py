"""Component self-test for the input layer (M2 pre-classification, M3 retrieval).

Both modules read their constraints from the RAGFlow knowledge base, so the
retrieval sections only produce results when the RAGFlow service is running and
the knowledge base has been built (see update_ragflow_kb.py)."""

import sys
import os
import json

os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from input.preclassifier import ConstraintPreClassifier
from input.retriever import HybridRetriever
from input.layer import CRAFTInputLayer


def test_module_m2():
    print("\n" + "=" * 60)
    print("Test M2: constraint pre-classifier")
    print("=" * 60)

    pc = ConstraintPreClassifier()

    test_cases = [
        ("Earthquake disaster relief: many buildings collapsed, people trapped; allocate "
         "drinking water and medical supplies, deploy doctors and nurses within 6 hours",
         "expects resource + personnel + deadline"),
        ("Flood disaster: drinking water contaminated, water purification and sanitation "
         "supplies needed", "expects resource"),
        ("Post-disaster resettlement: deploy enough medical staff and keep the bed "
         "capacity under control", "expects capacity"),
        ("Emergency response phase: search and rescue and supply delivery must finish "
         "within the golden rescue window", "deadline"),
    ]

    for desc, label in test_cases:
        print(f"\n--- Scenario: {label} ---")
        print(f"Description: {desc[:60]}...")
        types = pc.select_types(desc)
        print(f"  T(t) = {types}")
        result = pc.classify(desc)
        total = pc.total_count
        if total == 0:
            print("  the knowledge base is empty or unreachable - check RAGFlow")
            continue
        print(f"  candidate set: {total} -> {len(result)} rules")
        for c in result[:3]:
            print(f"    {c['id']} [{c['type']}] {c['desc']}")


def test_module_m3():
    print("\n" + "=" * 60)
    print("Test M3: hybrid retrieval engine")
    print("=" * 60)

    from config import HYBRID_WEIGHTS, ragflow_dataset_name
    print(f"knowledge base: {ragflow_dataset_name()}")
    print(f"fusion weights: {HYBRID_WEIGHTS}")

    query = ("Earthquake disaster relief: region1 is a severely affected area with 3000 "
             "people, region2 is a moderately affected area with 1500 people. Allocate "
             "medical supplies, food, drinking water and shelter materials, and deploy "
             "medical staff and transport vehicles.")

    print(f"\nQuery: {query[:80]}...")

    hr = HybridRetriever()
    if not hr.use_ragflow:
        print("  RAGFlow is unreachable - start the service and rebuild the knowledge "
              "base with update_ragflow_kb.py, then rerun this test")
        return

    top = hr.retrieve(query, top_k=10,
                      candidate_types=["resource", "capacity", "deadline"],
                      verbose=True)
    if not top:
        print("  no constraint returned - is the knowledge base built?")
        return

    print("\nTop 10 hybrid retrieval results (dual channel):")
    for i, c in enumerate(top, 1):
        print(f"  {i}. {c['id']} [{c['type']}] {c['desc']}")
        print(f"     sem={c['_vector_score']:.3f} kw={c['_keyword_score']:.3f} "
              f"-> final={c['_retrieval_score']:.3f}")


def test_input_layer():
    print("\n" + "=" * 60)
    print("Test CRAFT input layer (full M2 + M3 pipeline)")
    print("=" * 60)

    input_layer = CRAFTInputLayer(top_k=15)

    task = ("Disaster relief scenario: two affected regions need resource allocation. "
            "Region1: critical severity, 3000 people; "
            "Region2: urgent severity, 1500 people. "
            "Allocate medical supplies, food, drinking water and shelter materials, "
            "deploy medical staff and transport vehicles (trucks and helicopters). "
            "Meet the deadlines and keep the allocation fair.")

    text = input_layer.retrieve(task, verbose=True)
    if not text:
        print("  nothing retrieved - check RAGFlow")
    print("\n--- formatted output (first 500 chars) ---")
    print(text[:500])

    stats = input_layer.get_stats()
    print("\n--- retrieval statistics ---")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    return input_layer


def test_ablation():
    print("\n" + "=" * 60)
    print("Ablation: comparing dual-channel weight configurations")
    print("=" * 60)

    query = "Earthquake disaster: medical care and search and rescue needed"

    configs = [
        ("semantic only", {"semantic": 1.0, "keyword": 0.0}),
        ("keyword only", {"semantic": 0.0, "keyword": 1.0}),
        ("paper default alpha=0.7 / beta=0.3", {"semantic": 0.7, "keyword": 0.3}),
    ]

    for name, weights in configs:
        hr = HybridRetriever(weights=weights)
        if not hr.use_ragflow:
            print(f"\n{name}: skipped, RAGFlow is unreachable")
            continue
        top = hr.retrieve(query, top_k=5, verbose=False)
        top_ids = [c["id"] for c in top]
        print(f"\n{name}: {top_ids}")


if __name__ == "__main__":
    test_module_m2()

    test_module_m3()

    test_input_layer()

    test_ablation()

    print("\n" + "=" * 60)
    print("All tests finished!")
    print("=" * 60)
