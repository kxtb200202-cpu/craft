# CRAFT — Constraint-Aware Retrieval-Augmented Multi-Agent Task Planning

CRAFT plans constraint-dense tasks (e.g. disaster-relief deployment) with a team of LLM
agents whose behavior is bounded by explicit constraint rules. Three design choices
distinguish it from ordinary RAG pipelines:

1. **Constraint rules are the retrieval corpus** — not documents. Each rule is a
   structured JSON record; a hybrid retriever selects the Top-K most relevant rules and
   injects them into the planning context.
2. **The agent team is assembled dynamically.** A Chief Orchestrator reads a task
   profile, decides the formation and issues a global resource strategy. Roles follow an
   information-asymmetry principle: no single agent can see the whole task.
3. **Verification is deterministic.** An auditor agent, independent of the planning chain
   and blind to its reasoning traces, checks every rule with code only — no LLM takes
   part in auditing, so repeated audits of the same plan are reproducible.

Evaluated on REALM-Bench (5 domains × 100 randomized instances × 3 runs × 7 configs =
10,500 runs) and cross-checked on TravelPlanner.

The constraint knowledge base lives in RAGFlow — dataset `constraints_p7` by default, the
P7 disaster-relief rules — and is queried over the RAGFlow API at retrieval time, so no
constraint file is read from disk.

## Agents

| Agent | Responsibility | Key output |
|---|---|---|
| Chief Orchestrator | task features → formation → global strategy | allocation ratios, priority order |
| Constraint Curator | pre-classify rules; hybrid retrieval | Top-K constraint rules |
| Regional Planner ×N | plan one task unit under its quota | local plan (resources, personnel, schedule) |
| Resource Arbitrator | settle cross-region claims | final per-region allocation |
| Constraint Auditor | deterministic rule-by-rule verification | audit report |
| Plan Optimizer | shorten makespan of a verified plan | adjusted schedule |
| Consensus Builder | merge all outputs | final plan |

## Running the experiments

Experiments are launched from the REALM-Bench root, which discovers this directory as the
`craft` framework through `evaluation/framework_runners.py`.

```bash
# once: dependencies, then the endpoints and keys (LLM / embeddings / RAGFlow)
pip install -r requirements.txt
cp .env.example .env

# component self-tests (offline, no LLM and no RAGFlow needed)
python scripts/test_all_modules.py

# 100 randomized P7 (disaster relief) instances
python run_P7_instance_experiments.py --frameworks craft --runs 50 --no-viz

# multi-framework comparison on the same instances
python run_evaluation.py --frameworks langgraph,craft --tasks P7 --runs 3 --no-viz

# cheap smoke test: no LLM calls
python run_evaluation.py --mock

# resume an interrupted batch, then post-process
python run_remaining_experiments.py
python scripts/recompute_results.py
python scripts/generate_summary.py
```

Every run calls the LLM and embedding endpoints configured in `.env`, so start with
`--runs 1` or `--mock` to check the setup first; results land in `evaluation_results/`.
Ablation configurations are registered in `src/craft_runner.py` as `CRAFT-C` / `-P` / `-O`
/ `-A` / `-full`.
