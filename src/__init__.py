"""CRAFT - Constraint-Aware Role-based Framework for multi-agent Task-planning.

Package layout (mirrors the architecture in the paper):
    agents/       - the seven agent roles (paper Table 1)
    ragflow/      - RAGFlow knowledge-base and retrieval API
    input/        - input layer (M2 constraint pre-classification + M3 hybrid retrieval)
    planning/     - planning layer (M1 task decomposition + M4 LLM planning engine)
    verification/ - verification layer (M5 constraint auditor + M6 feedback revision)
    optimization/ - optimization layer (M7 plan optimizer)
    meta/         - meta layer (Chief Orchestrator + Consensus Builder)"""

__version__ = "0.3.0"
