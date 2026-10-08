"""Meta layer package - Chief Orchestrator + Consensus Builder.

Exports the orchestrator (with its task-feature extractor and data classes) and
the consensus builder; the full pipeline lives in layer.py."""

from .chief_orchestrator import (
    ChiefOrchestrator,
    TaskProfiler,
    TaskProfile,
    AssemblyDecision,
    AssemblyMode,
)
from .consensus_builder import ConsensusBuilder

__all__ = [
    "ChiefOrchestrator",
    "TaskProfiler",
    "TaskProfile",
    "AssemblyDecision",
    "AssemblyMode",
    "ConsensusBuilder",
    "get_meta_layer",
]


def get_meta_layer(**kwargs):
    from .layer import CRAFTMetaLayer
    return CRAFTMetaLayer(**kwargs)
