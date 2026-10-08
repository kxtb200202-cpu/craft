"""Meta layer: the Chief Orchestrator.

Extracts the task features, decides the agent formation (pi_light / pi_standard
/ pi_full) and the execution plan, issues the global resource strategy, and
settles disputes escalated from the other agents."""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class AssemblyMode(Enum):
    LIGHT = "light"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class TaskProfile:
    disaster_type: str = "general"
    region_count: int = 1
    constraint_density: int = 0
    severity_mix: str = "single"
    optimize_makespan: bool = False
    complexity_score: float = 0.0

    raw_description: str = ""
    extracted_keywords: List[str] = field(default_factory=list)


@dataclass
class AssemblyDecision:
    mode: AssemblyMode = AssemblyMode.LIGHT
    activate_input: bool = True
    activate_planning: bool = True
    activate_verification: bool = False
    activate_optimization: bool = False

    planning_mode: str = "global"

    reasoning: List[str] = field(default_factory=list)


class TaskProfiler:
    DISASTER_KEYWORDS = {
        "earthquake": ["earthquake", "quake", "collapse", "collapsed",
                       "aftershock", "epicenter", "magnitude"],
        "flood":      ["flood", "flooding", "inundation", "dam break",
                       "waterlogging", "rainstorm", "downpour"],
        "epidemic":   ["epidemic", "outbreak", "infectious", "infection",
                       "quarantine", "virus", "pandemic"],
        "chemical":   ["chemical", "hazardous", "toxic", "leak", "spill"],
        "storm":      ["typhoon", "hurricane", "storm", "cyclone"],
        "conflict":   ["conflict", "war", "refugee", "displaced", "displacement"],
    }

    def profile(self, task_description: str,
                constraints_count: int = 0,
                optimize: bool = False) -> TaskProfile:
        disaster_type = self._detect_disaster_type(task_description)

        region_count = self._detect_region_count(task_description)

        severity_mix = self._detect_severity_mix(task_description)

        keywords = self._extract_keywords(task_description)

        density = constraints_count if constraints_count > 0 else self._estimate_density(task_description)

        complexity = self._compute_complexity(
            region_count, density, severity_mix)

        return TaskProfile(
            disaster_type=disaster_type,
            region_count=region_count,
            constraint_density=density,
            severity_mix=severity_mix,
            optimize_makespan=optimize,
            complexity_score=round(complexity, 2),
            raw_description=task_description,
            extracted_keywords=keywords,
        )

    def _detect_disaster_type(self, text: str) -> str:
        scores = {}
        lowered = text.lower()
        for dtype, keywords in self.DISASTER_KEYWORDS.items():
            scores[dtype] = sum(1 for kw in keywords if kw in lowered)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "earthquake"

    def _detect_region_count(self, text: str) -> int:
        named = re.findall(r'[Rr]egion\s+([A-Za-z0-9_-]+)', text)
        if named:
            return len(set(named))
        numbered = re.findall(r'[Rr]egion\s*(\d+)', text)
        if numbered:
            return len(set(numbered))
        areas = re.findall(r'(?:area|zone)\s*[-_ ]?([A-Za-z0-9])', text, re.IGNORECASE)
        if areas:
            return len(set(areas))
        pops = re.findall(r'([\d,]+)\s*(?:people|persons|population|residents)',
                          text, re.IGNORECASE)
        if len(pops) >= 2:
            return len(pops)
        return 1

    def _detect_severity_mix(self, text: str) -> str:
        severities = []
        if re.search(r'critical|severe|extreme|high', text, re.IGNORECASE):
            severities.append("critical")
        if re.search(r'urgent|moderate|medium', text, re.IGNORECASE):
            severities.append("urgent")
        if re.search(r'normal|low|routine|minor', text, re.IGNORECASE):
            severities.append("normal")
        return "mixed" if len(severities) >= 2 else "single"

    def _extract_keywords(self, text: str) -> List[str]:
        keywords = []
        feature_map = {
            "collapse": "collapse", "collapsed": "collapse", "trapped": "trapped",
            "power cut": "power_out", "power outage": "power_out",
            "water cut": "water_out", "water outage": "water_out",
            "children": "children", "elderly": "elderly",
            "chemical": "chemical", "search and rescue": "search_rescue",
            "medical": "medical",
        }
        lowered = text.lower()
        for kw, tag in feature_map.items():
            if kw in lowered and tag not in keywords:
                keywords.append(tag)
        return keywords

    def _estimate_density(self, text: str) -> int:
        length = len(text)
        if length < 100:
            return 10
        elif length < 500:
            return 30
        else:
            return 50

    def _compute_complexity(self, regions: int, density: int,
                             severity_mix: str) -> float:
        score = 0.0
        if regions >= 3:
            score += 0.3
        elif regions == 2:
            score += 0.15
        if density > 30:
            score += 0.3
        elif density > 10:
            score += 0.15
        if severity_mix == "mixed":
            score += 0.2
        return min(score, 1.0)


class ChiefOrchestrator:
    def __init__(self, profiler: TaskProfiler = None):
        self.profiler = profiler or TaskProfiler()

    def design(self, task_description: str,
               constraints_count: int = 0,
               optimize_makespan: bool = False) -> Dict[str, Any]:
        profile = self.profiler.profile(
            task_description,
            constraints_count=constraints_count,
            optimize=optimize_makespan,
        )

        decision = self.decide_assembly(profile)

        execution_plan = self.plan_execution(profile, decision)

        return {
            "task_profile": profile,
            "assembly_decision": decision,
            "execution_plan": execution_plan,
        }

    def decide_assembly(self, profile: TaskProfile) -> AssemblyDecision:
        decision = AssemblyDecision()
        reasoning = []

        decision.activate_input = True
        reasoning.append("the input layer is always active (constraint retrieval is the CRAFT foundation)")

        decision.activate_planning = True

        if profile.region_count >= 3:
            decision.planning_mode = "collaborative"
            reasoning.append(f"regions={profile.region_count}>=3 -> collaborative planning mode")
        elif profile.region_count == 2:
            if profile.complexity_score > 0.5:
                decision.planning_mode = "collaborative"
                reasoning.append(f"2 regions + high complexity ({profile.complexity_score:.2f}) -> collaborative planning")
            else:
                decision.planning_mode = "decomposed"
                reasoning.append("2 regions -> decomposed planning mode")
        else:
            decision.planning_mode = "global"
            reasoning.append("a single region -> global planning mode")

        if profile.constraint_density > 10 or profile.complexity_score > 0.4:
            decision.activate_verification = True
            reasoning.append(
                f"constraint density={profile.constraint_density}>10 or "
                f"complexity={profile.complexity_score:.2f}>0.4 -> activate the verification layer")
        else:
            reasoning.append("low constraint density -> skip the verification layer (the planner self-check suffices)")

        if profile.optimize_makespan or profile.complexity_score > 0.5:
            decision.activate_optimization = True
            reasoning.append("makespan optimisation needed -> activate the optimisation layer")
        else:
            reasoning.append("no optimisation needed -> skip the optimisation layer")

        if decision.activate_verification and decision.activate_optimization:
            decision.mode = AssemblyMode.FULL
        elif profile.region_count >= 2:
            decision.mode = AssemblyMode.STANDARD
        else:
            decision.mode = AssemblyMode.LIGHT

        decision.reasoning = reasoning
        return decision

    def plan_execution(self, profile: TaskProfile,
                        decision: AssemblyDecision) -> List[Dict[str, Any]]:
        steps = []

        steps.append({
            "step": 1,
            "layer": "input",
            "action": "constraint retrieval",
            "depends_on": [],
            "description": f"RAGFlow hybrid retrieval of the Top-K constraints (density ~{profile.constraint_density})",
        })

        steps.append({
            "step": 2,
            "layer": "planning",
            "action": f"{decision.planning_mode}_planning",
            "depends_on": [1],
            "description": f"mode={decision.planning_mode}, regions={profile.region_count}",
        })

        if decision.activate_verification:
            steps.append({
                "step": 3,
                "layer": "verification",
                "action": "audit_and_correct",
                "depends_on": [2],
                "description": "constraint audit (M5) + feedback revision (M6, up to 2 rounds)",
            })

        if decision.activate_optimization:
            step4 = {
                "step": 4,
                "layer": "optimization",
                "action": "optimize",
                "depends_on": [3 if decision.activate_verification else 2],
                "description": "M7 Plan Optimizer: task parallelisation + critical-path compression, the candidate is audited again",
            }
            steps.append(step4)

        return steps

    def resolve_dispute(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        ctype = conflict.get("type", "")

        if ctype == "resource_conflict":
            return {
                "resolution": "conservative",
                "value": conflict.get("conservative_value", conflict.get("value_a")),
                "reason": "safety-side principle: take the more conservative allocation",
            }
        elif ctype == "schedule_conflict":
            return {
                "resolution": "latest_time",
                "value": max(conflict.get("time_a", 0), conflict.get("time_b", 0)),
                "reason": "safety-side principle: take the more conservative time estimate",
            }
        elif ctype == "constraint_disagreement":
            return {
                "resolution": "auditor_wins",
                "reason": "the Constraint Auditor holds the final authority on interpreting a constraint",
            }
        else:
            return {
                "resolution": "escalate",
                "reason": f"unknown dispute type: {ctype}, manual intervention required",
            }

    def generate_design_report(self, design_result: Dict[str, Any]) -> str:
        profile = design_result["task_profile"]
        decision = design_result["assembly_decision"]
        plan = design_result["execution_plan"]

        lines = [
            "=" * 50,
            "CRAFT Chief Orchestrator - formation decision report",
            "=" * 50,
            "",
            f"disaster type: {profile.disaster_type}",
            f"regions: {profile.region_count}",
            f"constraint density: {profile.constraint_density}",
            f"severity: {profile.severity_mix}",
            f"overall complexity: {profile.complexity_score:.2f}",
            f"formation: pi_{decision.mode.value}",
            "",
            "active layers:",
            f"  input layer: {'[ON]' if decision.activate_input else '[OFF]'}",
            f"  planning layer: {'[ON]' if decision.activate_planning else '[OFF]'} (mode={decision.planning_mode})",
            f"  verification layer: {'[ON]' if decision.activate_verification else '[OFF]'}",
            f"  optimisation layer: {'[ON]' if decision.activate_optimization else '[OFF]'}",
            "",
            "reasons:",
        ]
        for r in decision.reasoning:
            lines.append(f"  - {r}")

        lines.append("")
        lines.append("execution plan:")
        for s in plan:
            deps = f" (depends on steps: {s['depends_on']})" if s['depends_on'] else ""
            lines.append(f"  Step {s['step']}: [{s['layer']}] {s['action']}{deps}")
            lines.append(f"    {s['description']}")

        return "\n".join(lines)
