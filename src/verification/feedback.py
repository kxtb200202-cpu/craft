"""M6: the feedback revision loop.

Takes the audit report, re-retrieves the constraints that were violated or
lacked evidence, asks the responsible regional planner to revise the plan, and
re-audits; iterates until no FAIL/UNDEFINED verdict remains or the iteration
budget (M = 2) is exhausted."""

from typing import List, Dict, Any, Optional


class FeedbackLoop:
    def __init__(self, auditor=None, llm_engine=None, input_layer=None,
                 max_iterations: int = 2):
        self.auditor = auditor
        self.llm_engine = llm_engine
        self.input_layer = input_layer
        self.max_iterations = max_iterations

        self.history: List[Dict[str, Any]] = []

    def run(self, plan: Dict[str, Any],
            constraints: List[Dict[str, Any]],
            task_description: str = "",
            task_context: Dict = None) -> Dict[str, Any]:
        self.history = []
        current_plan = plan
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            print(f"[M6] audit round {iteration}/{self.max_iterations}...")

            audit = self.auditor.audit(current_plan, constraints, task_context)

            self.history.append({
                "iteration": iteration,
                "plan": current_plan,
                "audit": audit,
            })

            violated_count = len(audit.get("violated", []))
            undefined_count = len(audit.get("undefined", []))
            blocking_count = violated_count + undefined_count
            compliance = audit.get("compliance_rate", 0)

            print(f"[M6] compliance rate: {compliance:.1%} "
                  f"(violated {violated_count} rules, missing evidence {undefined_count} rules)")

            if blocking_count == 0:
                print(f"[M6] every constraint passed, converged after {iteration} rounds")
                converged = True
                break

            if iteration == self.max_iterations:
                print(f"[M6] reached the iteration limit ({self.max_iterations}), "
                      f"keeping the best plan found and marking it as not fully satisfied")
                break

            correction_prompt = self._build_correction_prompt(
                audit, current_plan, task_description)

            additional_constraints = self._re_retrieve(audit, task_description)

            print(f"[M6] revision round {iteration}: re-planning...")
            try:
                revised_plan = self._replan(current_plan, correction_prompt,
                                             additional_constraints)
                if revised_plan:
                    current_plan = revised_plan
                else:
                    print("[M6] re-planning failed, keeping the current plan")
                    break
            except Exception as e:
                print(f"[M6] re-planning error: {e}")
                break

        return {
            "final_plan": current_plan,
            "final_audit": self.history[-1]["audit"] if self.history else {},
            "iterations": len(self.history),
            "converged": converged,
            "history": self.history,
        }

    def _build_correction_prompt(self, audit: Dict[str, Any],
                                   plan: Dict[str, Any],
                                   task_description: str) -> str:
        violated = audit.get("violated", [])
        undefined = audit.get("undefined", [])
        if not violated and not undefined:
            return ""

        lines = [
            "## The constraint audit found the following problems; revise the plan",
            "",
        ]

        if violated:
            lines.append("### 1. Violated clauses")
            lines.append("")
            for i, v in enumerate(violated, 1):
                lines.append(f"{i}. [{v['constraint_id']}] {v['desc']}")
                lines.append(f"   field: {v.get('field', '?')}")
                lines.append(f"   current value: {v.get('actual_value', '?')}")
                lines.append(f"   constraint limit: {v.get('limit', '?')}")
                lines.append(f"   deviation: {v.get('deviation_pct', '?')}%")
                lines.append(f"   violation type: {v.get('reason', '?')}")
                lines.append("")

        if undefined:
            lines.append("### 2. Missing evidence (the fields required for the verdict were not provided; counted as unsatisfied)")
            lines.append("")
            for i, u in enumerate(undefined, 1):
                evidence = ", ".join(u.get("required_evidence") or []) or "see the constraint definition"
                lines.append(f"{i}. [{u['constraint_id']}] {u['desc']}")
                lines.append(f"   missing evidence: {evidence}")
                lines.append(f"   note: {u.get('actual_summary', '')}")
                lines.append("")

        lines.append("## Revision requirements")
        lines.append("1. Adjust the offending values above so that they strictly satisfy the constraint limits")
        lines.append("2. For every entry with missing evidence, add the corresponding evidence field to the plan:"
                     "schedule[].region_id / schedule[].task_tag, carriers, events;"
                     "fill in only what the plan really arranges, never invent work that is not actually carried out")
        lines.append("3. If a constraint is irrelevant to the task, explain why in the plan field")
        lines.append("4. Leave every other field that already passed unchanged")
        lines.append("5. Output the complete plan as JSON, not only the changed part")
        lines.append("")
        lines.append(f"## Original task\n{task_description}")

        return "\n".join(lines)

    def _re_retrieve(self, audit: Dict[str, Any],
                       task_description: str) -> str:
        if not self.input_layer:
            return ""

        violated_ids = ([v["constraint_id"] for v in audit.get("violated", [])] +
                        [u["constraint_id"] for u in audit.get("undefined", [])])
        if not violated_ids:
            return ""

        query = f"{task_description} {' '.join(violated_ids)}"
        try:
            print(f"[M6] second retrieval: the query carries {len(violated_ids)} violated constraints")
            additional = self.input_layer.retrieve(
                query, top_k=min(len(violated_ids) * 3, 20), verbose=False)
            return additional
        except Exception as e:
            print(f"[M6] second retrieval failed: {e}")
            return ""

    def _replan(self, original_plan: Dict[str, Any],
                 correction_prompt: str,
                 additional_constraints: str) -> Optional[Dict[str, Any]]:
        if not self.llm_engine:
            return None

        full_prompt = correction_prompt
        if additional_constraints:
            full_prompt += f"\n\n## Additional constraints (second retrieval)\n{additional_constraints}"

        try:
            result = self.llm_engine.plan(
                task_description=full_prompt,
                constraints_text=additional_constraints,
                verbose=False,
            )
            if not result.get("allocated_resources") and original_plan.get("allocated_resources"):
                result["allocated_resources"] = original_plan["allocated_resources"]
                result["assigned_personnel"] = original_plan["assigned_personnel"]
            return result
        except Exception as e:
            print(f"[M6] LLM re-planning failed: {e}")
            return None
