"""Verification-layer orchestrator (M5 + M6).

Runs the Constraint Auditor over the plan and, when violations or missing
evidence are detected, the feedback revision loop; returns the final audited
plan together with the audit report."""

from typing import List, Dict, Any

try:
    from .auditor import ConstraintAuditor, create_auditor
    from .feedback import FeedbackLoop
except ImportError:
    from verification.auditor import ConstraintAuditor, create_auditor
    from verification.feedback import FeedbackLoop


class CRAFTVerificationLayer:
    def __init__(self,
                 auditor: ConstraintAuditor = None,
                 llm_engine=None,
                 input_layer=None,
                 enable_feedback: bool = True,
                 max_iterations: int = 2):
        self.auditor = auditor or create_auditor(llm_engine=llm_engine)
        self.llm_engine = llm_engine
        self.input_layer = input_layer
        self.enable_feedback = enable_feedback

        self.feedback_loop = FeedbackLoop(
            auditor=self.auditor,
            llm_engine=self.llm_engine,
            input_layer=self.input_layer,
            max_iterations=max_iterations,
        ) if enable_feedback else None

        self.stats = {
            "total_verifications": 0,
            "direct_passes": 0,
            "corrected_passes": 0,
            "correction_iterations": [],
        }

    def verify(self, plan: Dict[str, Any],
               constraints: List[Dict[str, Any]],
               task_description: str = "",
               task_context: Dict[str, Any] = None,
               transport_capacity: Dict[str, Any] = None,
               verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_verifications"] += 1

        if verbose:
            print(f"\n{'='*50}")
            print("[Verification layer] audit start")
            print(f"[Verification layer] constraints: {len(constraints)} rules, "
                  f"resources: {len(plan.get('allocated_resources', {}))} items")
            print("=" * 50)

        current_plan = plan
        was_corrected = False
        feedback_result = None

        initial_audit = self.auditor.audit(current_plan, constraints, task_context)

        violation_count = len(initial_audit.get("violated", []))
        undefined_count = len(initial_audit.get("undefined", []))
        if verbose:
            print(f"[Verification layer] initial audit: {initial_audit['compliance_rate']:.1%} "
                  f"({violation_count} violations, {undefined_count} missing evidence)")

        if (violation_count + undefined_count) > 0 and self.enable_feedback and self.feedback_loop:
            if verbose:
                print(f"[Verification layer] Step 2: feedback revision ({violation_count} violations, "
                      f"{undefined_count} missing evidence)...")
            feedback_result = self.feedback_loop.run(
                plan=current_plan, constraints=constraints,
                task_description=task_description, task_context=task_context,
            )
            current_plan = feedback_result.get("final_plan", current_plan)
            was_corrected = feedback_result.get("iterations", 1) > 1
            self.stats["correction_iterations"].append(feedback_result.get("iterations", 0))
            if feedback_result.get("converged"):
                self.stats["corrected_passes"] += 1
        elif (violation_count + undefined_count) == 0:
            self.stats["direct_passes"] += 1

        if verbose:
            print("[Verification layer] Step 3: final audit...")
        final_audit = self.auditor.audit(current_plan, constraints, task_context)

        if verbose:
            print(f"[Verification layer] final compliance rate: {final_audit['compliance_rate']:.1%}")
            print("[Verification layer] audit done")

        return {
            "plan": current_plan,
            "audit_report": final_audit,
            "feedback_result": feedback_result,
            "was_corrected": was_corrected,
            "final_compliance_rate": final_audit.get("compliance_rate", 0),
        }

    def get_stats(self) -> Dict[str, Any]:
        avg_iterations = (sum(self.stats["correction_iterations"]) /
                          max(len(self.stats["correction_iterations"]), 1))

        return {
            **self.stats,
            "avg_correction_iterations": round(avg_iterations, 1),
            "direct_pass_rate": round(
                self.stats["direct_passes"] / max(self.stats["total_verifications"], 1), 3),
        }


def create_verification_layer(
        auditor: ConstraintAuditor = None,
        llm_engine=None,
        input_layer=None,
        max_iterations: int = 2,
) -> CRAFTVerificationLayer:
    return CRAFTVerificationLayer(
        auditor=auditor or create_auditor(llm_engine=llm_engine),
        llm_engine=llm_engine,
        input_layer=input_layer,
        max_iterations=max_iterations,
    )
