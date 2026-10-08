"""Meta-layer orchestrator: the end-to-end CRAFT pipeline.

Phase 0 task analysis and formation decision, Phase 1 constraint retrieval,
Phase 2 planning, Phase 3 verification and feedback revision, Phase 4 makespan
optimization behind a re-audit gate, Phase 5 consensus building."""

from typing import List, Dict, Any

try:
    from .chief_orchestrator import ChiefOrchestrator
    from .consensus_builder import ConsensusBuilder
    from ..input.layer import CRAFTInputLayer
    from ..planning.layer import CRAFTPlanningLayer, create_planning_layer
    from ..verification.layer import CRAFTVerificationLayer, create_verification_layer
    from ..optimization.layer import CRAFTOptimizationLayer, create_optimization_layer
except ImportError:
    from meta.chief_orchestrator import (
        ChiefOrchestrator
    )
    from meta.consensus_builder import ConsensusBuilder
    from input.layer import CRAFTInputLayer
    from planning.layer import CRAFTPlanningLayer, create_planning_layer
    from verification.layer import CRAFTVerificationLayer, create_verification_layer
    from optimization.layer import CRAFTOptimizationLayer, create_optimization_layer


class CRAFTMetaLayer:
    def __init__(self,
                 input_layer: CRAFTInputLayer = None,
                 planning_layer: CRAFTPlanningLayer = None,
                 verification_layer: CRAFTVerificationLayer = None,
                 optimization_layer: CRAFTOptimizationLayer = None,
                 chief_orchestrator: ChiefOrchestrator = None,
                 consensus_builder: ConsensusBuilder = None,
                 top_k: int = 15,
                 feedback_max_iterations: int = 2):
        self.top_k = top_k
        self.feedback_max_iterations = feedback_max_iterations

        self._input_layer = input_layer
        self._planning_layer = planning_layer
        self._verification_layer = verification_layer
        self._optimization_layer = optimization_layer

        self.chief_orchestrator = chief_orchestrator or ChiefOrchestrator()
        self.consensus_builder = consensus_builder or ConsensusBuilder()

        self.stats = {
            "total_orchestrations": 0,
            "mode_distribution": {"light": 0, "standard": 0, "full": 0},
            "total_time_estimate": [],
        }

    @property
    def input_layer(self):
        if self._input_layer is None:
            self._input_layer = CRAFTInputLayer(top_k=self.top_k)
        return self._input_layer

    @property
    def planning_layer(self):
        if self._planning_layer is None:
            self._planning_layer = create_planning_layer(
                input_layer=self.input_layer, top_k=self.top_k)
        return self._planning_layer

    @property
    def verification_layer(self):
        if self._verification_layer is None:
            self._verification_layer = create_verification_layer(
                llm_engine=self.planning_layer.llm_engine,
                input_layer=self.input_layer,
                max_iterations=self.feedback_max_iterations,
            )
        return self._verification_layer

    @property
    def optimization_layer(self):
        if self._optimization_layer is None:
            self._optimization_layer = create_optimization_layer(
                llm_engine=self.planning_layer.llm_engine,
                input_layer=self.input_layer,
            )
        return self._optimization_layer

    def orchestrate(self,
                    task_description: str,
                    goals: List = None,
                    resources: Dict[str, Any] = None,
                    constraints: List[Dict[str, Any]] = None,
                    regions: List[Dict[str, Any]] = None,
                    vehicle_capacity: Dict[str, Any] = None,
                    task_context: Dict[str, Any] = None,
                    planning_mode: str = None,
                    enable_verification: bool = True,
                    enable_optimization: bool = True,
                    verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_orchestrations"] += 1

        if verbose:
            print(f"\n{'='*60}")
            print("[CRAFT meta layer] full-pipeline orchestration started")
            print(f"[CRAFT meta layer] task: {task_description[:100]}...")
            print("=" * 60)

        if verbose:
            print("\n[Phase 0] Chief Orchestrator task analysis...")

        constraints_count = len(constraints) if constraints else 0
        design = self.chief_orchestrator.design(
            task_description=task_description,
            constraints_count=constraints_count,
            optimize_makespan=True,
        )

        profile = design["task_profile"]
        decision = design["assembly_decision"]
        self.stats["mode_distribution"][decision.mode.value] += 1

        if planning_mode:
            decision.planning_mode = planning_mode
            if verbose:
                print(f"[CRAFT meta layer] ablation mode: planning_mode={planning_mode}")
        if not enable_verification:
            decision.activate_verification = False
            if verbose:
                print("[CRAFT meta layer] ablation mode: verification layer disabled")
        if not enable_optimization:
            decision.activate_optimization = False
            if verbose:
                print("[CRAFT meta layer] ablation mode: optimisation layer disabled")

        if verbose:
            print(self.chief_orchestrator.generate_design_report(design))

        if verbose:
            print("\n[Phase 1] Input layer: constraint retrieval...")

        constraints_text = ""
        retrieved_ids = []
        topk_constraints = []

        if decision.activate_input:
            constraints_text = self.input_layer.retrieve(
                task_description, top_k=self.top_k, verbose=verbose)
            raw = self.input_layer.retrieve_raw(
                task_description, top_k=self.top_k, verbose=False)
            topk_constraints = raw or []
            retrieved_ids = [c.get("id", c.get("constraint_id", ""))
                           for c in topk_constraints
                           if c.get("id") or c.get("constraint_id")]
            if not constraints:
                constraints = topk_constraints
        else:
            if verbose:
                print("[Phase 1] skipped (the formation decision did not activate the input layer)")

        audit_constraints = topk_constraints if topk_constraints else (constraints or [])

        if verbose:
            print(f"\n[Phase 2] Planning layer: mode={decision.planning_mode}...")

        plan_result = None
        if decision.activate_planning:
            plan_result = self.planning_layer.plan(
                task_description=task_description,
                goals=goals,
                resources=resources,
                regions=regions,
                mode=decision.planning_mode,
                constraints_text=constraints_text,
                verbose=verbose,
            )
        plan_versions = []

        if plan_result:
            plan_versions.append({
                "source": "planning",
                "plan": plan_result,
                "mode": decision.planning_mode,
            })

        current_plan = plan_result

        verification_result = None
        if decision.activate_verification and current_plan and audit_constraints:
            if verbose:
                print("\n[Phase 3] Verification layer: constraint audit + feedback revision"
                      f" (audit scope: Top-K {len(audit_constraints)} rules)...")

            verification_result = self.verification_layer.verify(
                plan=current_plan,
                constraints=audit_constraints,
                task_description=task_description,
                task_context=task_context,
                verbose=verbose,
            )

            verified_plan = verification_result.get("plan", current_plan)
            plan_versions.append({
                "source": "verification",
                "plan": verified_plan,
                "compliance_rate": verification_result.get("final_compliance_rate"),
                "was_corrected": verification_result.get("was_corrected"),
            })
            current_plan = verified_plan
        elif verbose:
            print("[Phase 3] verification layer: skipped (not activated by the formation decision)")

        optimization_result = None
        if decision.activate_optimization and current_plan:
            if verbose:
                print("\n[Phase 4] Optimisation layer: makespan optimisation...")

            optimization_result = self.optimization_layer.optimize(
                plan=current_plan,
                resources=resources,
                vehicle_capacity=vehicle_capacity,
                verbose=verbose,
            )

            optimized_plan = optimization_result.get("optimized_plan", current_plan)

            if (decision.activate_verification and audit_constraints
                    and optimized_plan is not current_plan):
                auditor = self.verification_layer.auditor
                re_audit = auditor.audit(optimized_plan, audit_constraints,
                                         task_context)
                base_audit = auditor.audit(current_plan, audit_constraints,
                                           task_context)
                new_violations = (
                    {v["constraint_id"] for v in re_audit.get("violated", [])}
                    - {v["constraint_id"] for v in base_audit.get("violated", [])}
                )
                if new_violations:
                    if verbose:
                        print(f"[Phase 4] the re-audit found new violations {sorted(new_violations)}, "
                              f"rejecting the optimised plan and falling back to the verified one")
                    optimized_plan = current_plan
                    optimization_result["optimized_plan"] = current_plan
                    optimization_result["improvement_pct"] = 0

            plan_versions.append({
                "source": "optimization",
                "plan": optimized_plan,
                "improvement_pct": optimization_result.get("improvement_pct"),
            })
            current_plan = optimized_plan
        elif verbose:
            print("[Phase 4] optimisation layer: skipped (not activated by the formation decision)")

        if verbose:
            print(f"\n[Phase 5] Consensus Builder: merging {len(plan_versions)} versions...")

        if len(plan_versions) >= 1:
            final_plan = self.consensus_builder.build(
                plan_versions=plan_versions,
                chief_orchestrator=self.chief_orchestrator,
                task_context=task_context,
                verbose=verbose,
            )
        else:
            final_plan = {"plan": "", "allocated_resources": {},
                          "assigned_personnel": {}, "schedule": [],
                          "region_response_times": {}}

        benchmark_output = self.consensus_builder.format_for_benchmark(final_plan)

        benchmark_output["_craft_meta"] = {
            "task_profile": {
                "disaster_type": profile.disaster_type,
                "region_count": profile.region_count,
                "complexity_score": profile.complexity_score,
            },
            "assembly_mode": decision.mode.value,
            "layers_activated": {
                "input": decision.activate_input,
                "planning": decision.activate_planning,
                "planning_mode": decision.planning_mode,
                "verification": decision.activate_verification,
                "optimization": decision.activate_optimization,
            },
            "version_count": len(plan_versions),
        }

        benchmark_output["_retrieved_constraints"] = topk_constraints

        llm_addressed = set(benchmark_output.get("addressed_constraints", []))
        all_addressed = list(llm_addressed | set(retrieved_ids))
        benchmark_output["addressed_constraints"] = all_addressed
        benchmark_output["_retrieved_count"] = len(retrieved_ids)
        benchmark_output["_llm_addressed_count"] = len(llm_addressed)

        if verbose:
            print(f"\n{'='*60}")
            print("[CRAFT meta layer] full-pipeline orchestration finished")
            print(f"[CRAFT meta layer] mode=pi_{decision.mode.value}, "
                  f"versions={len(plan_versions)}, "
                  f"resources={len(benchmark_output.get('allocated_resources', {}))} items, "
                  f"tasks={len(benchmark_output.get('schedule', []))}")
            print("=" * 60)

        return benchmark_output

    def quick_run(self, task_description: str,
                  resources: Dict[str, Any] = None,
                  verbose: bool = True) -> Dict[str, Any]:
        return self.orchestrate(
            task_description=task_description,
            resources=resources,
            verbose=verbose,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "orchestrator_stats": {},
            "builder_stats": self.consensus_builder.get_stats(),
        }
