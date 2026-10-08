"""Agent roles and prompt templates of CRAFT.

    roles.py   - the seven AgentRole definitions (paper Table 1)
    prompts.py - system-prompt templates, prompt builders, JSON extraction"""

from .roles import (
    AgentRole,
    ALL_ROLES,
    get_role,
    get_roles_by_layer,
    get_roles_by_activation,
    get_assembly_templates,
    print_role_summary,
)

from .prompts import (
    CRAFT_SYSTEM_PROMPT,
    GLOBAL_STRATEGY_SYSTEM_PROMPT,
    REGIONAL_EXECUTION_SYSTEM_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    SUB_TASK_TEMPLATE,
    CONSTRAINT_SECTION_HEADER,
    CONSTRAINT_SECTION_FOOTER,
    build_task_description,
    build_global_strategy_prompt,
    build_regional_execution_prompt,
    extract_json_from_response,
)

__all__ = [
    "AgentRole",
    "ALL_ROLES",
    "get_role",
    "get_roles_by_layer",
    "get_roles_by_activation",
    "get_assembly_templates",
    "print_role_summary",
    "CRAFT_SYSTEM_PROMPT",
    "GLOBAL_STRATEGY_SYSTEM_PROMPT",
    "REGIONAL_EXECUTION_SYSTEM_PROMPT",
    "OPTIMIZER_SYSTEM_PROMPT",
    "SUB_TASK_TEMPLATE",
    "CONSTRAINT_SECTION_HEADER",
    "CONSTRAINT_SECTION_FOOTER",
    "build_task_description",
    "build_global_strategy_prompt",
    "build_regional_execution_prompt",
    "extract_json_from_response",
]
