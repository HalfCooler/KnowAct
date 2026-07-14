"""Prompt templates vendored from MobileWorld agent utilities."""

from guiclaw.agents.utils.prompts.gelab import (
    GELAB_INSTRUCTION_SUFFIX,
    GELAB_SYSTEM_PROMPT,
    GELAB_USER_PROMPT_TEMPLATE,
)
from guiclaw.agents.utils.prompts.general_e2e import GENERAL_E2E_PROMPT_TEMPLATE
from guiclaw.agents.utils.prompts.gui_owl_1_5 import (
    GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE,
)
from guiclaw.agents.utils.prompts.mai_ui import MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
from guiclaw.agents.utils.prompts.planner_executor import PLANNER_EXECUTOR_PROMPT_TEMPLATE
from guiclaw.agents.utils.prompts.qwen3vl import (
    MOBILE_QWEN3VL_ORIGINAL_PROMPT,
    MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER,
    MOBILE_QWEN3VL_USER_TEMPLATE,
)
from guiclaw.agents.utils.prompts.seed import SEED_PROMPT

__all__ = [
    "GENERAL_E2E_PROMPT_TEMPLATE",
    "GELAB_INSTRUCTION_SUFFIX",
    "GELAB_SYSTEM_PROMPT",
    "GELAB_USER_PROMPT_TEMPLATE",
    "GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE",
    "GUI_OWL_1_5_USER_PROMPT_TEMPLATE",
    "GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE",
    "MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP",
    "MOBILE_QWEN3VL_ORIGINAL_PROMPT",
    "MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER",
    "MOBILE_QWEN3VL_USER_TEMPLATE",
    "PLANNER_EXECUTOR_PROMPT_TEMPLATE",
    "SEED_PROMPT",
]
