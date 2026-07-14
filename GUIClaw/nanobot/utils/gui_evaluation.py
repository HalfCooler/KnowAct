"""Backward-compatible shim — canonical implementation moved to guiclaw.evaluation."""

from guiclaw.evaluation import (  # noqa: F401
    DEFAULT_API_BASE,
    DEFAULT_JUDGE_MODEL,
    evaluate_gui_trajectory,
    evaluate_gui_trajectory_sync,
    filter_step_rows,
    judge_success,
    load_screenshots_for_judge,
    load_traj_rows,
)
