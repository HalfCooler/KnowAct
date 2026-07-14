"""
guiclaw.evaluation
==================
Reusable evaluation helpers for single GUI trajectories.

Uses an OpenAI-compatible vision-language model to judge whether a GUI task
was completed successfully based on the execution trace and screenshots.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import json_repair
from openai import OpenAI

from guiclaw.trajectory.recorder import load_trajectory_events

DEFAULT_JUDGE_MODEL = "qwen3-vl-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_traj_rows(traj_path: Path, *, subtask_index: int = 1) -> list[dict[str, Any]]:
    return load_trajectory_events(traj_path, subtask_index=subtask_index)


def filter_step_rows(traj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in traj_rows if row.get("type") == "step"]


def load_screenshots_for_judge(
    traj_path: Path,
    traj_rows: list[dict[str, Any]],
) -> list[bytes]:
    screenshot_bytes: list[bytes] = []
    screenshot_dir = traj_path.parent

    for row in traj_rows:
        screenshot_ref = row.get("screenshot_file") or row.get("screenshot_path")
        if not screenshot_ref:
            continue
        ss_path = Path(str(screenshot_ref))
        if not ss_path.is_absolute():
            ss_path = screenshot_dir / str(screenshot_ref)
        if not ss_path.exists():
            continue
        try:
            screenshot_bytes.append(ss_path.read_bytes())
        except Exception:
            continue

    return screenshot_bytes


def _extract_json_obj(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def judge_success(
    client: OpenAI,
    instruction: str,
    traj_rows: list[dict[str, Any]],
    screenshots: list[bytes],
    model: str,
    *,
    task_id: str = "gui-task",
) -> tuple[bool, str]:
    trajectory_excerpt: list[dict[str, Any]] = []
    for row in traj_rows:
        trajectory_excerpt.append(
            {
                "step_num": row.get("step_num"),
                "action": row.get("action"),
                "model_output": row.get("model_output"),
                "interaction_target": row.get("interaction_target"),
            }
        )

    system_prompt = """
You are an expert evaluator for a mobile GUI agent task.

You will be given:
1) the user's original instruction,
2) the agent's execution trace (actions/response/done/info),
3) a final sequence of screenshots (PNG images).

Your job is to judge whether the task has been successfully completed.

【Evaluation rules】
1. "success" means the final task goal is achieved, not whether intermediate actions exist.
2. Screenshots are important evidence. If screenshot evidence is insufficient, judge failure.
3. "actions" are only supporting evidence; you must not label success based on actions alone.
4. You may accept semantically equivalent actions as correct, but evidence must still support success.
5. If the provided evidence does not clearly support achievement, judge as not completed.

【Output requirement】
Return ONLY strict JSON (no Markdown, no extra text) in exactly this schema:
{"success": true/false, "reason": "one short sentence explaining the key evidence"}
""".strip()

    user_prompt_text = "【Input】\n"
    user_prompt_text += f"task_id:\n{task_id}\n\n"
    user_prompt_text += "instruction:\n" + instruction + "\n\n"
    user_prompt_text += "actions_in_time_order:\n" + str(trajectory_excerpt) + "\n\n"
    user_prompt_text += "success_state (final goal):\n" + instruction + "\n"
    user_prompt_text += "\nJudge success based on trace evidence and screenshots."

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    for ss in screenshots:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(ss).decode('utf-8')}"},
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return False, "empty_judge_output"

    content_json = _extract_json_obj(content)
    try:
        obj = json.loads(content_json)
    except json.JSONDecodeError:
        try:
            obj = json_repair.loads(content_json)
        except Exception:
            return False, f"judge_parse_error: {content[:200]}"

    success = bool(obj.get("success", False))
    reason = str(obj.get("reason", "")).strip()
    return success, reason


def evaluate_gui_trajectory_sync(
    *,
    instruction: str,
    trace_path: Path,
    model: str,
    api_key: str,
    api_base: str | None,
    task_id: str,
    output_path: Path | None,
    subtask_index: int = 1,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Missing evaluation api_key")

    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
    traj_rows = filter_step_rows(load_traj_rows(trace_path, subtask_index=subtask_index))
    screenshots = load_screenshots_for_judge(trace_path, traj_rows)
    success, reason = judge_success(
        client=client,
        instruction=instruction,
        traj_rows=traj_rows,
        screenshots=screenshots,
        model=model,
        task_id=task_id,
    )

    result = {
        "task_id": task_id,
        "instruction": instruction,
        "trace_path": str(trace_path),
        "judge_model": model,
        "success": success,
        "reason": reason,
        "steps": len(traj_rows),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_path"] = str(output_path)
    return result


async def evaluate_gui_trajectory(
    *,
    instruction: str,
    trace_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: str,
    api_base: str | None = DEFAULT_API_BASE,
    task_id: str = "gui-task",
    output_path: Path | None = None,
    subtask_index: int = 1,
) -> dict[str, Any]:
    """Evaluate one GUI run and optionally persist a JSON artifact."""
    return await asyncio.to_thread(
        evaluate_gui_trajectory_sync,
        instruction=instruction,
        trace_path=trace_path,
        model=model,
        api_key=api_key,
        api_base=api_base,
        task_id=task_id,
        output_path=output_path,
        subtask_index=subtask_index,
    )
