"""Task capability and launch routes for the TUI backend."""

from fastapi import APIRouter, Depends, status

from nanobot.tui.dependencies import get_task_launch_service
from nanobot.tui.schemas import LaunchRunResponse, TaskContractResponse, TaskLaunchRequest
from nanobot.tui.services import TaskLaunchService

router = APIRouter()


@router.get("/tasks", response_model=TaskContractResponse)
def describe_tasks(
    service: TaskLaunchService = Depends(get_task_launch_service),
) -> TaskContractResponse:
    """Expose typed task-launch capability metadata."""

    return service.describe_capability()


@router.post(
    "/tasks/runs",
    response_model=LaunchRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def launch_task(
    payload: TaskLaunchRequest,
    service: TaskLaunchService = Depends(get_task_launch_service),
) -> LaunchRunResponse:
    """Launch one supported task kind asynchronously."""

    return service.launch_task(payload)
