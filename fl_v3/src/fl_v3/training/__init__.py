"""fl_v3 training — task-agnostic interface + generic train/eval loop."""
from fl_v3.training.loop import evaluate, train_local, train_one_epoch
from fl_v3.training.tasks import (
    TASK_REGISTRY,
    ClientData,
    Criterion,
    Task,
    available_tasks,
    build_criterion,
    get_task,
    register_task,
)

__all__ = [
    "TASK_REGISTRY",
    "Task",
    "ClientData",
    "Criterion",
    "get_task",
    "register_task",
    "available_tasks",
    "build_criterion",
    "train_local",
    "train_one_epoch",
    "evaluate",
]
from .phase1 import (
    Phase1CyclicScheduler,
    build_phase1_optimizer,
    phase1_training_components,
    validate_phase1_optimizer_identity,
)

__all__ += [
    "Phase1CyclicScheduler",
    "build_phase1_optimizer",
    "phase1_training_components",
    "validate_phase1_optimizer_identity",
]
