"""Optional validation-driven learning-rate scheduling for resumable training."""
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_lr_scheduler(optimizer, training_params):
    config = training_params.get("lr_scheduler", {})
    if not config or not config.get("enabled", False):
        return None
    if config.get("type", "plateau") != "plateau":
        raise ValueError("Only validation-driven plateau scheduling is supported")
    return ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config.get("factor", .5)),
        patience=int(config.get("patience", 8)),
        threshold=float(config.get("threshold", .005)), threshold_mode="rel",
        cooldown=int(config.get("cooldown", 2)),
        min_lr=float(config.get("min_lr", 1e-6)),
    )


def restore_lr_scheduler(scheduler, checkpoint):
    if scheduler is None:
        if checkpoint.get("lr_scheduler_state_dict") is not None:
            raise ValueError("Checkpoint used an LR scheduler; enable it to resume consistently")
        return
    saved = checkpoint.get("lr_scheduler_state_dict")
    if saved is not None:
        scheduler.load_state_dict(saved)
    else:
        # Upgrade a constant-LR run without replaying historical plateau steps.
        scheduler.best = float(checkpoint.get("best_score", checkpoint.get("val_loss", float("inf"))))
