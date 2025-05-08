import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(scheduler_name, optimizer):
    if scheduler_name == "linear":
        return lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == "constant":
        return None  # no scheduler
    else:
        raise ValueError(f"Unknown LR scheduler: {scheduler_name}")
