import torch.optim as optim

def build_optimizer(dna, model_parameters):
    opt_type = dna.optimizer["type"]
    lr = dna.optimizer["lr"]
    momentum = dna.optimizer.get("momentum", 0.0)

    if opt_type == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif opt_type == "sgd":
        return optim.SGD(model_parameters, lr=lr, momentum=momentum)
    elif opt_type == "adamw":
        return optim.AdamW(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
