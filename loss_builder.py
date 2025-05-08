import torch.nn as nn

def build_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "hinge":
        # PyTorch doesn't have direct hinge loss, so use MultiLabelMarginLoss as a proxy
        return nn.MultiLabelMarginLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
