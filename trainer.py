import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from build.optimizer_builder import build_optimizer
from build.loss_builder import build_loss_function
from build.scheduler_builder import build_scheduler
from dna.dna import EvoDNA

# 🔧 Simple feedforward model for MNIST
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(dna: EvoDNA, device="cpu"):
    # 🧬 Extract DNA traits
    loss_name = dna.loss_function

    # 📦 Load MNIST
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # 🧠 Build components
    model = MNISTModel().to(device)
    optimizer = build_optimizer(dna, model.parameters())
    loss_fn = build_loss_function(loss_name)
    scheduler = build_scheduler(dna.lr_schedule, optimizer)

    # 🚂 Train
    model.train()
    for epoch in range(2):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 🔄 Convert targets if needed
            if loss_name == "mse":
                y_batch = F.one_hot(y_batch, num_classes=10).float()
            elif loss_name == "hinge":
                y_batch = F.pad(y_batch.unsqueeze(1), (0, 9), value=-1)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

    # 🧪 Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy
