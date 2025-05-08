import random

class EvoDNA:
    def __init__(self):
        self.architecture = random.choice(["transformer_small", "transformer_tiny"])
        self.optimizer = {
            "type": random.choice(["adam", "sgd", "adamw"]),
            "lr": round(random.uniform(0.0001, 0.01), 5),
            "momentum": round(random.uniform(0.0, 0.99), 2),
        }
        self.loss_function = random.choice(["cross_entropy", "mse", "hinge"])
        self.lr_schedule = random.choice(["linear", "cosine", "constant"])

    def mutate(self):
        # Mutate learning rate slightly
        delta = random.uniform(-0.001, 0.001)
        self.optimizer["lr"] = max(1e-5, min(0.1, self.optimizer["lr"] + delta))
        # Randomly change components
        if random.random() < 0.3:
            self.optimizer["type"] = random.choice(["adam", "sgd", "adamw"])
        if random.random() < 0.3:
            self.loss_function = random.choice(["cross_entropy", "mse", "hinge"])
        if random.random() < 0.3:
            self.lr_schedule = random.choice(["linear", "cosine", "constant"])

    def __str__(self):
        return (f"DNA(arch={self.architecture}, opt={self.optimizer}, "
                f"loss={self.loss_function}, schedule={self.lr_schedule})")
