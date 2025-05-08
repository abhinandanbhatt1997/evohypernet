evohypernet/
│
├── main.py                        # Entry point: runs EvoDNA & evolution loops
│
├── dna/                           # DNA representation and mutation logic
│   └── dna.py                     # Defines EvoDNA class and mutation strategies
│
├── evolution/                     # Evolution strategies: selection, crossover, mutation
│   └── engine.py                  # Evolution engine (e.g., tournament selection)
│
├── models/                        # Model architectures (e.g., Transformer variants)
│   └── transformer.py             # Sample small Transformer
│
├── optimizers/                    # Evolved optimizers
│   └── registry.py                # Optimizer lookup + dynamic builder
│
├── losses/                        # Custom and composable loss functions
│   └── registry.py                # Dynamic loss composer based on DNA
│
├── schedules/                     # Learning rate schedules
│   └── registry.py                # Scheduler builder from DNA spec
│
├── experiments/                   # Configs and logs for training runs
│   └── test_run.py                # Sample training script using DNA
│
└── utils/                         # Logging, metrics, helper tools
    └── logger.py                  # Logging utilities
