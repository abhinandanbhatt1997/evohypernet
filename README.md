# EvoHyperNet - Evolutionary Neural Architecture Search

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

EvoHyperNet is an evolutionary algorithm framework for hyperparameter optimization and neural architecture search, implemented in PyTorch. It automatically discovers optimal model architectures and training configurations through genetic algorithms.

## Key Features

- 🧬 **DNA-based evolution**: Encodes network architectures and training parameters
- 🏗️ **Modular design**: Easily extendable components for models, optimizers, and losses
- 📊 **Visual tracking**: Generation-by-generation performance visualization
- ⚡ **GPU acceleration**: Full CUDA support for faster evolution
- 🛑 **Early stopping**: Smart termination when improvements plateau

## Installation

```bash
git clone https://github.com/yourusername/evohypernet.git
cd evohypernet
pip install -r requirements.txt
