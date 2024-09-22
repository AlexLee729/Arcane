# Arcane: A Minimal GPT-2 Inspired Model with LoRA

This repository contains **Arcane**, a small, educational GPT-like transformer model inspired by GPT-2. The primary goal of this project is to provide an easy-to-understand implementation of GPT and explore advanced concepts such as **LoRA (Low-Rank Adaptation)** for fine-tuning. This project is designed to help researchers and engineers grasp the inner workings of GPT models through hands-on coding and experimentation.

## Features
- **GPT Architecture**: Implements a small GPT-2 style transformer model.
- **LoRA Support**: Integrates LoRA layers for efficient fine-tuning, allowing experimentation with lower-rank projections.
- **Rotary Positional Embeddings (RoPE)**: Uses RoPE for positional encoding in self-attention.
- **Optimized Training**: Gradient accumulation, learning rate scheduling, and mixed-precision support for efficient training.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Validation](#validation)
  - [Text Generation](#text-generation)
- [LoRA Integration](#lora-integration)
- [Learning Rate Schedule](#learning-rate-schedule)
- [Checkpointing and Resuming Training](#checkpointing-and-resuming-training)
- [Acknowledgments](#acknowledgments)

## Overview
Arcane is based on the GPT-2 architecture and is built to help users learn how transformer models work by experimenting with a simplified model. The project includes support for **LoRA**, a fine-tuning technique that uses low-rank matrices to efficiently modify pre-trained models.

### Key Concepts:
- **Transformer Blocks**: Multiple transformer layers with self-attention mechanisms.
- **Multi-Head Self-Attention (MHSA)**: Implements attention across multiple heads.
- **Rotary Positional Embeddings (RoPE)**: Improves performance by adding position-dependent transformations to queries and keys.
- **LoRA (Low-Rank Adaptation)**: Reduces the number of trainable parameters for efficient fine-tuning.

## Model Architecture
MiniGPT is implemented in Python using **PyTorch**. The model contains the following key components:

1. **Embedding Layer**:
   - Token embedding (`wte`): Maps vocabulary tokens to a dense representation.
   - Positional embedding (`wpe`): Adds position information to the input tokens.
   
2. **Transformer Blocks**: Each block contains:
   - **Causal Self-Attention**: Computes self-attention with optional LoRA layers.
   - **MLP Layer**: A feed-forward network to project attention outputs.
   - **Layer Normalization**: Applied before the self-attention and MLP layers.
   
3. **LoRA Layers**: Optional low-rank layers can be added to modify query and key projections for fine-tuning.
   
4. **Language Modeling Head**: A linear layer maps the final transformer output back to token logits.

## Requirements
To run this project, you need:
- Python 3.8+
- PyTorch
- `tiktoken` tokenizer library

Install dependencies:
```bash
pip install torch tiktoken
```

## Evaluation
The model was evaluated using the hellaswag dataset. For reference, gpt-2/gpt-3 was trained on 300B tokens while Arcane was trained on 10B tokens
Arcane 124M without RoPE had a hellaswag accuracy of 0.3106
Arcane 124M with RoPE had a hellaswag accuracy of 0.3074
Gpt-2 124M had a hellaswag accuracy of 0.2955
GPT-3 124M had a hellaswag accuracy of 0.3357
