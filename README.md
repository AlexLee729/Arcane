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
- [Model Comparison](#model-comparison)

## Overview
Arcane is based on the GPT-2 architecture and is built to help users learn how transformer models work by experimenting with a simplified model. The project includes support for **LoRA**, a fine-tuning technique that uses low-rank matrices to efficiently modify pre-trained models.

### Key Concepts:
- **Transformer Blocks**: Multiple transformer layers with self-attention mechanisms.
- **Multi-Head Self-Attention (MHSA)**: Implements attention across multiple heads.
- **Rotary Positional Embeddings (RoPE)**: Improves performance by adding position-dependent transformations to queries and keys.
- **LoRA (Low-Rank Adaptation)**: Reduces the number of trainable parameters for efficient fine-tuning.

## Model Architecture
Arcane is implemented in Python using **PyTorch**. The model contains the following key components:

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

## Model Comparison on HellaSwag Accuracy

| Model                 | Data Size   | HellaSwag Accuracy | RoPE |
|-----------------------|-------------|--------------------|------|
| GPT-2 124M            | 100B        | 0.2955             | No   |
| GPT-3 124M            | 300B        | 0.3357             | No   |
| Arcane 124M           | 10B         | 0.3036             | No   |
| Arcane 124M           | 10B         | 0.3074             | Yes  |

