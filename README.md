# Arcane: A GPT-2 Inspired Model

### Multimodal Support
Arcane is a refined, educational GPT-style transformer model built in PyTorch. Inspired by GPT-2, it offers an accessible yet advanced implementation designed to illustrate modern techniques in transformer-based language modeling. Key enhancements include efficient activation checkpointing, rotary positional embeddings (RoPE), a custom RMS normalization layer with recomputation, and a modular design that facilitates further innovations such as low-rank adaptation (LoRA).

## Features
- **Compact Transformer Architecture**: A GPT-2–inspired model with a clear, modular structure.
- **Efficient Normalization**: Uses an RMS normalization layer with activation checkpointing to reduce memory usage during training.
- **Rotary Positional Embeddings (RoPE)**: Applies RoPE to enhance the positional encoding within self-attention.
- **Memory-Efficient Self-Attention**: Implements multi-head causal self-attention with key/value caching and activation checkpointing.
- **Advanced MLP Design**: Features a custom three-part feed-forward network that leverages SiLU activation.
- **Optimized Weight Initialization**: Employs improved initialization schemes for both linear and embedding layers.
- **Custom Optimizer Configuration**: Uses AdamW with distinct parameter groups for weight decay management.
- **Robust Text Generation**: Supports top‑k and nucleus (top‑p) sampling methods for generating coherent sequences.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Model Comparison](#model-comparison)

## Overview
Arcane is based on the GPT-2 architecture and is built to help users learn how transformer models work by experimenting with a simplified model. The project includes support for **LoRA**, a fine-tuning technique that uses low-rank matrices to efficiently modify pre-trained models.

### Key Concepts
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
- Python 3.10+
- PyTorch 2.0+
- `tiktoken` tokenizer library

Install dependencies:
```bash
pip install torch tiktoken
```

## Model Training Performance
The training and validation loss curves for the 3 models over training steps are shown below, the model was trained on 10B tokens which equates to 19073 steps for the 124M and 355M model and 9536 steps for the 1.3B model due to the increase in batch size.:
![Loss Graph](/Images/Train_Val_graph.png)
| Model                 | Training Loss   | Validation Loss |
|-----------------------|-----------------|-----------------|
| Arcane 124M           | 3.07            | 3.00            |
| Arcane 355M           | 2.89            | 2.88            | 
| Arcane 1.3B           | 2.88            | 2.87            | 

## Model Comparison on HellaSwag Accuracy

| Model                 | Data Size   | HellaSwag Accuracy | RoPE |
|-----------------------|-------------|--------------------|------|
| GPT-2 124M            | 100B        | 0.2955             | No   |
| GPT-3 124M            | 300B        | 0.3357             | No   |
| Arcane 124M           | 10B         | 0.3036             | No   |
| Arcane 124M           | 10B         | 0.3083             | Yes  |
| Arcane 355M           | 10B         | 0.3266             | Yes  |
| Arcane 1.3B           | 10B         | 0.3414             | Yes  |
| Arcane 1.3B           | 20B         | 0.3881             | Yes  |

## Future Implementations

I plan to expand Arcane with the following features:

### Reinforcement Learning with Human Feedback (RLHF)
Integrate RLHF to improve the model's performance by leveraging human feedback during training. This will help in refining the model's responses and making it more aligned with human expectations.

### Proximal Policy Optimization (PPO)
Implement PPO, a popular reinforcement learning algorithm, to optimize the policy of the model. This will enhance the model's ability to make decisions and generate more accurate and contextually appropriate responses.
