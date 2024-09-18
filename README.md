# Arcane

Arcane is a simple GPT model developed as a learning project. The model is currently set up to run on M-Series Macbook. The ultimate goal of this GPT model is to work on it to a point where it is able to write essays for you and have direct access to websites like google docs or coding sites like visual studio code to help you program. 

## Installation

```
pip install torch numpy tiktoken
```
Dependencies:

- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
-  `tiktoken` for OpenAI's fast BPE code

## Evaluation
The model was evaluated using the hellaswag dataset. For reference, gpt-2/gpt-3 was trained on 300B tokens while Arcane was trained on 10B tokens
Arcane 124M without RoPE had a hellaswag accuracy of 0.3106
Arcane 124M with RoPE had a hellaswag accuracy of 0.3074
Gpt-2 124M had a hellaswag accuracy of 0.2955
GPT-3 124M had a hellaswag accuracy of 0.3357
