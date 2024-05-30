import torch
import os
import tiktoken
from model import GPT
from tqdm import tqdm
import math

max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 100
gradient_accumulation_steps = 2

eval_interval = 1000
learning_rate = 6e-4
weight_decay = 1e-1
min_lr = 6e-5
eval_iters = 200

# Function to load a pre-trained model
def load_pretrained_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")

# Function to get a batch of data
def get_batch(data, config):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])  # Input sequences
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])  # Target sequences
    return x.to(config.device), y.to(config.device)

# Function to split data into training and validation sets
def data_split(text):
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

# Function to evaluate the model
@torch.no_grad()
def evaluate_model(model, data, config):
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        X, Y = get_batch(data, config)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Main training loop
def training_loop(text, gpt_model_path, config):
    model = GPT(config).to(config.device)
    load_pretrained_model(model, gpt_model_path)
    train_data, val_data = data_split(text)
    
    best_val_loss = 1e9
    
    # optimizer
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)
    
    accumulated_batches = 0
    accumulated_loss = 0.0
    
    for iter in tqdm(range(config.max_iters)):
        lr = get_lr(iter, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            train_loss = evaluate_model(model, train_data, config)
            val_loss = evaluate_model(model, val_data, config)

            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), gpt_model_path)
                best_val_loss = val_loss
        
        model.train()
        
        # Gradient accumulation
        xb, yb = get_batch(train_data, config)
        _, loss = model(xb, yb)
        
        loss = loss / config.gradient_accumulation_steps  # Normalize loss by accumulation steps
        accumulated_loss += loss.item()
        
        if (iter + 1) % config.gradient_accumulation_steps == 0:
            accumulated_loss.backward()
            
            # Clip gradients to prevent them from becoming too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            accumulated_loss = 0.0  # Reset accumulated loss
        
        accumulated_batches += 1
        
        if accumulated_batches == config.gradient_accumulation_steps:
            accumulated_batches = 0
