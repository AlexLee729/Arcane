import torch
import os
import tiktoken
from model import GPT
from tqdm import tqdm

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

# Main training loop
def training_loop(text, gpt_model_path, config):
    model = GPT(config).to(config.device)
    load_pretrained_model(model, gpt_model_path)
    train_data, val_data = data_split(text)
    
    best_val_loss = evaluate_model(model, val_data, config)
    
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate, weight_decay=1e-1)
    
    warmup_steps = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))
    
    for iter in tqdm(range(config.max_iters)):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            train_loss = evaluate_model(model, train_data, config)
            val_loss = evaluate_model(model, val_data, config)

            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), gpt_model_path)
                best_val_loss = val_loss

        xb, yb = get_batch(train_data, config)
        _, loss = model(xb, yb)

        loss.backward()
        # Clip gradients to prevent them from becoming too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)