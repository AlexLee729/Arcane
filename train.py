import torch
import os
import tiktoken
from config import *
from model import GPTLanguageModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Function to load a pre-trained model
def load_pretrained_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")

# Function to get a batch of data
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])  # Input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # Target sequences
    return x.to(device), y.to(device)

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
def evaluate_model(model, data, block_size, batch_size, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# Main training loop
def training_loop(text):
    model = GPTLanguageModel().to(device)
    load_pretrained_model(model, gpt_model_path)
    print(f"{model.num_parameters()}M parameters")
    train_data, val_data = data_split(text)
    
    if model:
        prev_val_loss = evaluate_model(model, val_data, block_size, batch_size, eval_iters)
    else:
        prev_val_loss = float('inf')

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    warmup_steps = 500
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))

    val_losses = []
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="logs")

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = evaluate_model(model, train_data, block_size, batch_size, eval_iters)
            val_loss = evaluate_model(model, val_data, block_size, batch_size, eval_iters)
            val_losses.append(val_loss)

            #print(f"Train loss: {train_loss} | Val Loss: {val_loss}")
            
            # Log metrics to TensorBoard
            writer.add_scalar("Train Loss", train_loss, iter)
            writer.add_scalar("Validation Loss", val_loss, iter)

            if val_loss < prev_val_loss:
                torch.save(model.state_dict(), gpt_model_path)
                prev_val_loss = val_loss

            # Early stopping: stop training if validation loss is increasing
            if len(val_losses) > 3 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
                print("Validation loss is increasing. Stopping training.")
                break

        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = model(xb, yb)
        del logits # delete logits since they arent used

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients to prevent them from becoming too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # Close TensorBoard writer
    writer.close()