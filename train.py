import torch, os, tiktoken
from config import *
from model import GPTLanguageModel
from tqdm import tqdm

def load_pretrained_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # Input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Target sequences
    return x.to(device), y.to(device)

def data_split(text):
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

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

def training_loop(text):
    model = GPTLanguageModel().to(device)
    load_pretrained_model(model, gpt_model_path)
    print(f"{model.num_parameters()} M parameters")

    train_data, val_data = data_split(text)
    prev_val_loss = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    val_losses = []

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            train_loss = evaluate_model(model, train_data, block_size, batch_size, eval_iters)
            val_loss = evaluate_model(model, val_data, block_size, batch_size, eval_iters)
            val_losses.append(val_loss)
            print(f'step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')

            if val_loss < prev_val_loss:
                torch.save(model.state_dict(), gpt_model_path)
                prev_val_loss = val_loss

            # Early stopping: stop training if validation loss is increasing
            if len(val_losses) > 3 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
                print("Validation loss is increasing. Stopping training.")
                break
        
        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients to prevent them from becoming too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # Final evaluation & model saving
    train_loss = evaluate_model(model, train_data, block_size, batch_size, eval_iters)
    val_loss = evaluate_model(model, val_data, block_size, batch_size, eval_iters)
    print(f'step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
    if val_loss < prev_val_loss:
        torch.save(model.state_dict(), gpt_model_path)