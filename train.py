import time
import math
import tiktoken
import os
from dataloader import DataLoader
import torch
from model import GPT, GPTConfig

# Learning rate schedule parameters
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

# autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Tokenizer
enc = tiktoken.get_encoding('gpt2')

# Batch parameters
total_batch_size = 2**19  # ~0.5M, in number of tokens
B = 4  # Micro batch size
T = 1024  # Sequence length
assert total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

# Data loaders
train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

# Model setup
torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# Learning rate scheduler
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4 * 3)

# Create Log Directory
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log2.txt")

# Load the checkpoint if it exists
start_step = 0
checkpoint_path = os.path.join(log_dir, "latest_checkpoint2.pt")
append_mode = False
if os.path.exists(checkpoint_path):
    checkpoint = torch.load("log/latest_checkpoint2.pt", map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step']
    train_loader.current_shard = checkpoint['current_shard']
    train_loader.current_position = checkpoint['current_position']
    print(f"Resuming training from step {start_step}, shard: {checkpoint['current_shard']}")
    append_mode = True

# Logging setup
with open(log_file, "a" if append_mode else "w") as f:
    if not append_mode:
        pass

# Training loop
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # Validation
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"step: {step} | val: {val_loss_accum.item():.4f}\n")
                
    # Checkpointing
    if step > 0 and (step % 250 == 0 or last_step):
        # optionally write model checkpoints
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
            'current_shard': train_loader.current_shard,
            'current_position': train_loader.current_position
        }
        torch.save(checkpoint, os.path.join(log_dir, "latest_checkpoint2.pt"))
        if step % 5000 == 0 or last_step:
            torch.save({'model': model.state_dict()}, os.path.join(log_dir, f"arcane2_{step}.pt"))
            
    # Training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    
    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y  = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Set learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    torch.mps.synchronize() # wait for GPU to finish
    
    t1 = time.time()
    dt = t1 - t0
    dt = dt / 60
    print(f"step:{step:5d} | train: {loss_accum.item():.6f} | dt: {dt:.2f}mins")
    with open(log_file, "a") as f:
        f.write(f"step:{step:5d} | train: {loss_accum.item():.6f}\n")