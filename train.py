import time
import math
import tiktoken
import os
from dataloader import DataLoader
import torch
from model import GPT, GPTConfig

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Hyperparameters and Scheduler
lr = 6e-4 * 3
warmup_steps = 715
max_steps = 19073

total_batch_size = 2**19    # ~0.5M tokens
B = 2                       # Micro batch size
T = 1024                    # Sequence length
assert total_batch_size % (B * T) == 0, "Batch size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size} tokens")
print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

# DataLoader Setup
train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

# Model setup
model = GPT(GPTConfig(vocab_size=50304)) # 200064 is the size of the tokenizer vocabulary
model.to(device)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps)

# Create log directory
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Load checkpoint if available
start_step = 0
checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
append_mode = False
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step']
    train_loader.current_shard = checkpoint['current_shard']
    train_loader.current_position = checkpoint['current_position']
    print(f"Resuming training from step {start_step}, shard: {checkpoint['current_shard']}")
    append_mode = True

# Logging setup
with open(log_file, "a" if append_mode else "w"):
    pass

# Training loop
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 250
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss / val_loss_steps

        val_loss = val_loss_accum.item()
        print(f"Validation loss: {val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"step: {step} | val: {val_loss:.4f}\n")

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
            'current_shard': train_loader.current_shard,
            'current_position': train_loader.current_position
        }
        torch.save(checkpoint, checkpoint_path)
        if step % 5000 == 0 or last_step:
            torch.save({'model': model.state_dict()}, os.path.join(log_dir, f"arcane_{step}.pt"))

    # Training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    # Logging
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}\n")