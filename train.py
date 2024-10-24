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

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Batch parameters
total_batch_size = 2**19  # ~0.5M tokens
B = 16  # Micro batch size
T = 1024  # Sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Data loaders
train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

# Model setup
model = GPT(GPTConfig(vocab_size=50304), use_lora=False)
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model 

# Learning rate scheduler
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it >= max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)

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
        val_loss_steps = 20
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss / val_loss_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
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
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Set learning rate and update optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(step)
    optimizer.step()

    torch.cuda.synchronize()

    # Logging
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}\n")

if ddp:
    destroy_process_group()