import time
import os
from dataloader import DataLoader
import torch
from model import GPT, GPTConfig

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Hyperparameters and Scheduler
lr = 1e-3 # 355M
max_steps = 19073 * 2

# Determine if this is a DDP run (DDP runs will have RANK set by torchrun)
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # DDP requires CUDA; we set the device based on the local rank
    assert torch.cuda.is_available(), "CUDA is required for DDP."
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    # For non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# Added after video: setting device_type for autocast
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Determine if this process is on the master node.
# In SLURM, SLURM_NODEID is typically set; we assume node "0" is the master node.
is_master_node = os.environ.get("SLURM_NODEID", "0") == "0"
# The master process is the one with ddp_rank 0 on the master node.
master_process = (ddp and ddp_rank == 0 and is_master_node) or (not ddp)

# Batch parameters
total_batch_size = 2**19  # ~0.5M tokens
B = 16  # Micro batch size
T = 1024  # Sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Data loaders
train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

# Model setup
model = GPT(GPTConfig(vocab_size=50304)) # 355M model
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  

optimizer = raw_model.configure_optimizers(weight_decay=1e-1, learning_rate=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=max_steps, anneal_strategy='cos', pct_start=0.04
)

# Create log directory and log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log2.txt")
if master_process:
    with open(log_file, "a") as f:
        f.write("")

# Load checkpoint if available
start_step = 0
checkpoint_path = os.path.join("/lustre/uschill-lab/users/3931", "latest_checkpoint_2.pt")
append_mode = False
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_step = checkpoint['step']
    train_loader.current_shard = checkpoint['current_shard']
    train_loader.current_position = checkpoint['current_position']
    if master_process:
        print(f"Resuming training from step {start_step}")
    append_mode = True
    
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
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss / val_loss_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss = val_loss_accum.item()
            with open(log_file, "a") as f:
                f.write(f"step: {step} | val: {val_loss:.4f}\n")

        # Save checkpoint
        checkpoint = {
            'model': raw_model.state_dict(),
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'current_shard': train_loader.current_shard,
            'current_position': train_loader.current_position
        }
        torch.save(checkpoint, os.path.join("/lustre/uschill-lab/users/3931", "latest_checkpoint_2.pt"))
        if step % 5000 == 0 or last_step:
            torch.save({'model': raw_model.state_dict()}, os.path.join("/lustre/uschill-lab/users/3931", f"arcane4_{step}.pt"))

    # Training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    # Logging
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}\n")

if ddp:
    destroy_process_group()
