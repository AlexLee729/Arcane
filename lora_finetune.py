import os
import time
import torch
import copy
import numpy as np
from dataloader import DataLoader
from model import GPT, GPTConfig
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP."
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

is_master_node = os.environ.get("SLURM_NODEID", "0") == "0"
master_process = (ddp and ddp_rank == 0 and is_master_node) or (not ddp)
device_type = "cuda" if device.startswith("cuda") else "cpu"

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
lr = 3e-4
max_steps = 19073 * 2
total_batch_size = 2**19  # 524,288 tokens
B, T = 8, 2048
tokens_per_micro_batch = B * T
grad_accum_steps = total_batch_size // (ddp_world_size * tokens_per_micro_batch)
assert grad_accum_steps > 0, "Batch size too small or too many GPUs"

if master_process:
    print(f"Total desired batch size: {total_batch_size} tokens")
    print(f"=> Gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

# -----------------------------------------------------------------------------
# Model Setup
# -----------------------------------------------------------------------------
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304, n_layer=24, n_head=16, n_embd=1024, block_size=T))

log_dir = "log"
checkpoint_path = os.path.join(log_dir, "arcane3.pt")

if os.path.exists(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
  
lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    target_modules=["c_attn", "c_proj", "c_fc", "c_fc2"],  
    lora_dropout=0.05,
    bias="lora_only"
)

model = get_peft_model(model, lora_config)
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

optimizer = raw_model.configure_optimizers(weight_decay=0.01, learning_rate=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=max_steps, anneal_strategy='cos', pct_start=0.04
)

# Load checkpoint if available
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log_ft.txt")

if master_process:
    with open(log_file, "w") as f:
        f.write("")

if master_process:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA): {trainable_params}")

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()

    # ----- Validation -----
    if step % 250 == 0 or step == max_steps - 1:
        model.eval()
        val_loss_accum = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for _ in range(250):
                x, y = val_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss / 250

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            val_loss = val_loss_accum.item()
            with open(log_file, "a") as f:
                f.write(f"Step {step:5d} | Validation Loss: {val_loss:.4f}\n")

    # ----- Checkpointing -----
    if step % 250 == 0 or step == max_steps - 1:
        checkpoint = {
            'model': raw_model.state_dict(),
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'current_shard': train_loader.current_shard,
            'current_position': train_loader.current_position
        }
        if master_process:
            torch.save(checkpoint, "/lustre/uschill-lab/users/3931/arcaneGPT_latest_checkpoint.pt")
            if (step % 5000 == 0 and step > 0) or step == max_steps - 1:
                model2 = copy.deepcopy(raw_model)
                model2 = model2.merge_and_unload()
                torch.save({'model': model2.state_dict()}, f"/lustre/uschill-lab/users/3931/ArcaneGPT_ft_step_{step}.pt")

    # ----- Final Model Save -----
    if step == max_steps - 1:
        merged_model = raw_model.merge_and_unload()
        if master_process:
            torch.save({'model': merged_model.state_dict()}, "/lustre/uschill-lab/users/3931/ArcaneGPT_ft.pt")

    # ----- Training -----
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

    if master_process:
        t1 = time.time()
        with open(log_file, "a") as f:
            f.write(f"Step {step:5d} | Training Loss: {loss_accum.item():.6f} | Time: {t1 - t0:.2f}s\n")

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
if ddp:
    dist.destroy_process_group()