import torch
import time
import tiktoken
import os
from dataloader import JSONDataLoader
from model import GPT, GPTConfig
from peft import LoraConfig, get_peft_model
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
        
data_dir = "./Data"  # Path to your data directory
# Hyperparameters
lr = 2e-5
max_steps = 314 * 10
block_size = 1024
batch_size = 8

ddp = int(os.environ.get('RANK', -1)) != -1 

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

device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 2 ** 20  # ~32k tokens
B, T = 8, 1024                # Micro-batch size and sequence length
grad_accum_steps = total_batch_size // (B * T)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
# Load GPT-2 tokenizer
enc = tiktoken.get_encoding('gpt2')

train_loader = JSONDataLoader(data_dir="Data", B=B, split="train")
val_loader = JSONDataLoader(data_dir="Data", B=B, split="val")

# Model configuration and initialization
torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig(vocab_size=50304, n_layer=24, n_head=16, n_embd=2048))

log_dir = "log"
checkpoint_path = os.path.join(log_dir, "ArcaneGPT.pt")

if os.path.exists(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
  
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    target_modules=["c_attn", "c_proj", "c_fc"],  
    lora_dropout=0.1,
    bias="lora_only"
)

model = get_peft_model(model, lora_config)
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module if ddp else model

# Optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=lr)

# Count trainable parameters in LoRA layers
trainable_params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Number of trainable parameters (LoRA layers): {trainable_params_count}")

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log1_3B_ft.txt")

for step in range(max_steps):
    t0 = time.time()
    
    # Validation check every 5 steps or at the last step
    if step % 200 == 0 or step == max_steps - 1:
        model.eval()
        val_loss_accum = 0.0
        val_loss_steps = 200
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y, attention_mask = val_loader.next_batch()
                x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y, attention_mask)
                val_loss_accum += loss / val_loss_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss = val_loss_accum.item()
            with open(log_file, "a") as f:
                f.write(f"step: {step} | val: {val_loss:.4f}\n")

    if step % 250 == 0 or step == max_steps - 1:
        raw_model.merge_and_unload()
        checkpoint_data = {
            'model': raw_model.state_dict(),  # Save the model weights
            'step': step,
            'val_loss': val_loss_accum.item(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint_data, os.path.join(log_dir, f"arcaneGPT_latest_checkpoint.pt"))
        if step % 2500 == 0 or step == max_steps - 1:
            torch.save({'model': raw_model.state_dict()}, os.path.join(log_dir, f"ArcaneGPT_ft.pt"))
    
    # Training with gradient accumulation
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y, attention_mask = train_loader.next_batch()
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, attention_mask)
        loss = loss / grad_accum_steps
        loss_accum += loss
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Logging
    t1 = time.time()
    if master_process:
        with open(log_file, "a") as f:
            f.write(f"Step {step:5d} | Training loss: {loss_accum:.6f} | Time: {t1 - t0:.2f}s\n")

if ddp:
    destroy_process_group()