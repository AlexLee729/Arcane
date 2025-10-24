import time
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from Arcane.dataloader import DataLoader
from Arcane.gpt import GPT, GPTConfig

def setup_distributed():
    """Initialize distributed training environment."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
    return ddp, rank, local_rank, world_size, device

def initialize_model_and_optimizer(gpt_config, config, device, ddp, local_rank):
    """Set up model and optimizer for training."""
    model = GPT(gpt_config).to(device)
    model = torch.compile(model, dynamic=False)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model
    optimizer = raw_model.setup_optimizers(weight_decay=1e-1, learning_rate=config.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.max_steps, 
        anneal_strategy='cos', pct_start=0.04
    )
    return model, raw_model, optimizer, scheduler

def load_checkpoint(raw_model, checkpoint_path, device, master_process):
    """Load model checkpoint if available."""
    start_step = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(checkpoint['model'])
        if master_process:
            print(f"Loaded checkpoint from {checkpoint_path}")
    return start_step

def save_checkpoint(raw_model, optimizer, scheduler, train_loader, step, val_loss, log_dir, max_steps):
    """Save training checkpoint."""
    checkpoint = {
        'model': raw_model.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'current_shard': train_loader.current_shard,
        'current_position': train_loader.current_position
    }
    torch.save(checkpoint, os.path.join(log_dir, "latest_checkpoint.pt"))
    if step % 5000 == 0 or step == max_steps - 1:
        torch.save({'model': raw_model.state_dict()}, os.path.join(log_dir, f"arcane_{step}.pt"))

def validate_model(model, val_loader, device, device_type, steps=250):
    """Perform validation and compute average loss."""
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = model(x, y)
            val_loss_accum += loss / steps
    return val_loss_accum

def main():
    # Configuration
    config = type('Config', (), {
        'lr': 3e-4 * 3,
        'max_steps': 19073,
        'total_batch_size': 2**19,
        'B': 1,
        'T': 1024
    })()

    # Setup distributed training
    ddp, ddp_rank, local_rank, world_size, device = setup_distributed()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    is_master_node = os.environ.get("SLURM_NODEID", "0") == "0"
    master_process = (ddp and ddp_rank == 0 and is_master_node) or (not ddp)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Validate batch configuration
    assert config.total_batch_size % (config.B * config.T * world_size) == 0
    grad_accum_steps = config.total_batch_size // (config.B * config.T * world_size)
    if master_process:
        print(f"Total batch size: {config.total_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Initialize data loaders
    train_loader = DataLoader(B=config.B, T=config.T, split="train")
    val_loader = DataLoader(B=config.B, T=config.T, split="val")

    # Initialize model
    gpt_config = GPTConfig(n_layer=24,
    n_head=16,
    n_kv_head=4,
    n_embd=1216,
    sequence_len=1024,
    vocab_size=50304)
    model, raw_model, optimizer, scheduler = initialize_model_and_optimizer(
        gpt_config, config, device, ddp, local_rank
    )

    # Setup logging
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log_ft.txt")
    if master_process:
        open(log_file, "a").close()  # Ensure log file exists

    # Load checkpoint
    checkpoint_path = os.path.join(log_dir, "arcane4_19072.pt")
    start_step = load_checkpoint(raw_model, checkpoint_path, device, master_process)

    # Training loop
    for step in range(start_step, config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1)

        # Validation phase
        if step % 250 == 0 or last_step:
            val_loss_accum = validate_model(model, val_loader, device, device_type)
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                with open(log_file, "a") as f:
                    f.write(f"step: {step} | val: {val_loss_accum.item():.4f}\n")
                save_checkpoint(
                    raw_model, optimizer, scheduler, train_loader, step, 
                    val_loss_accum.item(), log_dir, config.max_steps
                )

        # Training phase
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
                loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if master_process:
            dt = time.time() - t0
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * world_size
            tokens_per_sec = tokens_processed / dt
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}\n")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()