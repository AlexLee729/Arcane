from model import GPT, GPTConfig
import torch
from torch.nn import functional as F
import os

device = "cpu"
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

log_dir = "log"
checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

model.generate("Google,", max_length=100, num_return_sequences=1, device=device)
