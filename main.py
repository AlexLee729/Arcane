from model import GPT, GPTConfig
import torch
from torch.nn import functional as F
import os
import tiktoken
from hellaswag import iterate_examples, render_example

device = "cpu"
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

model = GPT(GPTConfig(vocab_size=100288, n_layer=24, n_head=16, n_embd=2048))
model.to(dtype=torch.bfloat16)
model.to(device)

log_dir = "log"
checkpoint_path = os.path.join(log_dir, "arcaneGPT.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])

result = model.generate("Hello, I'm a language model,", max_length=50, num_return_sequences=1, device=device)
decoded_output = tiktoken.get_encoding("gpt2").decode(result[0].tolist())
print("Result: ", decoded_output)

# @torch.no_grad()
# def get_most_likely_row(tokens, mask, logits):
#     # evaluate the autoregressive loss at all positions
#     shift_logits = (logits[..., :-1, :]).contiguous()
#     shift_tokens = (tokens[..., 1:]).contiguous()
#     flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#     flat_shift_tokens = shift_tokens.view(-1)
#     shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
#     shift_losses = shift_losses.view(tokens.size(0), -1)
#     # now get the average loss just for the completion region (where mask == 1), in each row
#     shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
#     masked_shift_losses = shift_losses * shift_mask
#     # sum and divide by the number of 1s in the mask
#     sum_loss = masked_shift_losses.sum(dim=1)
#     avg_loss = sum_loss / shift_mask.sum(dim=1)
#     # now we have a loss for each of the 4 completions
#     # the one with the lowest loss should be the most likely
#     pred_norm = avg_loss.argmin().item()
#     return pred_norm

# num_correct_norm = 0
# num_total = 0
# for i, example in enumerate(iterate_examples("val")):
#     # render the example into tokens and labels
#     _, tokens, mask, label = render_example(example)
#     tokens = tokens.to(device)
#     mask = mask.to(device)
#     # get the logits
#     with torch.no_grad():
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits, loss = model(tokens)
#         pred_norm = get_most_likely_row(tokens, mask, logits)
#     num_total += 1
#     num_correct_norm += int(pred_norm == label)
#     acc_norm = num_correct_norm / num_total
#     print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")