import sys
from collections import defaultdict

import torch
import tiktoken
from tqdm import tqdm

from Arcane.gpt import GPT, GPTConfig
from Benchmarks.mmlu import MMLU
from Benchmarks.hellaswag import iterate_examples, render_example, get_most_likely_row

# ----------------------------
# Configuration & Device Setup
# ----------------------------
CONFIG = GPTConfig(n_layer=28, n_head=18, n_embd=1152)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = "models/arcane.pt"
MAX_TOKENS = 100
TEMPERATURE = 0.6
TOP_K = 50

# ----------------------------
# Model Initialization
# ----------------------------
model = GPT(CONFIG).to(DEVICE)
model.eval()

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
except FileNotFoundError:
    print(f"No checkpoint found at {CHECKPOINT_PATH}, using initialized model")

model = torch.compile(model, mode="default")
tokenizer = tiktoken.get_encoding("gpt2")

# HellaSwag Evaluation
def evaluate_hellaswag():
    num_correct = 0
    num_total = 0

    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

        with torch.no_grad(), torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits, _ = model(tokens)
            pred = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct += int(pred == label)

        print(f"\rAccuracy: {num_correct}/{num_total} = {num_correct/num_total:.4f}", end="", flush=True)

    print(f"\nHellaSwag accuracy: {num_correct}/{num_total} = {num_correct/num_total:.4f}")

# MMLU Evaluation
def evaluate_mmlu():
    task = MMLU(subset="all", split="test")
    LETTERS = ("A", "B", "C", "D")
    letter_ids = torch.tensor([tokenizer.encode(l)[0] for l in LETTERS], device=DEVICE)

    @torch.inference_mode()
    def predict_letter(prompt: str) -> str:
        """Generate one token and return the first valid A/B/C/D answer."""
        tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        for token_id in model.generate(tokens, max_tokens=1, temperature=0.0, top_k=None):
            if token_id in letter_ids:
                return LETTERS[(letter_ids == token_id).nonzero(as_tuple=True)[0].item()]
        return "A"  # fallback

    correct, total = 0, 0
    per_subject = defaultdict(lambda: [0, 0])  # [correct, total]

    for i in tqdm(range(task.num_examples()), dynamic_ncols=True):
        convo = task.get_example(i)
        user_prompt = convo["messages"][0]["content"]
        gold_answer = convo["messages"][1]["content"]
        subject = convo["subject"]

        pred = predict_letter(user_prompt)
        total += 1
        per_subject[subject][1] += 1

        if pred == gold_answer:
            correct += 1
            per_subject[subject][0] += 1

    overall_acc = 100 * correct / total
    print(f"\n=== MMLU RESULTS (test split) ===")
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")
    print("Per-Subject Accuracy:")
    for subject, (c, t) in sorted(per_subject.items()):
        print(f"{subject:35s}: {100*c/t:.2f}%")

# Chatbot Response Generation
def generate_responses(prompt: str):
    """Generate text using the model with streaming output."""
    tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    try:
        for token_id in model.generate(tokens, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
            print(tokenizer.decode([token_id]), end="", flush=True)
        print()  # newline after response
    except Exception as e:
        print(f"Error generating response: {e}")

if __name__ == "__main__":
    if "-eval" in sys.argv:
        while True:
            print("\nSelect evaluation dataset:")
            print("1. MMLU")
            print("q. Quit")
            choice = input("Enter choice: ").strip().lower()

            if choice in ["q", "quit", "exit"]:
                print("Exiting evaluation mode.")
                break
            elif choice == "1":
                print("Running MMLU evaluation...\n")
                evaluate_mmlu()
                break
            else:
                print("Invalid choice. Please enter 1 or q.")
    else:
        print("==== Sinon Chatbot ====\n")
        while True:
            prompt = input("User: ").strip()
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting chatbot mode.")
                break
            generate_responses(prompt)
