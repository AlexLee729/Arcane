import torch
import sys
from model import GPT
from train import training_loop
import importlib.util

MAX_NEW_TOKENS = 500
TEMPERATURE = 1 

gpt_model_path = 'Models/Arcane_smallv2.pth'

def load_config():
    spec = importlib.util.spec_from_file_location("config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def sample(prompt, config, model_path):
    # Load the pre-trained GPT model
    model = GPT(config).to(config.device)
    model.load_state_dict(torch.load(model_path))

    # # Generate text based on the prompt
    # context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    result = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    # Load the configuration from the file
    config_file = sys.argv[1]
    config = load_config()

    # # Training
    file = f'Training Files/Val_text1.txt'
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    training_loop(text, gpt_model_path, config)

    # Sample from the trained model
    # prompt = input("Enter your prompt: ")
    # sample(prompt, config, gpt_model_path)
