import torch
from config import *
from model import GPTLanguageModel
from train import training_loop

MAX_NEW_TOKENS = 500
TEMPERATURE = 1 

def sample(prompt):
    # Load the pre-trained GPT model
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load(gpt_model_path))

    # Generate text based on the prompt
    result = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    print(result)

if __name__ == "__main__":
    # Read the training text
    file = f'Training Files/Val_text1.txt'
    text = open(file, "r", encoding="utf-8").read()

    # Train the GPT model
    training_loop(text, scheduler=True)

    # Sample from the trained model
    # sample(input("Enter your prompt: ")) 