import torch, tiktoken
from config import *
from model import GPTLanguageModel
from train import training_loop

MAX_NEW_TOKENS = 300
TEMPERATURE = 1 

def sample(prompt):
    # Load the pre-trained GPT model
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load(gpt_model_path))

    # Get the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Encode the input prompt
    prompt_tokens = enc.encode(prompt)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generate text based on the prompt
    result = model.generate(context, MAX_NEW_TOKENS + len(prompt), TEMPERATURE)[0].tolist()
    result = result[len(prompt_tokens):]
    result = enc.decode(result)
    print(result)

if __name__ == "__main__":
    # Read the training text
    file = 'Training Files/chunk_0.txt'
    text = open(file, "r", encoding="utf-8").read()

    # Train the GPT model
    training_loop(text)

    # Sample from the trained model
    # sample(input("Enter your prompt: "))