from transformers import GPT2Tokenizer, GPT2Model

# Specify the local directory where you want to save the model
local_directory = './gpt2-directory'

# Download the tokenizer and model and save them locally
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=local_directory)
model = GPT2Model.from_pretrained('gpt2', cache_dir=local_directory)

# Save the tokenizer and model locally
tokenizer.save_pretrained(local_directory)
model.save_pretrained(local_directory)
