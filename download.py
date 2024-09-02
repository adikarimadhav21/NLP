from huggingface_hub import hf_hub_download

# List of filenames to download
filenames = [
    "config.json",
    "merges.txt",
    "model.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "vocab.json"
]

# Iterate over the list of filenames and download each file
for filename in filenames:
   path= hf_hub_download(repo_id="pszemraj/led-base-book-summary", filename=filename)
   print(path)


