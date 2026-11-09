import torch

# 1. Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()
print(f"Is GPU available? {gpu_available}")

# 2. Get the name of the active GPU
if gpu_available:
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")

from accelerate import Accelerator
torch.cuda.is_available()

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import userdata
model_id = "gpt2"
token = userdata.get('hf_token')
os.environ["HF_TOKEN"] = token
tokenizer = AutoTokenizer.from_pretrained(model_id)
model =  AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

text = "Chat GPT is a "
i = 0
predicted_words = []
while(i<20):
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
        outputs = model(**inputs)
  next_token_logits = outputs.logits[0, -1, :]
  next_token_id = torch.argmax(next_token_logits).item()
  next_word = tokenizer.decode([next_token_id])
  predicted_words.append(next_word)
  i=i+1

for i in range(len(predicted_words)):
  print(predicted_words[i])