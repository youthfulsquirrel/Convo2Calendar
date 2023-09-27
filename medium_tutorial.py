import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('cuda')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")

prompt = "When was the lightbulb invented?"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

# Generate a response
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
