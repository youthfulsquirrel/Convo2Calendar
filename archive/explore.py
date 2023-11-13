import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('cuda')
model = AutoModelForCausalLM.from_pretrained("model/phi-1_5", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tokenizer/phi-1_5", torch_dtype="auto")


inputs = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

