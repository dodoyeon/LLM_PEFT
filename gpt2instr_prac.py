# Use a pipeline as a high-level helper
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipe = pipeline("text-generation", model="vicgalle/gpt2-open-instruct-v1")

tokenizer = AutoTokenizer.from_pretrained("vicgalle/gpt2-open-instruct-v1")
model = AutoModelForCausalLM.from_pretrained("vicgalle/gpt2-open-instruct-v1")

input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nPretend you are a counselor. Write some advices to a depression patient."
# Pretend you are an alien visiting Earth. Write three opinions you believe.\n\n### Response:\n\n1. Earth is a beautiful place. The sky is clear and the land is lush and diverse.\n2. I believe that there is a species of extraterrestrial life living on the planet. These are known as 'gods' or 'living beings'.\n3. I believe that there is a great conspiracy in place. The government is trying to control the planet and its resources.

model_inputs = tokenizer([input], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=40)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])