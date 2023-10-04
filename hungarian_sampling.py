from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("NYTK/PULI-GPT-2")


model = AutoModelForCausalLM.from_pretrained("NYTK/PULI-GPT-2", pad_token_id=tokenizer.eos_token_id).to(torch_device)


model_inputs = tokenizer('A kutya az', return_tensors='pt').to(torch_device)
"""

greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

beam_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    num_return_sequences=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
"""


from transformers import set_seed
set_seed(42)


sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    
    top_p=0.95,
    no_repeat_ngram_size=2,
    num_return_sequences=3,
    top_k=50
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
