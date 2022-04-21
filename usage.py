import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'WIP'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

if torch.cuda.is_available():
    model.cuda()
  
def generate(text, **kwargs):
    '''
    Generates text based on the input text.
    :param text: The input text.
    :param kwargs: A dictionary of parameters.
    :return: The generated text.
    '''
    inpt = tokenizer.encode(text, return_tensors="pt")
    if torch.cuda.is_available():
        out = model.generate(inpt.cuda(), **kwargs)
    else:
        out = model.generate(inpt, **kwargs)
    return tokenizer.decode(out[0])
  

act = "Текст"
print(generate(act, max_length=500, repetition_penalty=5.0, top_k=5, top_p=0.95, temperature=0.9))