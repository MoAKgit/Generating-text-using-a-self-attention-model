# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:38:58 2023

@author: Mohammad
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from models import *
from utils import *

def arg_parse():
    """
    Parse arguements
    
    """
    parser = argparse.ArgumentParser(description='GPT')
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 3e-4) 
    parser.add_argument("--batch_size",dest= 'batch_size', default= 64) 
    parser.add_argument("--block_size",dest= 'block_size', default= 256) 
    parser.add_argument("--max_iters",dest= 'max_iters', default= 7000) 
    parser.add_argument("--eval_interval",dest= 'eval_interval', default= 500) 
    parser.add_argument("--n_embd",dest= 'n_embd', default= 384) 
    parser.add_argument("--n_head",dest= 'n_head', default= 6) 
    parser.add_argument("--n_layer",dest= 'n_layer', default= 6) 
    parser.add_argument("--dropout",dest= 'dropout', default= 0.5) 
    parser.add_argument("--eval_iters",dest= 'eval_iters', default= 200) 
    
    return parser.parse_args()



args = arg_parse()

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# ------------


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print(len(text))
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)


'''
In this projec I am going to represent a simple charactor level attention based 
decorer for randomly generating reasonably meaningfull text.
'''

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# print(stoi)
# print(itos)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]



x, y = get_batch(args, 'train', train_data, val_data, device)

# context = decode(x)
# target = decode(y)
# print(x.shape)
# print(x)
# print(y.shape)
# print(y)



model = GPTLanguageModel(args,vocab_size, device )
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


logic, loss  = model(x, y)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)



for iter in range(args.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss(args, model, train_data, val_data, device)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(args, 'train', train_data, val_data, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))





