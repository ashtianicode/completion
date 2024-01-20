

# Generatively Pretrained Transformer 



#%%
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))

# %%

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(len(chars))




# %%
# map of chars to integers

ctoi = { ch:i for i,ch in enumerate(chars)}
itoc = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [ ctoi[c] for c in s ]
decode = lambda l: ''.join([ itoc[i] for i in l ])

print(encode("hello"))
print(decode([46, 43, 50, 50, 53]))

# %%

import tiktoken 
enc = tiktoken.get_encoding("gpt2")
enc.encode(".")
enc.decode([31373,13])

# %%

import torch 
data = torch.tensor(encode(text), dtype=torch.int16)
print(decode(data[:100].tolist()))

# %%
# train val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# %%
# chunks to feed into transformer 


block_size = 8
data[:block_size+1]


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]

    print(f"when context is:{context}, the tartget is:{target}")

# %%

torch.manual_seed(4242)

batch_size = 4 # how mnay sequences will we process in parallel 
block_size = 8 # max length of a sequence to fit into context (used for predicting the target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


for b in range(batch_size): # batch dimention
    for t in range(block_size): # time dimension 
        context  = xb[b : t+1 ]
        target = yb[b,t]

        print(f"when context is {context.tolist()}, the target is {target}")


# %%
print(xb)

# %%


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(4242)



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,targets=None):
        idx = idx.long()  # Convert indices to LongTensor
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            targets = targets.long()  # Convert targets to LongTensor
            loss = F.cross_entropy(logits,targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions (logits)
            logits , loss = self(idx) 
            # focus on last step (what comes next)
            logits = logits[:,-1,:] #(B,C)
            # logits to probs
            probs = F.softmax(logits, dim=-1)
            # sample probs and get 1 
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # appened sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx 


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
#print(out.shape)
print(loss)


idx = torch.zeros((1,1),dtype=torch.int16)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
# %%
device = torch.device("cpu")
#device = torch.device('mps')



import time 
t1 = time.time()

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer =  torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
loss_sum = 0.0  # Initialize loss_sum

for steps in range(8000):
    if steps % 1000 == 999:
        loss_sum += loss.item()  # Update loss_sum with the current loss
        print('training...',loss_sum)
        loss_sum = 0 

    xb, yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(int(time.time() - t1),'s')

# %%timeit

idx = torch.zeros((1,1),dtype=torch.int16)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
# %%

# %%
print(torch.__version__)


# %%
