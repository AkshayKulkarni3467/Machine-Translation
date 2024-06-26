# Using Transformer Architecture for Machine Translation
A transformer is an encoder decoder model which uses the concept of self-attention to get a sense of the context of words.

Architecture of transformer:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/a73bd2b0-3649-47be-a417-a6ba25910411)


## Overview

- `input embeddings`: Embeddings like word2vec which capture the context of words in global space can be used for the encoder and decoder input. Here we use the Embeddings layer by pytorch.
```python
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
```
> Shape : input = (batch, sequence_length) -> (batch, sequence_length, embedding_size)

Here, number of parameters = embedding_size * vocabulary_size

- `positional encoding `: Positional encodings are used to capture the position's information for the model to learn. This is done because the self-attention block doesnt know the position of the context.
```python
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_length:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        #Create a matrix of shape (seq_length,d_model)
        po = torch.zeros(seq_length,d_model)
        #Create a vector of shape (seq_length)
        position = torch.arange(0,seq_length,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #Apply the sin to even positions
        po[:,0::2] = torch.sin(position*div_term)
        po[:,1::2] = torch.cos(position*div_term)
        po = po.unsqueeze(0) # (1, seq_length, d_model)
        
        self.register_buffer('po',po)
        
    def forward(self,x):
        x = x + (self.po[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
```
> Shape : input = (batch, sequence_length, embedding_size) -> (batch, sequence_length, embedding_size)

There are no learnable parameters here. We use the formule given in the paper "Attention is all you need" to encode the positional values using the formula:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/594aa705-77f9-4790-9c47-fd6ded161778)


- `multi-head attention `: Attention is used to capture the context of the tokens in each sentence. Consider 1 Attention head, the input is passed through three linear layers, to form the Q, k, V tensors. Then, we use the formula softmax(Q.K/sqrt(len(K))).V and pass it through a linear layer for the output.

