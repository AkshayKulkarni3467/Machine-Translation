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
```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % num_heads == 0, "d_model should be divisible by number of heads"
        
        self.d_k = d_model // num_heads #512/8 = 64 in each attention head
        self.w_q = nn.Linear(d_model,d_model) #WQ
        self.w_k = nn.Linear(d_model,d_model) #WK
        self.w_v = nn.Linear(d_model,d_model) #WV
        
        self.w_o = nn.Linear(d_model,d_model) #W0
    
    @staticmethod    
    def selfAttention(q,k,v,mask,dropout):
        d_k = q.shape[-1]
        
        attention_scores = (q @ k.transpose(-1,-2)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
            
        attention_scores = attention_scores.softmax(dim=-1) # (batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ v), attention_scores
        
        
    def forward(self,q,k,v,mask):
        query = self.w_q(q) #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        
        query = query.view(query.shape[0],query.shape[1],self.num_heads,self.d_k).transpose(1,2) # (batch, num_heads, seq_len, d_k)
        key = key.view(key.shape[0],key.shape[1],self.num_heads,self.d_k).transpose(1,2) # (batch, num_heads, seq_len, d_k)
        value = value.view(value.shape[0],value.shape[1],self.num_heads,self.d_k).transpose(1,2) # (batch, num_heads, seq_len, d_k)
        
        x,self.attention_scores = MultiHeadAttentionBlock.selfAttention(query,key,value,mask,dropout=self.dropout)
        
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.num_heads*self.d_k) # (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        
        return self.w_o(x) # (batch, seq_len ,d_model)
```
> Shape : input = (batch, sequence_length, embedding_size) -> (batch, sequence_length, num_heads, single_matrix_embedd) -> (batch, num_heads, sequence_length, single_matrix_embedd) -> output = (batch, sequence_length, embedding_size)

Here, we use multi-head attention and divide the embedding dimension into 8 dimensions. Hence, we train the Q, K, V matrices 8 times and concat them again for the required output. 

Learnable parameters : 3 x embedding_size x embedding_size (Q, K, V) + 1 x embedding_size x embedding_size (W)

-`layer normalization`: Layer Normalization is used instead of batch normalization to avoid considering the unnecessary paddings added to the sentence to reach sequence length. Layer normalization normalises across every features in the sequence in a batch.

```python
class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 10**-6)-> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # Multiplicative
        self.beta = nn.Parameter(torch.ones(1)) # Additive
        
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.gamma*((x-mean)/(std+self.eps)) + self.beta
```
> Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, embedding_size)

We use two learnable parameters, gamma and beta, so that the model can tune the parameters in case it doesnt need the normalization.

Here, number of learnable parameters = 2

-`residual connection`: Residual connections are used in tranformers to reduce the vanishing gradients. We use residual connections 2 times in the encoder block and 3 times in the decoder block

```python
class ResidualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
       
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```
> Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, embedding_size)

We use the layer normalization and then add the residual connection, we is reverse of want they implement in the paper.

=`feed forward layer`: This is a normal multi layer perceptron.
```python
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,hidden_dims,dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model,hidden_dims)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden_dims,d_model)
        
    def forward(self,x):
        # input -> (batch,seq_len,d_model)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x
```
> Shape : input = (batch, ,sequence_length, embedding_size) -> (batch, sequence_length, 2048) -> output = (batch, sequence_length, embedding_size)

Number of parameters: 2 x hidden_dims x embedding_size

-`encoder block`: An encoder block is the combination of all the above layers. It takes the positional encodings as the input, passes it through multi-head attention and feed forward layer, and gives the output. The encoder block uses two residual connections.

```python
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, mlp : FeedForwardBlock, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.mlp = mlp
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
     
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x,x,x,src_mask)) 
        x =  self.residual_connections[1](x, lambda x : self.mlp(x))
        return x
```
> Shape : input (positional encoding) = (batch, sequence_length, embedding_size) -> output (batch, sequence_length, embedding_size)

-`encoder`: The encoder can contain many layers of encoder blocks to capture the entire position and context of a given sentence.

```python
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()
        
    def forward(self,x,src_mask):
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)
```
> Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, embedding_size)

In this implementation, we use 6 layers of the encoder. The output of the encoder is used as the K, V tensors for the input of the decoder.

-`decoder block`: The decoder block takes in the positional embeddings of the output language, performs masked self-attention on it. Masked self-attention makes sure that a token in any given sentence cannot compute the attention scores of the words ahead of it. The output of masked self-attention is used as the Q value in the cross self-attention block. Here, self-attention is performed between the Q and the K, V values of the output of the encoder. This is than fed into a feed forward layer which gives us the output of the decoder. The decoder block uses 3 residual connections.

```python
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attention_block: MultiHeadAttentionBlock, mlp: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.mlp = mlp
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self,x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x,lambda x : self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,lambda x : self.mlp(x))
        return x
```
> Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, embedding_size)

Number of parameters in Decoder : 3 x embedding_size + 1 x embedding_size (masked self attention) + 3 x embedding_size + 1 x embedding_size (cross self attention) + 2 x hidden_dim x embedding_size (feed forward layer) + 2 * 3 ( layer normalization)

-`decoder`: The decoder consists of layers of decoder blocks. In this project, we use 6 layers of decoder blocks

```python
class Decoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,encoder_output,src_mask,target_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,target_mask)
        return self.norm(x)
```
> Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, embedding_size)

-`projection layer`: The output of decoder is than passed through a projection layer which projects the embedding_size into target_vocab_size and applies softmax operation on that dimension. Hence, we can retrive the maximum probabiliy for a translated word in the sentence.

```python
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        # to do : (batch, seqlen, d_model) -> (batch, seqlen, vocab_size) -> (batch, seqlen
        return torch.log_softmax(self.proj(x), dim = -1)
```
>Shape : input = (batch, sequence_length, embedding_size) -> output = (batch, sequence_length, target_vocab_size)



  
## Transformer

- The transformer combines the embedddings, positional encoding, encoder block layers, decoder block layers, projection layer to output its prediction.
- We can calculate the cost function in transformer using the CrossEntropyLoss function, by the formula:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/8aff3d6b-c88f-46d9-a5df-3e1c259ff060)

```python
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder : Decoder, src_embedd: InputEmbeddings, target_embedd: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding,projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedd = src_embedd
        self.target_embedd = target_embedd
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embedd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target,target_mask):
        target = self.target_embedd(target)
        target = self.target_pos(target)
        return self.decoder(target,encoder_output,src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)
```
>Shape : input = (batch, sequence_length) -> output = (batch, sequence_length, target_vocab_size)

## Training the Transformer for Machine Translation:

We use the subset of english to italian translation from helsinki dataset to train the model.

Cost function after 10 epochs of training:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/83a66ff1-e3fa-4f49-acda-d797951c064c)

## Visualising the attention scores after 10 epochs:

- Attention scores in encoder block in layer 0 and head 0,1:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/3fdcaa6b-3ad4-42e6-9dd0-4fcdf2c4f559)

- Attention scores in masked attention decoder block in layer 0 and head 0,1:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/6682f05a-421a-4ada-9b7c-46b066995ca1)

As you can see, there's no attention scores in the upper triangular area of the matrix since the attention is masked.

- Attention scores in cross attention decoder block in layer 0 and head 0,1:

![image](https://github.com/AkshayKulkarni3467/Machine-Translation/assets/129979542/9661f2be-2544-46db-987b-b4f62a729fc7)


> Model achieving the right meaning even though the labels and the predicted output are different:
```text
--------------------------------------------------------------------------------
    SOURCE: 'Wait a moment!
    TARGET: — Ah, lascia stare!
 PREDICTED: — Ah , aspetta !
--------------------------------------------------------------------------------
```

  








