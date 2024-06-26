import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
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
    
class ResidualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
       
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
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
    
    
    
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()
        
    def forward(self,x,src_mask):
        for layer in self.layers:
            x = layer(x,src_mask)
        return self.norm(x)
        
        
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
    
class Decoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,encoder_output,src_mask,target_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,target_mask)
        return self.norm(x)
        
        
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        # to do : (batch, seqlen, d_model) -> (batch, seqlen, vocab_size) -> (batch, seqlen
        return torch.log_softmax(self.proj(x), dim = -1)
    
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
    
        
def build_transformer(src_vocab_size, target_vocab_size, src_seq_len, target_seq_len,d_model=512,N = 6, num_heads = 8, dropout = 0.1, hidden_dims = 2048,) -> Transformer:
    src_embedd = InputEmbeddings(d_model, src_vocab_size)
    target_embedd = InputEmbeddings(d_model,target_vocab_size)
    
    src_pos = PositionalEncoding(d_model, src_seq_len,dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len,dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,num_heads,dropout)
        mlp = FeedForwardBlock(d_model, hidden_dims,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,mlp,dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,num_heads,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,num_heads,dropout)
        mlp = FeedForwardBlock(d_model,hidden_dims,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,mlp,dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model,target_vocab_size)
    
    transformer = Transformer(encoder,decoder,src_embedd,target_embedd,src_pos,target_pos,projection_layer)
    
    #Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            
    return transformer