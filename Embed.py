import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pdb

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model) #initializing the embedding function
    def forward(self, x):
   
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
       
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len): #assign positionan encoding vecor for each 
            for i in range(0, d_model, 2): # positional encoding for each dimention is get through sin and cos fubctuion
     
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)


     
        self.register_buffer('pe', pe) #Adding this to the state dict they are tensors not variables
 
    
    def forward(self, x):

        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
  
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
         
        
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)