import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm

import pdb

class EncoderLayer(nn.Module): #This create single encoder layer block
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model) #Initialize a normalization layer
        self.norm_2 = Norm(d_model) #Initializae another noralzation layer
    
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout) #Initialize the self.attention (This is the self.attention)
        self.ff = FeedForward(d_model, dropout=dropout) #feedforward layer
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
     
        
        x2 = self.norm_1(x) #Normalize the input 
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))  #Resnidual connection
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2)) #output emebeddings for 8 heads in sequencial length
      
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout) #This one for the attention between encorder outputs and decoder querry
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout) #This for the self attention in decorder inputs 
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask): #Here the x means the target outputs, 
        x2 = self.norm_1(x)
     
        #self attention in targets
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)) #Targt mask has list of list where each elemet of the list is a mask that maskout future inputs
        

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))    #get the attention of target attention layers with encoder attention layers


        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x