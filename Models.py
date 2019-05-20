import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy
import pdb

def get_clones(module, N):
    
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model) #Initializing embedding matrix for the source words
      
        self.pe = PositionalEncoder(d_model, dropout=dropout) #Initialize the positional encoding 

        

        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N) #Repeat the Initialized EncorderLayer structure

  
        self.norm = Norm(d_model) #Normalization layer initialization

    def forward(self, src, mask):

        x = self.embed(src)
        x = self.pe(x)

    
        for i in range(self.N):    #staking layers on top of each other and getting the encoder output  
            
            x = self.layers[i](x, mask)

     
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N,  heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model) #Initializing embedding matrix for the target words
        self.pe = PositionalEncoder(d_model, dropout=dropout) #Initialize the positoonal encodin
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N) #Repeating the Decorderlayer structure of the ntimes
        self.norm = Norm(d_model) #Normalization layer
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
           
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
            
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
       
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout) #initalizin the encoder
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout) #initializing the decoder
        self.out = nn.Linear(d_model, trg_vocab) #linear output layer. output scores for words equal to the size of the target language tokens
    def forward(self, src, trg, src_mask, trg_mask):
      
        e_outputs = self.encoder(src, src_mask) #There is a different encoding for each sequencial element after all stacked encoding blocks 
    
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output) #output a probability distribtion
   
        return output #for each word output

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    
 

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout) #Initializing the model
    


       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model
    
