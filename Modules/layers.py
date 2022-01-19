import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *

class myRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dp=0,bd=False):
        super(myRNN,self).__init__()
        self.hidden_dim = hidden_size
        self.n_layers = num_layers
        self.RNN = nn.GRU(input_size = input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dp,batch_first=True,bidirectional=bd)
       
    def forward(self,x,h0=None):
        out,h = self.RNN(x,h0)
        return out,h

class attention(nn.Module):
    def __init__(self,qembed_dim, kembed_dim=None, vembed_dim=None, hidden_dim=None, out_dim=None, dropout=0):
        super(attention, self).__init__()
        if kembed_dim is None:
            kembed_dim = qembed_dim
        if hidden_dim is None:
            hidden_dim = kembed_dim
        if out_dim is None:
            out_dim = kembed_dim
        if vembed_dim is None:
            vembed_dim = kembed_dim
            
        self.qembed_dim = qembed_dim
        self.kembed_dim = kembed_dim
        self.vembed_dim = vembed_dim
        
        self.hidden_dim = hidden_dim
        self.for_key = nn.Linear(kembed_dim,hidden_dim)
        self.for_query = nn.Linear(qembed_dim,hidden_dim)
        self.for_value = nn.Linear(vembed_dim,hidden_dim)
        self.normalise_factor = hidden_dim**(1/2)
    
    def mask_score(self,s,m):
        for i in range(s.size()[0]):
            for j in range(s.size()[1]):
                for k in range(s.size()[2]):
                    if m[i][j][k] == 0:
                        s[i][j][k] = float('-inf')   #So that after softmax, 0 weight is given to it
        return s
    
    def forward(self,key,query,mask=None):
        if len(query.shape) == 1:
            query = torch.unsqueeze(query, dim=0)
        if len(key.shape) == 1:
            key = torch.unsqueeze(key, dim=0)
            
        if len(query.shape) == 2:
            query = torch.unsqueeze(query, dim=1)
        if len(key.shape) == 2:
            key = torch.unsqueeze(key, dim=1)
            
        new_query = self.for_query(query)
        new_key = self.for_key(key)
        new_value = self.for_value(key)
        
        score = torch.bmm(new_query,new_key.permute(0,2,1))/self.normalise_factor
        
        if mask != None:
            score = self.mask_score(score,mask)
            
        score = F.softmax(score,-1)
        score.data[score!=score] = 0         #removing nan values
        
        output = torch.bmm(score,new_value)
        return output,score

class interact(nn.Module):
    def __init__(self,hidden_dim,weight_matrix,utt2idx):
        super(interact, self).__init__()
        self.hidden_size = hidden_dim

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weight_matrix,utt2idx)
        self.rnnD = myRNN(embedding_dim, hidden_dim,1)   #Dialogue
        self.drop1 = nn.Dropout()
        
        self.rnnG = myRNN(embedding_dim*3, hidden_dim,1)   #Global level
        self.drop2 = nn.Dropout()
        
        self.attn = attention(embedding_dim)
        
        self.rnnS = myRNN(embedding_dim*2, embedding_dim*2,1)   #Speaker representation
        self.drop3 = nn.Dropout()

    def forward(self, chat_ids, speaker_info, sp_dialogues, sp_ind, inputs):
        whole_dialogue_indices = inputs
        
        bert_embs = self.embedding(whole_dialogue_indices)
               
        dialogue, h1 = self.rnnD(bert_embs)    #Get global level representation
        dialogue = self.drop1(dialogue)

        device = inputs.device
        
        fop = torch.zeros((dialogue.size()[0],dialogue.size()[1],dialogue.size()[2])).to(device)
        fop2 = torch.zeros((dialogue.size()[0],dialogue.size()[1],dialogue.size()[2]*3)).to(device)
        op = torch.zeros((dialogue.size()[0],dialogue.size()[1],dialogue.size()[2])).to(device)
        spop = torch.zeros((dialogue.size()[0],dialogue.size()[1],dialogue.size()[2]*2)).to(device)
                    
        h0 = torch.randn(1, 1, self.hidden_size*2).to(device)
        d_h = torch.randn(1, 1, self.hidden_size).to(device)
        attn_h = torch.randn(1, 1, self.hidden_size).to(device)
        
        for b in range(dialogue.size()[0]):
            d_id = chat_ids[b]
            speaker_hidden_states = {}
            for s in range(dialogue.size()[1]):
                fop = op.clone()
                
                current_utt = dialogue[b][s]
                
                current_speaker = speaker_info[d_id][s]
                
                if current_speaker not in speaker_hidden_states:
                    speaker_hidden_states[current_speaker] = h0
                
                h = speaker_hidden_states[current_speaker]
                current_utt_emb = torch.unsqueeze(torch.unsqueeze(current_utt,0),0)
                
                key = fop[b][:s+1].clone()
                key = torch.unsqueeze(key,0)
                
                if s == 0:
                    tmp = torch.cat([attn_h,current_utt_emb],-1).to(device)
                    spop[b][s], h_new = self.rnnS(tmp,h)
                else:
                    query = current_utt_emb
                    attn_op,_ = self.attn(key,query)
                    
                    tmp = torch.cat([attn_op,current_utt_emb],-1).to(device)
                    spop[b][s], h_new = self.rnnS(tmp,h)
                
                spop[b][s] = spop[b][s].add(tmp)        # Residual Connection        
                speaker_hidden_states[current_speaker] = h_new
                
                fop2[b][s] = torch.cat([spop[b][s],dialogue[b][s]],-1)
                tmp = torch.unsqueeze(torch.unsqueeze(fop2[b][s].clone(),0),0)
                op[b][s],d_h = self.rnnG(tmp,d_h)

        return op,spop
    
class fc_e(nn.Module):
    def __init__(self,inp_dim,op_dim):
        super(fc_e,self).__init__()
        self.linear1 = nn.Linear(inp_dim,int(inp_dim/2))
        self.drop1 = nn.Dropout()
        
        self.linear2 = nn.Linear(int(inp_dim/2),int(inp_dim/4))
        self.drop2 = nn.Dropout(0.6)
        
        self.linear3 = nn.Linear(int(inp_dim/4),op_dim)
        self.drop3 = nn.Dropout(0.7)
    def forward(self,x):
        ip = x.float()
    
        op = self.linear1(ip)
        op = self.drop1(op)
        
        op = self.linear2(op)
        op = self.drop2(op)
        
        op = self.linear3(op)
        op = self.drop3(op)
        
        return op

class fc_t(nn.Module):
    def __init__(self,inp_dim,op_dim):
        super(fc_t,self).__init__()
        self.linear1 = nn.Linear(inp_dim,inp_dim)
        self.drop1 = nn.Dropout(0.7)
        
        self.linear2 = nn.Linear(inp_dim,inp_dim)
        self.drop2 = nn.Dropout(0.7)
        
        self.linear3 = nn.Linear(inp_dim,int(inp_dim/2))
        self.drop3 = nn.Dropout(0.7)
        
        self.linear4 = nn.Linear(int(inp_dim/2),int(inp_dim/4))
        self.drop4 = nn.Dropout(0.7)
        
        self.linear5 = nn.Linear(int(inp_dim/4),op_dim)
        self.drop5 = nn.Dropout(0.7)
    def forward(self,x):
        ip = x.float()
    
        op = self.linear1(ip)
        op = self.drop1(op)
        
        op = self.linear2(ip)
        op = self.drop2(op)
        
        op = self.linear3(ip)
        op = self.drop3(op)
        
        op = self.linear4(op)
        op = self.drop4(op)
        
        op = self.linear5(op)
        op = self.drop5(op)
        
        return op
    
class maskedattn(nn.Module):
    def __init__(self,batch_size, s_len, emb_size):
        super(maskedattn,self).__init__()
        self.b_len = batch_size
        self.s_len = s_len
        self.emb_size = emb_size
        self.attn = attention(emb_size*2, kembed_dim=emb_size, out_dim=emb_size)
    
    def create_mask(self,n):
        mask = torch.zeros((1, self.s_len, self.emb_size), dtype=torch.uint8)
        mask[:n+1] = torch.ones((self.emb_size), dtype=torch.uint8)
        mask = mask.repeat(self.b_len,1,1)
        return mask
        
    def forward(self,key,query):
        device = key.device

        ops = torch.zeros([key.size()[0],key.size()[1], key.size()[2]], dtype=torch.float32).to(device)
        for i in range(key.size()[1]):
          mask = self.create_mask(i)
          op,_ = self.attn(key,query,mask=mask)
          for b in range(op.size()[0]):
            ops[b][i] = op[b][i]
        return ops
    
class memnet(nn.Module):
  def __init__(self,num_hops,hidden_size,batch_size,seq_len):
    super(memnet,self).__init__()
    self.num_hops = num_hops
    self.rnn = myRNN(hidden_size, hidden_size, 1)
    self.masked_attention = maskedattn(batch_size,seq_len,hidden_size)
  
  def forward(self,globl,spl):
    X = globl
    for hop in range(self.num_hops):
      dialogue,h = self.rnn(X)
      X = self.masked_attention(dialogue,spl)
    return X

class pool(nn.Module):
    def __init__(self,mode="mean"):
        super(pool,self).__init__()
        self.mode = mode
    def forward(self,x):
        device = x.device
        op = torch.zeros((x.size()[0],x.size()[1],x.size()[2])).to(device)
        for b in range(x.size()[0]):
            this_tensor = []
            for s in range(x.size()[1]):
                this_tensor.append(x[b][s])
                if self.mode == "mean":
                    op[b][s] = torch.mean(torch.stack(this_tensor),0)
                elif self.mode == "max":
                    op[b][s],_ = torch.max(torch.stack(this_tensor),0)
                elif self.mode == "sum":
                    op[b][s] = torch.sum(torch.stack(this_tensor),0)
                else:
                    print("Error: Mode can be either mean or max only")
        return op

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)