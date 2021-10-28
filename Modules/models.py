import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers import *
from utils import *

class ERC_MMN(nn.Module):
    def __init__(self,hidden_size,weight_matrix,utt2idx,batch_size,seq_len):
        super(ERC_MMN,self).__init__()
        self.ia = interact(hidden_size,weight_matrix,utt2idx)
        self.mn = memnet(4,hidden_size,batch_size,seq_len)
        self.pool = pool()
        
        self.rnn_c = myRNN(hidden_size*3,hidden_size*2,1)
        
        self.rnn_e = myRNN(hidden_size*2,hidden_size*2,1)
                
        self.linear1 = fc_e(hidden_size*2,7)

    def forward(self,c_ids,speaker_info,sp_dialogues,sp_em,sp_ind,x1,mode="train"):
        glob, splvl = self.ia(c_ids,speaker_info,sp_dialogues,sp_ind,x1)

        op = self.mn(glob,splvl)
        op = self.pool(op)

        op = torch.cat([splvl,op],dim=2)

        rnn_c_op,_ = self.rnn_c(op)

        rnn_e_op,_ = self.rnn_e(rnn_c_op)
        fip = rnn_e_op.add(rnn_c_op)      # Residual Connection
        fop1 = self.linear1(fip)

        return fip,fop1

class EFR_TX(nn.Module):
    def __init__(self, weight_matrix, utt2idx, nclass, ninp, nhead, nhid, nlayers, device, dropout=0.5):
        super(EFR_TX, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder, num_embeddings, embedding_dim = create_emb_layer(weight_matrix,utt2idx)
        self.ninp = ninp
        self.decoder = nn.Linear(2*ninp, nclass)

        self.init_weights()
        self.device = device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, d_ids, ut_len):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        decoder_ip = torch.zeros(output.size()[0],output.size()[1],output.size()[2]*2).to(self.device)
        for b in range(output.size()[0]):
            d_id = d_ids[b][0]
            main_utt = output[b][ut_len[d_id]-1]
            for s in range(ut_len[d_id]):
                this_utt = output[b][s]
                decoder_ip[b][s] = torch.cat([this_utt,main_utt],-1)
        
        output = self.decoder(decoder_ip)
        
        return decoder_ip,output

class ERC_true_EFR(nn.Module):
    def __init__(self, weight_matrix, utt2idx, nclass, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(ERC_true_EFR, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder, num_embeddings, embedding_dim = create_emb_layer(weight_matrix,utt2idx)
        
        self.emoGRU = myRNN(7,100,1)
        self.ninp = ninp
        self.decoder = nn.Linear(2*ninp+100, nclass)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, em_seq, d_ids, ut_len):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        emo_seq,_ = self.emoGRU(em_seq.float())
        
        decoder_ip = torch.zeros(output.size()[0],output.size()[1],output.size()[2]*2).cuda()
        for b in range(output.size()[0]):
            d_id = d_ids[b][0]
            main_utt = output[b][ut_len[d_id]-1]
            for s in range(ut_len[d_id]):
                this_utt = output[b][s]
                decoder_ip[b][s] = torch.cat([this_utt,main_utt],-1)
        
        decoder_ip = torch.cat([decoder_ip,emo_seq],-1)
        output = self.decoder(decoder_ip)
        
        return output

class ERC_EFR_multitask(nn.Module):
    def __init__(self,hidden_size,weight_matrix,utt2idx,batch_size,seq_len):
        super(ERC_EFR_multitask,self).__init__()
        self.ia = interact(hidden_size,weight_matrix,utt2idx)
        self.mn = memnet(4,hidden_size,batch_size,seq_len)
        self.pool = pool()
        
        self.rnn_c = myRNN(hidden_size*3,hidden_size*2,1)
        
        self.rnn_e = myRNN(hidden_size*2,hidden_size*2,1)
        self.rnn_t = myRNN(hidden_size*2,hidden_size,1)

        self.linear1 = fc_e(hidden_size*2,7)
        self.linear2 = fc_t(hidden_size*2,2)

    def forward(self,c_ids,speaker_info,sp_dialogues,sp_em,sp_ind,freeze,x1,mode="train"):
        speaker_emo = {}
        speaker_emo_distance = {}
        
        for d_id in c_ids:
            speaker_emo[d_id] = {}
            speaker_emo_distance[d_id] = {}
                    
        if freeze:
            with torch.no_grad():
                glob, splvl = self.ia(c_ids,speaker_info,sp_dialogues,sp_ind,x1)
        
                op = self.mn(glob,splvl)
                op = self.pool(op)

                op = torch.cat([splvl,op],dim=2)

                rnn_c_op,_ = self.rnn_c(op)

                rnn_e_op,_ = self.rnn_e(rnn_c_op)
                rnn_e_op = rnn_e_op.add(rnn_c_op)      # Residual Connection
                fop1 = self.linear1(rnn_e_op)
        else:
            glob, splvl = self.ia(c_ids,speaker_info,sp_dialogues,sp_ind,x1)
        
            op = self.mn(glob,splvl)
            op = self.pool(op)
            
            op = torch.cat([splvl,op],dim=2)

            rnn_c_op,_ = self.rnn_c(op)

            rnn_e_op,_ = self.rnn_e(rnn_c_op)
            rnn_e_op = rnn_e_op.add(rnn_c_op)      # Residual Connection
            fop1 = self.linear1(rnn_e_op)
        
        rnn_t_op,_ = self.rnn_t(rnn_c_op)

        fop2_final = []
        for b in range(rnn_t_op.size()[0]):
            d_id = c_ids[b]
            fop2_final_tmp = []
            for s in range(rnn_t_op.size()[1]):
                fop2_final_tmp_tmp = []
                concerned_utt = rnn_t_op[b][s]
                
                if s < 4:
                    r = s+1
                else:
                    r = 4
                
                for s2 in range(r,-1,-1):
                    this_utt = rnn_t_op[b][s-s2]
                    tmp = torch.cat((concerned_utt,this_utt),-1)
                    fop2 = self.linear2(tmp)

                    fop2_final_tmp_tmp.append(fop2)
                fop2_final_tmp.append(fop2_final_tmp_tmp)
            fop2_final.append(fop2_final_tmp)
        return fop1,fop2_final

class cascade(nn.Module):
    def __init__(self,hidden_size,nclasses):
        super(cascade,self).__init__()        
        self.linear = fc_e(hidden_size*4,nclasses)
    
    def forward(self,x1):
        op = self.linear(x1)
        return op