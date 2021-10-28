import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from models import ERC_MMN, EFR_TX, cascade
from pickle_loader import load_erc, load_efr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
seq_len = 15
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
    speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
    final_speaker_info, final_speaker_dialogues, final_speaker_emotions,\
    final_speaker_indices, final_utt_len = load_erc()

weight_matrix = weight_matrix.to(device)
train_cnt = len(my_dataset_train)

_, _, _, _, _, _, _, my_dataset_train2, my_dataset_test2,\
        global_speaker_info2, speaker_dialogues2, speaker_emotions2, \
        speaker_indices2, utt_len2, global_speaker_info_test2, speaker_dialogues_test2, \
        speaker_emotions_test2, speaker_indices_test2, utt_len_test2 = load_efr()
    
def get_train_test_loader(bs):
    train_data_iter = data.DataLoader(my_dataset_train,batch_size=bs, drop_last=True)
    test_data_iter = data.DataLoader(my_dataset_test,batch_size=bs, drop_last=True)
    
    train_data_iter2 = data.DataLoader(my_dataset_train2,batch_size=1)
    test_data_iter2 = data.DataLoader(my_dataset_test2,batch_size=1)
    
    return train_data_iter, test_data_iter, train_data_iter2, test_data_iter2

def train(trigger_model, emotion_model, model, train_data_loader, train_data_loader2, epochs):
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none').to(device)
    
    params = list(model.parameters()) + list(emotion_model.parameters())
    optimizer = torch.optim.Adam(params,lr=1e-6,weight_decay=1e-6)
        
    max_f1_1 = 0
    for epoch in tqdm.tqdm(range(epochs)):
        train_loader2 = enumerate(train_data_loader2)
        
        print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
        model.train()
        emotion_model.train()
        trigger_model.eval()
        
        avg_loss = 0
        
        y_true1 = []
        y_pred1 = []
        
        y_pred1_old = []
            
        for i_batch, sample_batched in tqdm.tqdm(enumerate(train_data_loader)):
            dialogue_ids = sample_batched[0].tolist()
            inputs = sample_batched[1].to(device)           
            targets1 = sample_batched[2].to(device)

            optimizer.zero_grad()
            
#             with torch.no_grad():
            op_old_rep, op_old = emotion_model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs)
                        
            loss = 0
            for b in range(batch_size):
              loss1 = 0
              current_speaker_emo = {}
                
              for s in range(final_utt_len[dialogue_ids[b]]):
                curr_speaker = final_speaker_info[dialogue_ids[b]][s]
                curr_emotion = final_speaker_emotions[dialogue_ids[b]][s]
                
                if curr_speaker not in current_speaker_emo.keys():
                    current_speaker_emo[curr_speaker] = curr_emotion
                    trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
                else:
                    prev_emotion = current_speaker_emo[curr_speaker]
                    current_speaker_emo[curr_speaker] = curr_emotion
                    if prev_emotion != curr_emotion:
                        _,trig_ip = next(train_loader2)
                        trig_ids = trig_ip[0].tolist()
                        trig_ips = trig_ip[1].to(device)

                        with torch.no_grad():
                            trig_op_tmp,_ = trigger_model(trig_ips, trig_ids, utt_len2)
                            trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
                            for t1 in range(len(trig_op_tmp)):
                                trig_op[t1] = trig_op_tmp[0][t1]
                    else:
                        trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
                
                new_ip = torch.cat([op_old_rep[b],trig_op],-1)
                outputs = model(new_ip)
                                
                pred1_old = torch.unsqueeze(op_old[b][s],dim=0).to(device)
                pred_emo_old = torch.argmax(F.softmax(pred1_old,-1),-1)
                
                pred1 = torch.unsqueeze(outputs[s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
                y_pred1_old.append(pred_emo_old.item())
                
                y_pred1.append(pred_emo.item())
                y_true1.append(truth1.item())

                loss1 += criterion1(pred1,truth1)
              
              loss1 /= final_utt_len[dialogue_ids[b]]
              loss += loss1
            
            loss /= batch_size
            avg_loss += loss

            loss.backward()            
            optimizer.step()
            
        avg_loss /= len(train_data_loader)
        print("Average Loss = ",avg_loss)
        
        f1_1, v_loss = validate(emotion_model,trigger_model,model,data_iter_test,data_iter_test2,epoch)
        
        # if f1_1 > max_f1_1:
        #     print(f"Saving model at epoch {epoch}")
        #     max_f1_1 = f1_1
        #     torch.save(model.state_dict(), "./model_{}_weight_state_dict.pth".format(model_name))
        #     torch.save(emotion_model.state_dict(), "./emotion_model_{}_weight_state_dict.pth".format(model_name))

    return model

def validate(emotion_model, trigger_model, model, test_data_loader, test_data_loader2, epoch):
    print("\n\n***VALIDATION ({})***\n\n".format(epoch))
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none')
    
    trigger_model.eval()
    emotion_model.eval()
    model.eval()
    
    test_loader2 = enumerate(test_data_loader2)

    with torch.no_grad():
      avg_loss = 0
        
      y_true1 = []
      y_pred1 = []
        
      y_pred1_old = []
            
      for i_batch, sample_batched in tqdm.tqdm(enumerate(test_data_loader)):
          dialogue_ids = sample_batched[0].tolist()
          dialogue_ids = [d+train_cnt for d in dialogue_ids]
          inputs = sample_batched[1].to(device)           
          targets1 = sample_batched[2].to(device)
           
          with torch.no_grad():
                op_old_rep, op_old = emotion_model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs)
                        
          loss = 0
          for b in range(batch_size):
            loss1 = 0
            current_speaker_emo = {}
              
            for s in range(final_utt_len[dialogue_ids[b]]):
              curr_speaker = final_speaker_info[dialogue_ids[b]][s]
              curr_emotion = final_speaker_emotions[dialogue_ids[b]][s]
              
              if curr_speaker not in current_speaker_emo.keys():
                  current_speaker_emo[curr_speaker] = curr_emotion
                  trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
              else:
                  prev_emotion = current_speaker_emo[curr_speaker]
                  current_speaker_emo[curr_speaker] = curr_emotion
                  if prev_emotion != curr_emotion:
                     _,trig_ip = next(test_loader2)
                     trig_ids = trig_ip[0].tolist()
                     trig_ips = trig_ip[1].to(device)

                     with torch.no_grad():
                         trig_op_tmp,_ = trigger_model(trig_ips, trig_ids, utt_len_test2)
                         trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
                         for t1 in range(len(trig_op_tmp)):
                           trig_op[t1] = trig_op_tmp[0][t1]
                  else:
                     trig_op = torch.zeros((seq_len,hidden_size*2)).to(device)
                
              new_ip = torch.cat([op_old_rep[b],trig_op],-1)
              with torch.no_grad():
                    outputs = model(new_ip)
              
              pred1_old = torch.unsqueeze(op_old[b][s],dim=0).to(device)
              pred_emo_old = torch.argmax(F.softmax(pred1_old,-1),-1)
                
              pred1 = torch.unsqueeze(outputs[s],dim=0).to(device)
              truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

              pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
              y_pred1_old.append(pred_emo_old.item())
                
              y_pred1.append(pred_emo.item())
              y_true1.append(truth1.item())

              loss1 += criterion1(pred1,truth1)
              
            loss1 /= final_utt_len[dialogue_ids[b]]
            loss += loss1
            
          loss /= batch_size
          avg_loss += loss

      avg_loss /= len(test_data_loader)

      class_report = classification_report(y_true1,y_pred1)
      wtd_f1 = f1_score(y_true1,y_pred1,average="weighted")

      conf_mat1_old = confusion_matrix(y_true1,y_pred1_old)
      conf_mat1 = confusion_matrix(y_true1,y_pred1)

      print(class_report)
      print("Confusion Matrix:-\n",conf_mat1)
      return wtd_f1, avg_loss
    
nclass = 2
emsize = 768
nhid = 768
nlayers = 6
nhead = 2
dropout = 0.2
trigger_model = EFR_TX(weight_matrix, utt2idx, nclass, emsize, nhead, nhid, nlayers, device, dropout).to(device)
# trigger_model.load_state_dict(torch.load("efr_best_model.pth"))

emotion_model = ERC_MMN(hidden_size, weight_matrix, utt2idx, batch_size, seq_len).to(device)
# emotion_model.load_state_dict(torch.load("erc_best_model.pth"))

model = cascade(hidden_size, 7).to(device)

weights1 = [1.0]*7
data_iter_train, data_iter_test, data_iter_train2, data_iter_test2 = get_train_test_loader(batch_size)
model = train(trigger_model,emotion_model,model, data_iter_train, data_iter_train2, epochs = 100)