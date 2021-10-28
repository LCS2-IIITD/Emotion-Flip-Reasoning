import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from models import ERC_EFR_multitask
from pickle_loader import load_erc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
seq_len = 15
seq2_len = 15
emb_size = 768
hidden_size = 768
batch_first = True

idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
    speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
    final_speaker_info, final_speaker_dialogues, final_speaker_emotions,\
    final_speaker_indices, final_utt_len = load_erc()
    
def get_train_test_loader(bs):
    train_data_iter = data.DataLoader(my_dataset_train, batch_size=bs, shuffle = True)
    test_data_iter = data.DataLoader(my_dataset_test, batch_size=bs, shuffle = True)
    
    return train_data_iter, test_data_iter

def train(model, train_data_loader, epochs,log_step=2):
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none').to(device)
    
    class_weights2 = torch.FloatTensor(weights2).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights2,reduction='none').to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)

    max_f1_1, max_f1_2 = 0,0
    
    freeze = False
    for epoch in tqdm.tqdm(range(epochs)):
#         if epoch > 120:
#             freeze = True
        print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
        model.train()
        
        avg_loss = 0
        
        y_true1 = []
        y_pred1 = []
        
        y_true2 = []
        y_pred2 = []
            
        for i_batch, sample_batched in tqdm.tqdm(enumerate(train_data_loader)):
            dialogue_ids = sample_batched[0].tolist()
            inputs = sample_batched[1].to(device)
            targets1 = sample_batched[2].to(device)
            targets2 = sample_batched[3].to(device)

            optimizer.zero_grad()
            
            outputs = model(dialogue_ids,final_speaker_info,final_speaker_dialogues,final_speaker_emotions,final_speaker_indices,freeze,inputs)
            
            loss = 0
            for b in range(outputs[0].size()[0]):
              loss1 = 0
              loss2 = 0
              current_speaker_emo = {}
              for s in range(final_utt_len[dialogue_ids[b]]):
                pred1 = torch.unsqueeze(outputs[0][b][s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
                y_pred1.append(pred_emo.item())
                y_true1.append(truth1.item())

                loss1 += criterion1(pred1,truth1)
                
                curr_speaker = final_speaker_info[dialogue_ids[b]][s]
                curr_emotion = final_speaker_emotions[dialogue_ids[b]][s]
                
                flip_cnt = 0
                
                if curr_speaker not in current_speaker_emo.keys():
                    current_speaker_emo[curr_speaker] = curr_emotion
                else:
                    prev_emotion = current_speaker_emo[curr_speaker]
                    current_speaker_emo[curr_speaker] = curr_emotion
                    if prev_emotion != curr_emotion:
                        flip_cnt += 1
                        pred2 = outputs[1][b][s]                       
                        truth2 = targets2[b][s]

                        if s < 5:
                            r = s+1
                        else:
                            r = 5
                            
                        for s2 in range(r):
                          pred_flip = torch.argmax(F.softmax(pred2[s2].to(device),-1),-1)
                          pred2_ = torch.unsqueeze(pred2[s2].to(device),dim=0)
                            
                          if s < 5:
                            truth_flip = truth2[s2].long().to(device)
                            truth2_ = torch.unsqueeze(truth2[s2].long().to(device),dim=0)
                          else:
                            truth_flip = truth2[s-5+s2+1].long().to(device)
                            truth2_ = torch.unsqueeze(truth2[s-5+s2+1].long().to(device),dim=0)

                          y_pred2.append(pred_flip.item())
                          y_true2.append(truth_flip.item())

                          loss2 += criterion2(pred2_,truth2_)
                        loss2 /= len(pred2)
              loss1 /= final_utt_len[dialogue_ids[b]]
              if flip_cnt == 0:
                    flip_cnt = 1
              loss2 /= flip_cnt
                
              if freeze:
                loss += loss2
              else:
                loss += (loss1_wt*loss1+loss2_wt*loss2)/2                
            loss /= outputs[0].size()[0]
            avg_loss += loss

            loss.backward()            
            optimizer.step()
            
        avg_loss /= len(train_data_loader)
        print("Average Loss = ",avg_loss)
        
        f1_1, f1_2_cls, v_loss = validate(model,data_iter_test,epoch)
        
        ##Optimizing for emotion detection
        # if f1_1 > max_f1_1:
        #     print(f"Saving model at epoch {epoch}")
        #     max_f1_1 = f1_1
        #     torch.save(model.state_dict(), "./model_{}_weight_state_dict.pth".format(model_name))

    return model

def validate(model, test_data_loader, epoch):
    print("\n\n***VALIDATION ({})***\n\n".format(epoch))
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none')
    
    class_weights2 = torch.FloatTensor(weights2).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights2,reduction='none')

    model.eval()

    with torch.no_grad():
      avg_loss = 0
        
      y_true1 = []
      y_pred1 = []

      y_true2 = []
      y_pred2 = []
        
      y_true2_flip = []
      y_pred2_flip = []

      for i_batch, sample_batched in tqdm.tqdm(enumerate(test_data_loader)):
            dialogue_ids = sample_batched[0].tolist()
            dialogue_ids = [train_cnt+d for d in dialogue_ids]
            inputs = sample_batched[1].to(device)
            targets1 = sample_batched[2].to(device)
            targets2 = sample_batched[3].to(device)
                        
            outputs = model(dialogue_ids,final_speaker_info,final_speaker_dialogues,final_speaker_emotions,final_speaker_indices,True,inputs,mode="valid")
            
            loss = 0
            for b in range(outputs[0].size()[0]):
              loss1 = 0
              loss2 = 0
              current_speaker_emo = {}
              for s in range(final_utt_len[dialogue_ids[b]]):
                pred1 = torch.unsqueeze(outputs[0][b][s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
                y_pred1.append(pred_emo.item())
                y_true1.append(truth1.item())

                loss1 += criterion1(pred1,truth1)
                
                curr_speaker = final_speaker_info[dialogue_ids[b]][s]
                curr_emotion = final_speaker_emotions[dialogue_ids[b]][s]
                
                flip_cnt = 0
                
                if curr_speaker not in current_speaker_emo.keys():
                    current_speaker_emo[curr_speaker] = curr_emotion
                else:
                    prev_emotion = current_speaker_emo[curr_speaker]
                    current_speaker_emo[curr_speaker] = curr_emotion
                    if prev_emotion != curr_emotion:
                        flip_cnt += 1
                        pred2 = outputs[1][b][s]                       
                        truth2 = targets2[b][s]

                        if s < 5:
                            r = s+1
                        else:
                            r = 5
                            
                        for s2 in range(r):
                          pred_flip = torch.argmax(F.softmax(pred2[s2].to(device),-1),-1)
                          pred2_ = torch.unsqueeze(pred2[s2].to(device),dim=0)
                            
                          if s < 5:
                            truth_flip = truth2[s2].long().to(device)
                            truth2_ = torch.unsqueeze(truth2[s2].long().to(device),dim=0)
                          else:
                            truth_flip = truth2[s-5+s2+1].long().to(device)
                            truth2_ = torch.unsqueeze(truth2[s-5+s2+1].long().to(device),dim=0)

                          y_pred2_flip.append(pred_flip.item())
                          y_true2_flip.append(truth_flip.item())

                if s != 0:
                    pred2 = outputs[1][b][s]
                    truth2 = targets2[b][s]
                    
                    if s < 5:
                        r = s+1
                    else:
                        r = 5

                    for s2 in range(r):
                        pred_flip = torch.argmax(F.softmax(pred2[s2].to(device),-1),-1)
                        pred2_ = torch.unsqueeze(pred2[s2].to(device),dim=0)
                            
                        if s < 5:
                          truth_flip = truth2[s2].long().to(device)
                          truth2_ = torch.unsqueeze(truth2[s2].long().to(device),dim=0)
                        else:
                          truth_flip = truth2[s-5+s2+1].long().to(device)
                          truth2_ = torch.unsqueeze(truth2[s-5+s2+1].long().to(device),dim=0)

                        y_pred2.append(pred_flip.item())
                        y_true2.append(truth_flip.item())

                        loss2 += criterion2(pred2_,truth2_)
                    loss2 /= len(pred2)
              loss1 /= final_utt_len[dialogue_ids[b]]
              loss2 /= final_utt_len[dialogue_ids[b]]
              
              loss += (loss1_wt*loss1+loss2_wt*loss2)/2
            loss /= outputs[0].size()[0]
            avg_loss += loss

      avg_loss /= len(test_data_loader)

      class_report1 = classification_report(y_true1,y_pred1)
      class_report2 = classification_report(y_true2,y_pred2)
      class_report3 = classification_report(y_true2_flip,y_pred2_flip)

      wtd_f11 = f1_score(y_true1,y_pred1,average="weighted",zero_division=0)

      conf_mat1 = confusion_matrix(y_true1,y_pred1)
      conf_mat2 = confusion_matrix(y_true2,y_pred2)
      conf_mat3 = confusion_matrix(y_true2_flip,y_pred2_flip)

      print("ERC:-\n",class_report1)
      print("EFR:-\n",class_report3)

      return wtd_f11, avg_loss
    
model = ERC_EFR_multitask(hidden_size,weight_matrix,utt2idx,batch_size,seq_len).to(device)
weights1 = [1.0]*7
weights2 = [1.0,2.5]

loss1_wt = 1    #weighting to first ojective (ERC)
loss2_wt = 1    ##weighting to second ojective (EFR)

data_iter_train, data_iter_test = get_train_test_loader(batch_size)
model = train(model, data_iter_train, epochs = 100)