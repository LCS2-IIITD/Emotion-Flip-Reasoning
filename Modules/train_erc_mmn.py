import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from models import ERC_MMN
from pickle_loader import load_erc

batch_size = 8
seq_len = 15
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
    speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
    final_speaker_info, final_speaker_dialogues, final_speaker_emotions,\
    final_speaker_indices, final_utt_len = load_erc()

weight_matrix = weight_matrix.to(device)
train_cnt = len(my_dataset_train)

def get_train_test_loader(bs):
    train_data_iter = data.DataLoader(my_dataset_train,batch_size=bs)
    test_data_iter = data.DataLoader(my_dataset_test,batch_size=bs,drop_last=True)
    
    return train_data_iter, test_data_iter

def train(model, train_data_loader, epochs):
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none').to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)    
    max_f1_1 = 0
    
    for epoch in tqdm.tqdm(range(epochs)):
        print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
        model.train()
        
        avg_loss = 0
        
        y_true1 = []
        y_pred1 = []
            
        for i_batch, sample_batched in tqdm.tqdm(enumerate(train_data_loader)):
            dialogue_ids = sample_batched[0].tolist()            
            inputs = sample_batched[1].to(device)
            targets1 = sample_batched[2].to(device)
                 
            optimizer.zero_grad()
            
            _, outputs = model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs)
            
            loss = 0
            for b in range(outputs.size()[0]):
              loss1 = 0
              for s in range(final_utt_len[dialogue_ids[b]]):
                pred1 = torch.unsqueeze(outputs[b][s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
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
        f1_1,v_loss = validate(model,data_iter_test,epoch)
        
        # if f1_1 > max_f1_1:
        #     print(f"Saving model at epoch {epoch}")
        #     max_f1_1 = f1_1
        #     torch.save(model.state_dict(), "./best_model.pth".format(model_name))

    return model

def validate(model, test_data_loader,epoch):
    print("\n\n***VALIDATION ({})***\n\n".format(epoch))
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none')
    
    model.eval()

    with torch.no_grad():
      avg_loss = 0
        
      y_true1 = []
      y_pred1 = []

      for i_batch, sample_batched in tqdm.tqdm(enumerate(test_data_loader)):
            dialogue_ids = sample_batched[0].tolist()
            dialogue_ids = [train_cnt+d for d in dialogue_ids]
            inputs = sample_batched[1].to(device)
            targets1 = sample_batched[2].to(device)

            _, outputs = model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs, mode="valid")
            
            loss = 0
            for b in range(outputs.size()[0]):
              loss1 = 0
              for s in range(final_utt_len[dialogue_ids[b]]):
                pred1 = torch.unsqueeze(outputs[b][s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
                y_pred1.append(pred_emo.item())
                y_true1.append(truth1.item())

                loss1 += criterion1(pred1,truth1)

              loss1 /= final_utt_len[dialogue_ids[b]]
              loss += loss1

            loss /= batch_size
            avg_loss += loss

      avg_loss /= len(test_data_loader)

      class_report = classification_report(y_true1,y_pred1)
      conf_mat1 = confusion_matrix(y_true1,y_pred1)

      print(class_report)
      print("Confusion Matrix: \n",conf_mat1)
    
      wtd_f1 = f1_score(y_true1,y_pred1,average="weighted")
      return wtd_f1, avg_loss
    
model = ERC_MMN(hidden_size,weight_matrix,utt2idx,batch_size,seq_len).to(device)
weights1 = [1.0]*7
data_iter_train, data_iter_test = get_train_test_loader(batch_size)

model = train(model, data_iter_train, epochs = 100)