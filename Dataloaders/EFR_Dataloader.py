import pandas as pd
import numpy as np
import torch
import pickle
import tqdm
import random
import nlp_utils as nu
from collections import Counter
from torchtext import datasets
from torch.utils import data

pickle_path = "../Pickles/"
our_training_path = "../Data/meld-fr_partial_train.csv"
our_testing_path = "../Data/meld-fr_partial_test.csv"
save_path = pickle_path

our_training_csv = pd.read_csv(our_training_path)
our_testing_csv = pd.read_csv(our_testing_path)

with open(pickle_path+"idx2utt.pickle","rb") as f:
    idx2utt = pickle.load(f)
with open(pickle_path+"utt2idx.pickle","rb") as f:
    utt2idx = pickle.load(f)
    
with open(pickle_path+"idx2emo.pickle","rb") as f:
    idx2emo = pickle.load(f)
with open(pickle_path+"emo2idx.pickle","rb") as f:
    emo2idx = pickle.load(f)
    
with open(pickle_path+"idx2speaker.pickle","rb") as f:
    idx2speaker = pickle.load(f)
with open(pickle_path+"speaker2idx.pickle","rb") as f:
    speaker2idx = pickle.load(f)

batch_size = 8
seq_len = 5
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

##################
### Reading training data
X_train = []
X_train_d_id = []
X_train_emo = []
y_train_flip = []

X_train_tmp = []
X_train_d_id_tmp = []
X_train_emo_tmp = []
y_train_flip_tmp = []

global_speaker_info = {}
speaker_dialogues = {}
speaker_emotions = {}
speaker_indices = {}
utt_len = {}

d_id = 0
c_id = 0
for i in range(len(our_training_csv)):
    if np.isnan(float(our_training_csv["Dialogue_Id"][i])):
        if len(X_train_tmp) > seq_len:
            utt_len[d_id] = seq_len
        else:
            utt_len[d_id] = len(X_train_tmp)
          
        if len(X_train_tmp) < seq_len:
            for k in range(len(X_train_tmp),seq_len):
                X_train_tmp.append(utt2idx["<pad>"])
                X_train_d_id_tmp.append(d_id)
                X_train_emo_tmp.append(emo2idx["neutral"])
                y_train_flip_tmp.append(0)
        else:
            diff = len(X_train_tmp)-seq_len
            X_train_tmp = X_train_tmp[diff:]
            X_train_d_id_tmp = X_train_d_id_tmp[diff:]
            X_train_emo_tmp = X_train_emo_tmp[diff:]
            y_train_flip_tmp = y_train_flip_tmp[diff:]
            
        if len(global_speaker_info[d_id].keys()) < seq_len:
            for k in range(len(global_speaker_info[d_id].keys()),seq_len):
                global_speaker_info[d_id][k] = speaker2idx["<pad>"]
                speaker_emotions[d_id][k] = emo2idx["neutral"]
        else:
            tmp_speaker_info = global_speaker_info[d_id].copy()
            tmp_speaker_emotions = speaker_emotions[d_id].copy()
            
            for k_i,k in enumerate(range(diff,len(global_speaker_info[d_id].keys()))):
                global_speaker_info[d_id][k_i] = tmp_speaker_info[k]
                speaker_emotions[d_id][k_i] = tmp_speaker_emotions[k]
        
        for every_sp in speaker_dialogues[d_id].keys():
            if len(speaker_dialogues[d_id][every_sp]) < seq_len:
                for k in range(len(speaker_dialogues[d_id][every_sp]),seq_len):
                    speaker_dialogues[d_id][every_sp].append(utt2idx["<pad>"])

        X_train.append(X_train_tmp)
        X_train_d_id.append(X_train_d_id_tmp)
        X_train_emo.append(X_train_emo_tmp)
        y_train_flip.append(y_train_flip_tmp)
        
        X_train_tmp = []
        X_train_d_id_tmp = []
        X_train_emo_tmp = []
        y_train_flip_tmp = []

        d_id += 1
        c_id = 0
    else:
        if d_id not in global_speaker_info.keys():
            global_speaker_info[d_id] = {}
            speaker_dialogues[d_id] = {}
            speaker_emotions[d_id] = {}
            speaker_indices[d_id] = {}

        utt = utt2idx[nu.preprocess_text(our_training_csv["Utterance"][i])]
        sp = speaker2idx[our_training_csv["Speaker"][i]]
        flip = float(our_training_csv["Annotate(0/1)"][i])
        emo = emo2idx[our_training_csv["Emotion_name"][i]]
        if np.isnan(flip):
            flip = 0
        
        X_train_tmp.append(utt)
        X_train_d_id_tmp.append(d_id)
        X_train_emo_tmp.append(emo)
        y_train_flip_tmp.append(flip)
        
        global_speaker_info[d_id][c_id] = sp
        if sp in speaker_dialogues[d_id].keys():
            speaker_dialogues[d_id][sp].append(utt)
        else:
            speaker_dialogues[d_id][sp] = [utt]
        speaker_emotions[d_id][c_id] = emo
        if sp in speaker_indices[d_id].keys():
            speaker_indices[d_id][sp].append(c_id)
        else:
            speaker_indices[d_id][sp] = [c_id]
        
        c_id += 1
        
D = torch.LongTensor(X_train_d_id)
X = torch.LongTensor(X_train)
E = torch.LongTensor(X_train_emo)
Y = torch.LongTensor(y_train_flip)

my_dataset_train = data.TensorDataset(D,X,E,Y)

##################
### Reading testing data
X_test = []
X_test_d_id = []
X_test_emo = []
y_test_flip = []

X_test_tmp = []
X_test_d_id_tmp = []
X_test_emo_tmp = []
y_test_flip_tmp = []

global_speaker_info_test = {}
speaker_dialogues_test = {}
speaker_emotions_test = {}
speaker_indices_test = {}
utt_len_test = {}

d_id = 0
c_id = 0
for i in range(len(our_testing_csv)):
    if np.isnan(float(our_testing_csv["Dialogue_Id"][i])):
        if len(X_test_tmp) > seq_len:
            utt_len_test[d_id] = seq_len
        else:
            utt_len_test[d_id] = len(X_test_tmp)
          
        if len(X_test_tmp) < seq_len:
            for k in range(len(X_test_tmp),seq_len):
                X_test_tmp.append(utt2idx["<pad>"])
                X_test_d_id_tmp.append(d_id)
                X_test_emo_tmp.append(emo2idx["neutral"])
                y_test_flip_tmp.append(0)
        else:
            diff = len(X_test_tmp)-seq_len
            X_test_tmp = X_test_tmp[diff:]
            X_test_d_id_tmp = X_test_d_id_tmp[diff:]
            X_test_emo_tmp = X_test_emo_tmp[diff:]
            y_test_flip_tmp = y_test_flip_tmp[diff:]
            
        if len(global_speaker_info_test[d_id].keys()) < seq_len:
            for k in range(len(global_speaker_info_test[d_id].keys()),seq_len):
                global_speaker_info_test[d_id][k] = speaker2idx["<pad>"]
                speaker_emotions_test[d_id][k] = emo2idx["neutral"]
        else:
            tmp_speaker_info = global_speaker_info_test[d_id].copy()
            tmp_speaker_emotions = speaker_emotions_test[d_id].copy()
            
            for k_i,k in enumerate(range(diff,len(global_speaker_info_test[d_id].keys()))):
                global_speaker_info_test[d_id][k_i] = tmp_speaker_info[k]
                speaker_emotions_test[d_id][k_i] = tmp_speaker_emotions[k]
        
        for every_sp in speaker_dialogues_test[d_id].keys():
            if len(speaker_dialogues_test[d_id][every_sp]) < seq_len:
                for k in range(len(speaker_dialogues_test[d_id][every_sp]),seq_len):
                    speaker_dialogues_test[d_id][every_sp].append(utt2idx["<pad>"])

        X_test.append(X_test_tmp)
        X_test_d_id.append(X_test_d_id_tmp)
        X_test_emo.append(X_test_emo_tmp)
        y_test_flip.append(y_test_flip_tmp)
        
        X_test_tmp = []
        X_test_d_id_tmp = []
        X_test_emo_tmp = []
        y_test_flip_tmp = []

        d_id += 1
        c_id = 0
    else:
        if d_id not in global_speaker_info_test.keys():
            global_speaker_info_test[d_id] = {}
            speaker_dialogues_test[d_id] = {}
            speaker_emotions_test[d_id] = {}
            speaker_indices_test[d_id] = {}

        utt = utt2idx[nu.preprocess_text(our_testing_csv["Utterance"][i])]
        sp = speaker2idx[our_testing_csv["Speaker"][i]]
        flip = float(our_testing_csv["Annotate(0/1)"][i])
        emo = emo2idx[our_testing_csv["Emotion_name"][i]]
        if np.isnan(flip):
            flip = 0
        
        X_test_tmp.append(utt)
        X_test_d_id_tmp.append(d_id)
        X_test_emo_tmp.append(emo)
        y_test_flip_tmp.append(flip)
        
        global_speaker_info_test[d_id][c_id] = sp
        if sp in speaker_dialogues_test[d_id].keys():
            speaker_dialogues_test[d_id][sp].append(utt)
        else:
            speaker_dialogues_test[d_id][sp] = [utt]
        speaker_emotions_test[d_id][c_id] = emo
        if sp in speaker_indices_test[d_id].keys():
            speaker_indices_test[d_id][sp].append(c_id)
        else:
            speaker_indices_test[d_id][sp] = [c_id]
        
        c_id += 1
        
D = torch.LongTensor(X_test_d_id)
X = torch.LongTensor(X_test)
E = torch.LongTensor(X_test_emo)
Y = torch.LongTensor(y_test_flip)

my_dataset_test = data.TensorDataset(D,X,E,Y)

##################
### Saving everything
with open(save_path+"train_data_trig.pickle","wb") as f:
    pickle.dump(my_dataset_train,f)

with open(save_path+"test_data_trig.pickle","wb") as f:
    pickle.dump(my_dataset_test,f)
        
with open(save_path+"global_speaker_info_trig.pickle","wb") as f:
    pickle.dump(global_speaker_info,f)
    
with open(save_path+"speaker_dialogues_trig.pickle","wb") as f:
    pickle.dump(speaker_dialogues,f)
    
with open(save_path+"speaker_emotions_trig.pickle","wb") as f:
    pickle.dump(speaker_emotions,f)
    
with open(save_path+"speaker_indices_trig.pickle","wb") as f:
    pickle.dump(speaker_indices,f)
    
with open(save_path+"utt_len_trig.pickle","wb") as f:
    pickle.dump(utt_len,f)
    
    
with open(save_path+"global_speaker_info_test_trig.pickle","wb") as f:
    pickle.dump(global_speaker_info_test,f)
    
with open(save_path+"speaker_dialogues_test_trig.pickle","wb") as f:
    pickle.dump(speaker_dialogues_test,f)
    
with open(save_path+"speaker_emotions_test_trig.pickle","wb") as f:
    pickle.dump(speaker_emotions_test,f)
    
with open(save_path+"speaker_indices_test_trig.pickle","wb") as f:
    pickle.dump(speaker_indices_test,f)
    
with open(save_path+"utt_len_test_trig.pickle","wb") as f:
    pickle.dump(utt_len_test,f)