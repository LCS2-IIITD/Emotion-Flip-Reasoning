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

our_training_path = "../Data/meld-fr_partial_train.csv"
our_testing_path = "../Data/meld-fr_partial_test.csv"
save_path = "../Pickles/"

our_training_csv = pd.read_csv(our_training_path)
our_testing_csv = pd.read_csv(our_testing_path)


##################
### Making utt2idx, emo2idx, speaker2idx, etc dicts
our_utterances = our_training_csv["Utterance"]
our_utterances_test = our_testing_csv["Utterance"]

all_utterances = our_training_csv["Utterance"]
all_utterances = all_utterances.append(our_testing_csv["Utterance"])

all_emotions = our_training_csv["Emotion_name"]
all_emotions = all_emotions.append(our_testing_csv["Emotion_name"])

all_speakers = our_training_csv["Speaker"]
all_speakers = all_speakers.append(our_testing_csv["Speaker"])

all_annotations = our_training_csv["Annotate(0/1)"]
all_annotations = all_annotations.append(our_testing_csv["Annotate(0/1)"])

f_utterances = ["<pad>"]
f_emotions = []
f_speakers = ["<pad>"]
f_annots = []

for utt,emo,sp,ann in zip(all_utterances,all_emotions,all_speakers,all_annotations):
    if type(utt) == str:
        f_utterances.append(nu.preprocess_text(utt))
        f_emotions.append(emo)
        f_speakers.append(sp)
        if np.isnan(float(ann)):
            ann = 0
        f_annots.append(ann)
    
utts = list(set(f_utterances))
idx2utt = {}
utt2idx = {}
ctr = 1

for utt in utts:
  idx2utt[ctr] = utt
  utt2idx[utt] = ctr
  ctr += 1

emos = ['disgust', 'joy', 'surprise', 'anger', 'fear', 'neutral', 'sadness']
idx2emo = {}
emo2idx = {}
ctr = 0

for emo in emos:
  idx2emo[ctr] = emo
  emo2idx[emo] = ctr
  ctr += 1

speaks = list(set(f_speakers))
idx2speaker = {}
speaker2idx = {}
ctr = 0

for speak in speaks:
  idx2speaker[ctr] = speak
  speaker2idx[speak] = ctr
  ctr += 1

##################
### Making weight matrix
with open('../Pickles/sent2emb.pickle','rb') as f:
    sent2emb = pickle.load(f)

batch_size = 8
seq_len = 15
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

matrix_len = len(idx2utt)+1
weight_matrix = np.zeros((matrix_len, hidden_size))

for utt in idx2utt.values():
    pp_utt = nu.preprocess_text(utt)
    if pp_utt != "pad":
      weight_matrix[utt2idx[pp_utt]] = sent2emb[pp_utt].cpu()

weight_matrix = torch.Tensor(weight_matrix)

##################
### Reading training data
df_train = pd.read_csv(our_training_path)

prev_idx = 0
prev_d_id = 0
d_id = 0
annot_tmp = {}
annot = {}
final_annot = []

X_train = []
X_train_d_id = []
y_train_emo = []
y_train_flip = []
y_train_emo_lvl1 = []

global_speaker_info = {}
speaker_dialogues = {}
speaker_emotions = {}
speaker_indices = {}
utt_len = {}

ch_id = 0
for i in tqdm.tqdm(range(len(df_train))):
  if np.isnan(float(df_train["Dialogue_Id"][i])):
    sorted_dict = sorted(annot_tmp.items())
    last_ch_id = sorted_dict[-1][0]
    annot_vals = [item[1] for item in sorted_dict]
    
    annot[last_ch_id] = annot_vals
    annot_tmp = {}
    ch_id = 0

  elif df_train["Dialogue_Id"][i] != prev_d_id:
    prev_d_id = df_train["Dialogue_Id"][i]

    dialogue = []
    emotion = []
    emotion_lvl1 = []
    
    curr_idx = i-1
    for j in range(curr_idx-1,0,-1):
      if np.isnan(float(df_train["Dialogue_Id"][j])):
        prev_idx = j+1
        break

    this_d_id = df_train["Dialogue_Id"][prev_idx]
    global_speaker_info[int(this_d_id)] = {}
    speaker_dialogues[int(this_d_id)] = {}
    speaker_emotions[int(this_d_id)] = {}
    speaker_indices[int(this_d_id)] = {}
    
    chat_id = 0
    
    for j in range(prev_idx,curr_idx):
      utt = nu.preprocess_text(df_train["Utterance"][j])
      sp = df_train["Speaker"][j]
      emo = df_train["Emotion_name"][j]

      dialogue.append(utt2idx[utt])
      emotion.append(emo2idx[emo])
      
      if emo == "neutral":
        emotion_lvl1.append(1)
      else:
        emotion_lvl1.append(0)
            
      speaker_emotions[int(this_d_id)][chat_id] = emo2idx[emo]
      
      if speaker2idx[sp] in speaker_indices[int(this_d_id)].keys():
        speaker_indices[int(this_d_id)][speaker2idx[sp]].append(chat_id)
      else:
        speaker_indices[int(this_d_id)][speaker2idx[sp]] = [chat_id]

      if chat_id < seq_len:
        global_speaker_info[int(this_d_id)][chat_id] = speaker2idx[sp]

      if speaker2idx[sp] in speaker_dialogues[int(this_d_id)].keys():
        speaker_dialogues[int(this_d_id)][speaker2idx[sp]].append(utt2idx[utt])
      else:
        speaker_dialogues[int(this_d_id)][speaker2idx[sp]] = [utt2idx[utt]]
      
      if chat_id in annot.keys():
        final_annot.append(annot[chat_id])
      else:
        final_annot.append([0]*seq_len)
        
      chat_id += 1
      
    if len(dialogue) > seq_len:
      utt_len[d_id] = seq_len
    else:
      utt_len[d_id] = len(dialogue)

    if len(speaker_emotions[int(this_d_id)].keys()) < seq_len:
      for k in range(len(speaker_emotions[int(this_d_id)].keys()),seq_len):
        speaker_emotions[int(this_d_id)][k] = emo2idx["neutral"]
    
    if len(dialogue) < seq_len:
      for k in range(len(dialogue),seq_len):
        dialogue.append(utt2idx["<pad>"])
    elif len(dialogue) > seq_len:
      dialogue = dialogue[:seq_len]

    if len(emotion) < seq_len:
      for k in range(len(emotion),seq_len):
        emotion.append(emo2idx["neutral"])
    elif len(emotion) > seq_len:
      emotion = emotion[:seq_len]
    
    if len(emotion_lvl1) < seq_len:
      for k in range(len(emotion_lvl1),seq_len):
        emotion_lvl1.append(1)
    elif len(emotion_lvl1) > seq_len:
      emotion_lvl1 = emotion_lvl1[:seq_len]

    if len(final_annot) < seq_len:
      for k in range(len(final_annot),seq_len):
        final_annot.append([0]*seq_len)
    elif len(final_annot) > seq_len:
      final_annot = final_annot[:seq_len]

    for an_i,an in enumerate(final_annot):
      if len(an) < seq_len:
        for k in range(len(an),seq_len):
          final_annot[an_i].append(0)
      elif len(an) > seq_len:
        final_annot[an_i] = final_annot[an_i][:seq_len]

    if chat_id < seq_len:
      for k in range(chat_id,seq_len):
        global_speaker_info[int(this_d_id)][k] = speaker2idx["<pad>"]
      
    for sp in speaker_dialogues[int(this_d_id)].keys():
      if len(speaker_dialogues[int(this_d_id)][sp]) < seq2_len:
        for k in range(len(speaker_dialogues[int(this_d_id)][sp]),seq2_len):
          speaker_dialogues[int(this_d_id)][sp].append(utt2idx["<pad>"])
      elif len(speaker_dialogues[int(this_d_id)][sp]) > seq2_len:
        speaker_dialogues[int(this_d_id)][sp] = speaker_dialogues[int(this_d_id)][sp][:seq2_len]

    X_train.append(dialogue)
    y_train_emo.append(emotion)
    y_train_emo_lvl1.append(emotion_lvl1)
    y_train_flip.append(final_annot)
    X_train_d_id.append(d_id)
    d_id += 1

    annot = {}
    final_annot = []
    prev_idx = int(df_train["Dialogue_Id"][i])
  else:
    a = df_train["Annotate(0/1)"][i]
    if np.isnan(a):
      a = 0
    annot_tmp[ch_id] = a
    ch_id += 1
    
D = torch.LongTensor(X_train_d_id)
X = torch.LongTensor(X_train)
Y1 = torch.LongTensor(y_train_emo)
Y2 = torch.LongTensor(y_train_flip)
Y3 = torch.LongTensor(y_train_emo_lvl1)

my_dataset_train = data.TensorDataset(D,X,Y1,Y2,Y3)

##################
### Reading testing data
df_test = pd.read_csv(our_testing_path)

prev_idx = 0
prev_d_id = 0
d_id = 0
annot_tmp = {}
annot = {}
final_annot = []

X_test = []
X_test_d_id = []
y_test_emo = []
y_test_flip = []
y_test_emo_lvl1 = []

global_speaker_info_test = {}
speaker_dialogues_test = {}
speaker_emotions_test = {}
speaker_indices_test = {}
utt_len_test = {}

ch_id = 0
for i in tqdm.tqdm(range(len(df_test))):
  if np.isnan(float(df_test["Dialogue_Id"][i])):
    sorted_dict = sorted(annot_tmp.items())
    last_ch_id = sorted_dict[-1][0]
    annot_vals = [item[1] for item in sorted_dict]
    
    annot[last_ch_id] = annot_vals
    annot_tmp = {}
    ch_id = 0

  elif df_test["Dialogue_Id"][i] != prev_d_id:
    prev_d_id = df_test["Dialogue_Id"][i]

    dialogue = []
    emotion = []
    emotion_lvl1 = []
    
    curr_idx = i-1
    for j in range(curr_idx-1,0,-1):
      if np.isnan(float(df_test["Dialogue_Id"][j])):
        prev_idx = j+1
        break

    this_d_id = df_test["Dialogue_Id"][prev_idx]
    global_speaker_info_test[int(this_d_id)] = {}
    speaker_dialogues_test[int(this_d_id)] = {}
    speaker_emotions_test[int(this_d_id)] = {}
    speaker_indices_test[int(this_d_id)] = {}
    
    chat_id = 0
    
    for j in range(prev_idx,curr_idx):
      utt = nu.preprocess_text(df_test["Utterance"][j])
      sp = df_test["Speaker"][j]
      emo = df_test["Emotion_name"][j]

      dialogue.append(utt2idx[utt])
      emotion.append(emo2idx[emo])
    
      if emo == "neutral":
        emotion_lvl1.append(1)
      else:
        emotion_lvl1.append(0)
        
      speaker_emotions_test[int(this_d_id)][chat_id] = emo2idx[emo]
        
      if speaker2idx[sp] in speaker_indices_test[int(this_d_id)].keys():
        speaker_indices_test[int(this_d_id)][speaker2idx[sp]].append(chat_id)
      else:
        speaker_indices_test[int(this_d_id)][speaker2idx[sp]] = [chat_id]

      if chat_id < seq_len:
        global_speaker_info_test[int(this_d_id)][chat_id] = speaker2idx[sp]

      if speaker2idx[sp] in speaker_dialogues_test[int(this_d_id)].keys():
        speaker_dialogues_test[int(this_d_id)][speaker2idx[sp]].append(utt2idx[utt])
      else:
        speaker_dialogues_test[int(this_d_id)][speaker2idx[sp]] = [utt2idx[utt]]
      
      if chat_id in annot.keys():
        final_annot.append(annot[chat_id])
      else:
        final_annot.append([0]*seq_len)
        
      chat_id += 1

    if len(dialogue) > seq_len:
      utt_len_test[d_id] = seq_len
    else:
      utt_len_test[d_id] = len(dialogue)
      
    if len(speaker_emotions_test[int(this_d_id)].keys()) < seq_len:
      for k in range(len(speaker_emotions_test[int(this_d_id)].keys()),seq_len):
        speaker_emotions_test[int(this_d_id)][k] = emo2idx["neutral"]
        
    if len(dialogue) < seq_len:
      for k in range(len(dialogue),seq_len):
        dialogue.append(utt2idx["<pad>"])
    elif len(dialogue) > seq_len:
      dialogue = dialogue[:seq_len]

    if len(emotion) < seq_len:
      for k in range(len(emotion),seq_len):
        emotion.append(emo2idx["neutral"])
    elif len(emotion) > seq_len:
      emotion = emotion[:seq_len]
    
    if len(emotion_lvl1) < seq_len:
      for k in range(len(emotion_lvl1),seq_len):
        emotion_lvl1.append(1)
    elif len(emotion_lvl1) > seq_len:
      emotion_lvl1 = emotion_lvl1[:seq_len]

    if len(final_annot) < seq_len:
      for k in range(len(final_annot),seq_len):
        final_annot.append([0]*seq_len)
    elif len(final_annot) > seq_len:
      final_annot = final_annot[:seq_len]

    for an_i,an in enumerate(final_annot):
      if len(an) < seq_len:
        for k in range(len(an),seq_len):
          final_annot[an_i].append(0)
      elif len(an) > seq_len:
        final_annot[an_i] = final_annot[an_i][:seq_len]

    if chat_id < seq_len:
      for k in range(chat_id,seq_len):
        global_speaker_info_test[int(this_d_id)][k] = speaker2idx["<pad>"]
      
    for sp in speaker_dialogues_test[int(this_d_id)].keys():
      if len(speaker_dialogues_test[int(this_d_id)][sp]) < seq_len:
        for k in range(len(speaker_dialogues_test[int(this_d_id)][sp]),seq_len):
          speaker_dialogues_test[int(this_d_id)][sp].append(utt2idx["<pad>"])
      elif len(speaker_dialogues_test[int(this_d_id)][sp]) > seq_len:
        speaker_dialogues_test[int(this_d_id)][sp] = speaker_dialogues_test[int(this_d_id)][sp][:seq_len]

    X_test.append(dialogue)
    y_test_emo.append(emotion)
    y_test_emo_lvl1.append(emotion_lvl1)
    y_test_flip.append(final_annot)
    X_test_d_id.append(d_id)
    d_id += 1

    annot = {}
    final_annot = []
    prev_idx = int(df_test["Dialogue_Id"][i])
  else:
    a = df_test["Annotate(0/1)"][i]
    if np.isnan(a):
      a = 0
    annot_tmp[ch_id] = a
    ch_id += 1

D = torch.LongTensor(X_test_d_id)
X = torch.LongTensor(X_test)
Y1 = torch.LongTensor(y_test_emo)
Y2 = torch.LongTensor(y_test_flip)
Y3 = torch.LongTensor(y_test_emo_lvl1)

my_dataset_test = data.TensorDataset(D,X,Y1,Y2,Y3)

##################
X_data =[]
y_data = []
my_data = []

X_d_id =[]
X = []
y_emo = []
y_flip = []
y_emo_lvl1 = []

final_speaker_info = {}
final_speaker_dialogues = {}
final_speaker_emotions = {}
final_speaker_indices = {}
final_utt_len = {}

d_id = 0
for d,dialogue,emo,trig,n_nn in zip(X_train_d_id,X_train,y_train_emo,y_train_flip,y_train_emo_lvl1):
    final_speaker_info[d] = global_speaker_info[d]
    final_speaker_dialogues[d] = speaker_dialogues[d]
    final_speaker_emotions[d] = speaker_emotions[d]
    final_speaker_indices[d] = speaker_indices[d]
    final_utt_len[d] = utt_len[d]
    
    X_data.append((d,dialogue))
    y_data.append((d,emo,trig,n_nn))
    my_data.append((d,dialogue,emo,trig,n_nn))
    
    X_d_id.append(d)
    X.append(dialogue)
    y_emo.append(emo)
    y_flip.append(trig)
    y_emo_lvl1.append(n_nn)
    d_id += 1

print("d_id -> ",d_id)

for d,dialogue,emo,trig,n_nn in zip(X_test_d_id,X_test,y_test_emo,y_test_flip,y_test_emo_lvl1):
    final_speaker_info[d_id] = global_speaker_info_test[d]
    final_speaker_dialogues[d_id] = speaker_dialogues_test[d]
    final_speaker_emotions[d_id] = speaker_emotions_test[d]
    final_speaker_indices[d_id] = speaker_indices_test[d]
    final_utt_len[d_id] = utt_len_test[d]
    
    X_data.append((d_id,dialogue))
    y_data.append((d_id,emo,trig,n_nn))
    my_data.append((d_id,dialogue,emo,trig,n_nn))
    
    X_d_id.append(d_id)
    X.append(dialogue)
    y_emo.append(emo)
    y_flip.append(trig)
    y_emo_lvl1.append(n_nn)
    d_id += 1

##################
### Saving everything
with open(save_path+"idx2utt.pickle","wb") as f:
    pickle.dump(idx2utt,f)
    
with open(save_path+"utt2idx.pickle","wb") as f:
    pickle.dump(utt2idx,f)
    
with open(save_path+"idx2emo.pickle","wb") as f:
    pickle.dump(idx2emo,f)
    
with open(save_path+"emo2idx.pickle","wb") as f:
    pickle.dump(emo2idx,f)
    
with open(save_path+"idx2speaker.pickle","wb") as f:
    pickle.dump(idx2speaker,f)
    
with open(save_path+"speaker2idx.pickle","wb") as f:
    pickle.dump(speaker2idx,f)
    

with open(save_path+"weight_matrix.pickle","wb") as f:
    pickle.dump(weight_matrix,f)
    
    
with open(save_path+"test_data.pickle","wb") as f:
    pickle.dump(my_dataset_test,f)

with open(save_path+"train_data.pickle","wb") as f:
    pickle.dump(my_dataset_train,f)
    
    
with open(save_path+"final_speaker_info.pickle","wb") as f:
    pickle.dump(final_speaker_info,f)
    
with open(save_path+"final_speaker_dialogues.pickle","wb") as f:
    pickle.dump(final_speaker_dialogues,f)
    
with open(save_path+"final_speaker_emotions.pickle","wb") as f:
    pickle.dump(final_speaker_emotions,f)
    
with open(save_path+"final_speaker_indices.pickle","wb") as f:
    pickle.dump(final_speaker_indices,f)
    
with open(save_path+"final_utt_len.pickle","wb") as f:
    pickle.dump(final_utt_len,f)