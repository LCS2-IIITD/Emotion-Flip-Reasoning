import pickle

def load_erc():
    with open("../Pickles/idx2utt.pickle","rb") as f:
        idx2utt = pickle.load(f)
    with open("../Pickles/utt2idx.pickle","rb") as f:
        utt2idx = pickle.load(f)
        
    with open("../Pickles/idx2emo.pickle","rb") as f:
        idx2emo = pickle.load(f)
    with open("../Pickles/emo2idx.pickle","rb") as f:
        emo2idx = pickle.load(f)
        
    with open("../Pickles/idx2speaker.pickle","rb") as f:
        idx2speaker = pickle.load(f)
    with open("../Pickles/speaker2idx.pickle","rb") as f:
        speaker2idx = pickle.load(f)

    with open("../Pickles/weight_matrix.pickle","rb") as f:
        weight_matrix = pickle.load(f)

    with open("../Pickles/train_data.pickle","rb") as f:
        my_dataset_train = pickle.load(f)
        
    with open("../Pickles/test_data.pickle","rb") as f:
        my_dataset_test = pickle.load(f)
        
    with open("../Pickles/final_speaker_info.pickle","rb") as f:
        final_speaker_info = pickle.load(f)
        
    with open("../Pickles/final_speaker_dialogues.pickle","rb") as f:
        final_speaker_dialogues = pickle.load(f)
        
    with open("../Pickles/final_speaker_emotions.pickle","rb") as f:
        final_speaker_emotions = pickle.load(f)
        
    with open("../Pickles/final_speaker_indices.pickle","rb") as f:
        final_speaker_indices = pickle.load(f)
        
    with open("../Pickles/final_utt_len.pickle","rb") as f:
        final_utt_len = pickle.load(f)

    return idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
        speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
        final_speaker_info, final_speaker_dialogues, final_speaker_emotions,\
        final_speaker_indices, final_utt_len

def load_efr():
    with open("../Pickles/idx2utt.pickle","rb") as f:
        idx2utt = pickle.load(f)
    with open("../Pickles/utt2idx.pickle","rb") as f:
        utt2idx = pickle.load(f)
        
    with open("../Pickles/idx2emo.pickle","rb") as f:
        idx2emo = pickle.load(f)
    with open("../Pickles/emo2idx.pickle","rb") as f:
        emo2idx = pickle.load(f)
        
    with open("../Pickles/idx2speaker.pickle","rb") as f:
        idx2speaker = pickle.load(f)
    with open("../Pickles/speaker2idx.pickle","rb") as f:
        speaker2idx = pickle.load(f)

    with open("../Pickles/weight_matrix.pickle","rb") as f:
        weight_matrix = pickle.load(f)

    with open("../Pickles/train_data_trig.pickle","rb") as f:
        my_dataset_train = pickle.load(f)

    with open("../Pickles/test_data_trig.pickle","rb") as f:
        my_dataset_test = pickle.load(f)
        
    with open("../Pickles/global_speaker_info_trig.pickle","rb") as f:
        global_speaker_info = pickle.load(f)
        
    with open("../Pickles/speaker_dialogues_trig.pickle","rb") as f:
        speaker_dialogues = pickle.load(f)
        
    with open("../Pickles/speaker_emotions_trig.pickle","rb") as f:
        speaker_emotions = pickle.load(f)
        
    with open("../Pickles/speaker_indices_trig.pickle","rb") as f:
        speaker_indices = pickle.load(f)
        
    with open("../Pickles/utt_len_trig.pickle","rb") as f:
        utt_len = pickle.load(f)
        
    with open("../Pickles/global_speaker_info_test_trig.pickle","rb") as f:
        global_speaker_info_test = pickle.load(f)
        
    with open("../Pickles/speaker_dialogues_test_trig.pickle","rb") as f:
        speaker_dialogues_test = pickle.load(f)
        
    with open("../Pickles/speaker_emotions_test_trig.pickle","rb") as f:
        speaker_emotions_test = pickle.load(f)
        
    with open("../Pickles/speaker_indices_test_trig.pickle","rb") as f:
        speaker_indices_test = pickle.load(f)
        
    with open("../Pickles/utt_len_test_trig.pickle","rb") as f:
        utt_len_test = pickle.load(f)

    return idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
        speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
        global_speaker_info, speaker_dialogues, speaker_emotions, \
        speaker_indices, utt_len, global_speaker_info_test, speaker_dialogues_test, \
        speaker_emotions_test, speaker_indices_test, utt_len_test