# Discovering Emotion and Reasoning its Flip in Multi-Party Conversations using Masked Memory Networks and Transformer

This folder contains the code required to reproduce the results of the paper - **"Discovering Emotion and Reasoning its Flip in Multi-Party Conversations using Masked Memory Networks and Transformer"**.

The paper introduces a novel problem, called **Emotion-Flip Reasoning** aka **EFR** in conversations. The goal is to find all utterances that trigger a flip in emotion of a speaker within a dialog.
An example scenario is depicted in the figure below:

![EFR Example](/imgs/efr-eg-2.png "EFR Example")

There is a single emotion-flip, i.e., *u_6 (neutral) -> u_8 (fear)*. The responsible trigger utterances are marked with arrows in the figure.

We augment [MELD](https://affective-meld.github.io/), a benchmark ERC dataset with ground-truth EFR labels. The resultant dataset, called **MELD-FR**, contains ~8,500 trigger utterances for ~5,400 emotion-flips.

## Contribution
In this work, our major contributions are four-fold: 
1. We propose a novel task, called emotion-flip reasoning (EFR), in the conversation dialog.
2. We develop a new ground-truth dataset for EFR, called MELD-FR.
3. We benchmark MELD-FR through a Transformer-based model and present a strong baseline for the EFR task.
4. We develop a masked memory network based architecture for ERC, which outperforms several recent baselines.

## Dataset
We provide the dataset along with the code (*"Data/MELD_\<set\>_efr.csv"*). You can use your own dataset if you follow the expected format for the data.
The models in this repo expects the data to follow the following structure:
| Dialogue Id | Speaker | Emotion | Utterance | EFR Label |
|-------------|---------|---------|-----------|-----------|

## Models
![Model Architecture](/imgs/architecture.png "Model Architecture for ERC and EFR tasks")

### Emotion Recognition in Conversations (ERC)
We employ a memory network to supplement the global and local emotion dynamics in dialogues captured through a series of recurrent layers. In the repo we provide with 3 models that perform the ERC task.
- <code>ERC-MMN</code>
- <code>EFR-ERC<sub>multi</sub></code>
- <code>EFR->ERC<sub>cas</sub></code> (Best Performance) 

### Emotion Flip Reasoning (EFR)
We employ a transformer-based model to model the EFR task. We provide 4 models to perform the ERC task in this repo.
- <code>EFR-TX</code>
- <code>EFR-ERC<sub>multi</sub></code>
- <code>ERC->EFR<sub>cas</sub></code>
- <code>ERC<sup>True</sup>->EFR<sub>cas</sub></code> (Best Performance)

## Code Reproducibility
The results reported in the paper are achieved on the full dataset and hence would not be achieved by the dataset provided in this folder. However you can run the codes by following these steps:
1. Run the Dataloaders for ERC and EFR tasks respectively to get the necessary pickle files.
2. Run the train_*model* file where *model* can be one of "erc_mmn", "efr_tx", "multitask", "cascade_efr_erc", "cascade_erc_efr" and "cascade_ercT_efr".

### Running the code for custom data
Since our model uses [BERT](https://arxiv.org/pdf/1810.04805.pdf) embeddings as utterance reprsentation, first you'll need to generate these BERT embeddings for your data. After you have a dictionary mapping for your utterances to embeddings, you'll need to change the paths in the ERC and EFR dataloaders. Specifically, the paths for training file, testing file and embedding files needs to be changed. After executing the teh dataloaders you'll get the necessary pickle files to run the models.
