# Visual Pronoun Coreference Resolution in Dialogues

## Introduction
This is the data and the source code for EMNLP 2019 paper "What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues".

### Abstract
Grounding pronouns to a visual object it refers to requires complex reasoning from various information sources, especially in conversational scenarios.
For example, when people in a conversation talk about something all speakers can see (e.g., <b>the statue</b>), they often directly use pronouns (e.g., <b>it</b>) to refer it without previous introduction.
This fact brings a huge challenge for modern natural language understanding systems, particularly conventional context-based pronoun coreference models.
To tackle this challenge, in this paper, we formally define the task of visual-aware pronoun coreference resolution (PCR), and introduce VisPro, a large-scale dialogue PCR dataset, to investigate whether and how the visual information can help resolve pronouns in dialogues.
We then propose a novel visual-aware PCR model, VisCoref, for this task and conduct comprehensive experiments and case studies on our dataset.
Results demonstrate the importance of the visual information in this PCR case and shows the effectiveness of the proposed model.

<div align=center>
<img width="500" src="fig/dialog_example.PNG">
</div>

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{yu2019visualpcr,
  author    = {Xintong Yu and 
               Hongming Zhang and
               Yangqiu Song and
               Yan Song and
               Changshui Zhang},
  title     = {What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues},
  booktitle = {Proceedings of EMNLP, 2019},
  year      = {2019}
}
```



## VisPro Dataset
The train, val, and test split of VisPro dataset are in `data` directory.

## Usage of VisCoref

### An Example of VisCoref Prediction
<div align=center>
<img width="800" src="fig/case_study1.png">
</div>

### Getting Started
* Install python 3.7 and the following requirements: `pip install -r requirements.txt`. Set default python under your system to python 3.7.
* Download supplementary data for training VisCoref and pretrained model from [Data](https://drive.google.com/open?id=1dSeGz5k57bU2GXCt7sY9krykLvmnbiVx) and extract: `tar -xzvf VisCoref.tar.gz`.
* Move VisPro data and supplementary data end with `.jsonlines` to `data` directory and move the pretrained model to `logs` directory.
* Download GloVe embeddings and build custom kernels by running `setup_all.sh`.
    * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Setup training files by running `setup_training.sh`.

### Traning Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* For training and prediction, set the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* Training: `python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Prediction: `python predict.py <experiment>`
* Evaluation: `python evaluate.py <experiment>`

## Acknowledgment
We built the training framework based on the original [End-to-end Coreference Resolution](https://github.com/kentonl/e2e-coref).

## Others
If you have questions about the data or the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.