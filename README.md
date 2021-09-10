# Visual Pronoun Coreference Resolution in Dialogues

## Introduction
This is the data and the source code for EMNLP 2019 paper "What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues". [[PAPER](https://www.aclweb.org/anthology/D19-1516.pdf)][[PPT](https://drive.google.com/open?id=1T5911qE1XrToNcMTOhKoAFEiqEgco2Sv)]

### Abstract
Grounding pronouns to a visual object it refers to requires complex reasoning from various information sources, especially in conversational scenarios.
For example, when people in a conversation talk about something all speakers can see (e.g., <b>the statue</b>), they often directly use pronouns (e.g., <b>it</b>) to refer it without previous introduction.
This fact brings a huge challenge for modern natural language understanding systems, particularly conventional context-based pronoun coreference models.
To tackle this challenge, in this paper, we formally define the task of visual-aware pronoun coreference resolution (PCR), and introduce VisPro, a large-scale dialogue PCR dataset, to investigate whether and how the visual information can help resolve pronouns in dialogues.
We then propose a novel visual-aware PCR model, VisCoref, for this task and conduct comprehensive experiments and case studies on our dataset.
Results demonstrate the importance of the visual information in this PCR case and show the effectiveness of the proposed model.

<div align=center>
<img width="500" src="fig/dialog_example.PNG">
</div>

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{DBLP:conf/emnlp/YuZSSZ19,
  author    = {Xintong Yu and
               Hongming Zhang and
               Yangqiu Song and
               Yan Song and
               Changshui Zhang},
  title     = {What You See is What You Get: Visual Pronoun Coreference Resolution
               in Dialogues},
  booktitle = {Proceedings of {EMNLP-IJCNLP} 2019},
  pages     = {5122--5131},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://doi.org/10.18653/v1/D19-1516},
  doi       = {10.18653/v1/D19-1516},
}
```



## VisPro Dataset
VisPro dataset contains coreference annotation of 29,722 pronouns from 5,000 dialogues.

The train, validation, and test split of VisPro dataset are in `data` directory.

### An example of VisPro
<div align=center>
<img width="600" src="fig/data_example.PNG">
</div>
Mentions in the same coreference cluster are in the same color.

### Annotation Format
Each line contains the annotation of one dialog.
```
{
    "doc_key": str, # e.g. in "dl:train:0", "dl" indicates "dialog" genre to be compatible with the CoNLL format, and it is the same for all dialogs in VisPro; "train" means that it is from the train split of VisDial (note that the split of VisPro is not the same as VisDial); "0" is the original index in the randomly selected 5,000 VisDial dialogs; basically this key serves as an index of the dialog
    "image_file": str, # the image filename of the dialog
    "object_detection": list, # the ids of object labels from 80 categories of MSCOCO object detection challenge
    "sentences": list,
    "speakers": list,
    "cluster": list, # each element is a cluster, and each element within a cluster is a mention
    "correct_caption_NPs": list, # the noun phrases in the caption
    "pronoun_info": list
}
```

Each element of `"pronoun_info"` contains the annotation of one pronoun.
```
{
    "current_pronoun": list,
    "reference_type": int,
    "not_discussed": bool,
    "candidate_NPs": list,
    "correct_NPs": list
}
```
Text spans are denoted as [index_start, index_end] of their positions in the whole dialogue, and the indices is counted by concatenating all sentences of the dialogue together.

`"current_pronoun"`, `"candidate_NPs"`, and `"correct_NPs"` are positions of the pronouns, the candidate noun phrases and the correct noun phrases of antecedents respectively.

`"reference_type"` has 3 values. 0 for pronouns which refers to noun phrases in the text, 1 for pronouns whose antecedents are not in the candidate list, 2 for non-referential pronouns.

`"not_discussed"` indicates whether the antecedents of the pronoun is discussed in the dialogue text.

Take the first dialog in the test split of VisPro as example:
```
{
  "pronoun_info": [{"current_pronoun": [15, 15], "candidate_NPs": [[0, 1], [3, 4], [6, 8], [10, 10], [12, 12]], "reference_type": 0, "correct_NPs": [[0, 1], [10, 10]], "not_discussed": false}]，
  "sentences": [["A", "firefighter", "rests", "his", "arm", "on", "a", "parking", "meter", "as", "another", "walks", "past", "."], ["Is", "he", "in", "his", "gear", "?"]],
  "doc_key": "dl:train:152"
}
```
Here [0, 1] indicates the phrase of "a firefighter", [3, 4] indicates "his arms", [6, 8] indicates "a parking meter", [10, 10] indicates "another", [12, 12] indicates "past", and [15, 15] indicates "he."
For the current pronoun "he", "candidate_NPs" means that "a firefighter", "his arms", "a parking meter", "another", "past" all serve as candidates for antecedents, while "correct_caption_NPs" means that only "a firefighter" and "another" are correct antecedents.

The "doc_key" means that it is the 152th selected dialog from the train split of VisDial.


## Usage of VisCoref

### An Example of VisCoref Prediction
<div align=center>
<img width="700" src="fig/case_study1.png">
</div>

The figure shows an example of a VisCoref prediction with the image, the relevant part of the dialogue, the prediction result, and the heatmap of the text-object similarity. We indicate the target pronoun with the *underlined italics* font and the candidate mentions with <b>bold</b> font. The row of the heatmap represents the mention in the context and the column means the detected object labels from the image.

### Getting Started
* Install python 3.7 and the following requirements: `pip install -r requirements.txt`. Set default python under your system to python 3.7.
* Download supplementary data for training VisCoref and the pretrained model from [Data](https://drive.google.com/open?id=1dSeGz5k57bU2GXCt7sY9krykLvmnbiVx) and extract: `tar -xzvf VisCoref.tar.gz`.
* Move VisPro data and supplementary data end with `.jsonlines` to `data` directory and move the pretrained model to `logs` directory.
* Download GloVe embeddings and build custom kernels by running `setup_all.sh`.
    * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Setup training files by running `setup_training.sh`.

### Traning Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* For training and prediction, set the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* (optional) For the "End-to-end + Visual" baseline, first download images from [VisDial](https://visualdialog.org/data) to the `data/images` folder, then run `python get_im_fc.py` to get image features.
* Training: `python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Prediction: `python predict.py <experiment>`
* Evaluation: `python evaluate.py <experiment>`

## Acknowledgment
VisPro dataset is based on [VisDial v1.0](https://visualdialog.org/).

We built the training framework based on the original [End-to-end Coreference Resolution](https://github.com/kentonl/e2e-coref).

## Others
If you have questions about the data or the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.