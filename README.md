# Visual Pronoun Coreference Resolution in Dialogues

## Introduction
This is the data and the source code for EMNLP 2019 paper "What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues".

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{zhang2019pronoun,
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

## Data
Download the VisPro dataset from [Data](https://drive.google.com/open?id=13FdN34JCkyyNJxoTYOkp73n5aplJAcDs).

## Usage of VisCoref

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