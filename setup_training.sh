#!/bin/bash

python get_char_vocab.py
python filter_embeddings.py 
python filter_embeddings.py --embedding glove_50_300_2.txt
python cache_elmo.py --dataset vispro
python cache_elmo.py --dataset vispro_cdd
python cache_elmo.py --dataset vispro_mscoco