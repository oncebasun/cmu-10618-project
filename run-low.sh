#!/bin/bash
cd code
export NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
python main.py --ac_share_emb --batch_size 1 --seed $1 --optim adadelta --lr 1.0 --rnn_type $2 --pos_sp --output_pred --epochs 20 --patience 20 --lang $3 --split low
