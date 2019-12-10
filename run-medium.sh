#!/bin/bash
cd code
export NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
python main.py --ac_share_emb --batch_size 1 --config configs.$1-low --optim adadelta --lr 1.0 --pos_sp --output_pred --epochs 15 --patience 15 --lang $1 --split medium
