#!/bin/bash
batch_size=64
if [ $3 == "rest14DevSplit" ] || [ $3 == "rest14_hard" ]
then
    batch_size=25
fi
if [ $2 == "train" ]
then
    python train_ACSC.py --device $1 --dataset $3 --epoch $4 --lr 0.01 --l2reg 1e-5 --batch_size $batch_size --model_name CDT_categoryEmbedding_gcnCat_extractTerm --dropout 0.3
elif [ $2 == "test" ]
then 
    python test_ACSC.py --batch_size $batch_size --device $1 --dataset $3 --model_name CDT_categoryEmbedding_gcnCat_extractTerm
fi