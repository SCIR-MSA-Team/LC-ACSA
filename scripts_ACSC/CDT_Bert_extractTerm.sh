#!/bin/bash
if [ $2 == "train" ]
then
    python train_ACSC.py --model_name CDT_Bert_extractTerm --device $1 --dataset $3 --hidden_size_bert 768 --epoch $4 --batch_size 16 --lr 1e-5
elif [ $2 == "test" ]
then
    python test_ACSC.py --model_name CDT_Bert_extractTerm --device $1 --dataset $3 --batch_size 16
fi