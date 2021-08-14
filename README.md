![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# LC-ACSA
The code repository for NLPCC2021 paper "Locate and Combine: A Two-Stage Framework for Aspect-Category Sentiment Analysis".

## Data preparation
Download the [glove.840B.300d.txt](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip),[bert-base-uncased](https://huggingface.co/bert-base-uncased),[restaurant post-training BERT](https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/pytorch-pretrained-bert.md) to the corresponding paths in your computer.Remenber to modify the dataset path according to your setting.

## Set up the environment
pip install -r requirement.txt

## Running the code
Run the following commands for restaurant 2014 dataset. For the restaurant 2014 hard, restaurant large and restaurant large hard datasets, change rest14DevSplit in the following commands to rest14_hard, rest_large and rest_large_hard respectively.

### Train (for LC-LSTM)
1. python train_ATE.py --dataset allRest --lr 3e-5 --dropout 0.2
2. python train_ACD.py --dataset rest14DevSplit  --epoch 10 
3. sh scripts_ACSC/CDT_categoryEmbedding_gcnCat_extractTerm.sh cuda:0 train rest14DevSplit 40 

### Test (for LC-LSTM)
1. sh scripts_ACSC/CDT_categoryEmbedding_gcnCat_extractTerm.sh cuda:0 test rest14DevSplit


## Trained models
Trained models can be found [here](https://drive.google.com/file/d/1f4pJ3SQQK7gadyCT6YHcdmrwCUaYwgdQ/view?usp=sharing). For Chinese, trained models can also be found [here](https://pan.baidu.com/s/1niwufdgzwIOtggden4QgYw) and the key is u6m0.
