import random
import numpy as np
import torch
import torch.nn as nn
import os
import re
import math
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,mean_absolute_error,confusion_matrix
from torch.utils.data import Dataset

def setSeed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.manual_seed_all(seed)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\' "," \' ", string)
    return string.strip()

def cuda2cpu(pred):
    if type(pred)==list:
        return pred
    if pred.is_cuda:
        pred_cpu=list(pred.cpu().numpy())
    else:
        pred_cpu=list(pred.numpy())
    return pred_cpu

def estimate(labels,results):
    labels=cuda2cpu(labels)
    results=cuda2cpu(results)
    accuracy=accuracy_score(labels,results)
    precision=precision_score(labels,results,average='macro')
    recall=recall_score(labels,results,average='macro')
    f1score=f1_score(labels,results,average='macro')
    cm=confusion_matrix(labels,results)
    return accuracy,precision,recall,f1score,cm

def reset_params(model,initializer):
    for p in model.parameters():
        if p.requires_grad:
            if len(p.shape) > 1:
                initializer(p)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Linear):
        m.weight.data.normal_(-0.01,0.01)
        m.bias.data.zero_()

def process_extractTerm(extractTerm_aspect):
    temp_list=[]
    for i in extractTerm_aspect:
        temp=[]
        for j in i:
            if len(j.split())==1:
                temp.append(j)
            else:
                temp.extend(j.split())
        temp_list.append(temp)
    extractTerm_aspect=temp_list
    return extractTerm_aspect

class myDataset(Dataset):
    pad=0
    def __init__(self,sents,y,needed_data,aspect=None,adjMatrix=None,extractTerm_aspect=0,extractTerm_index=0,pad=0):
        self.data=dict()
        self.data['sents']=sents if 'sents' in needed_data else None
        self.data['aspect']=aspect if 'aspect' in needed_data else None
        self.data['adjMatrix']=adjMatrix if 'adjMatrix' in needed_data else None
        self.data['labels']=y if 'labels' in needed_data else None
        self.data['extractTerm_aspect']=extractTerm_aspect if 'extractTerm_aspect' in needed_data else None
        self.data['extractTerm_index']=extractTerm_index if 'extractTerm_index' in needed_data else None
        self.len=len(sents)
        self.needed_data=needed_data
        myDataset.pad=pad
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        data=dict()
        for i in self.needed_data:
            data[i]=self.data[i][index]
        return data

    @staticmethod
    def to_input_tensor(sents):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(input_id) for input_id in sents],batch_first=True,padding_value=myDataset.pad)

    @staticmethod
    def collate_fn(batch):
        return batch

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)