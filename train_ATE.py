import argparse
import torch
import sys
import numpy as np
import torch.nn as nn
from test_ATE import test
from models_ATE.bert_pt import bert_pt
from vocab_ATE import buildVocab,Vocab4AllRest,Vocab4Rest141516Lap14
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from data_utils import reset_params,myDataset,setSeed
import warnings
warnings.filterwarnings("ignore")

def train(vocab,opt):
    model=opt.model(vocab,opt)
    model=model.to(opt.device)
    trainLoader=DataLoader(myDataset(vocab.trainSent,vocab.trainY,opt.needed_data),batch_size=opt.batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    devLoader=DataLoader(myDataset(vocab.devSent,vocab.devY,opt.needed_data),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    t_total=opt.epoch*len(trainLoader)
    bert_parameters=model.bert.named_parameters()
    param_optimizer=[(k,v) for k,v in model.named_parameters() if v.requires_grad==True]
    param_optimizer=[n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters=[
        {'params':[p for n,p in bert_parameters if not any(nd in n for nd in no_decay)],'weight_decay':1e-5,'lr':opt.lr_bert},
        {'params':[p for n,p in bert_parameters if any(nd in n for nd in no_decay)],'weight_decay':0.0,'lr':opt.lr_bert},
        {'params':[p for n,p in model.linear.named_parameters()],'weight_decay':1e-5,'lr':opt.lr}
    ]
    optimizer=opt.optim(optimizer_grouped_parameters)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(t_total*0.1),num_training_steps=t_total)
    crossentry=nn.CrossEntropyLoss()
    max_f1=0
    global_step=0
    model_path='./pkl_files_ATE/model_'+opt.dataset+'_'+opt.model_name+'.pkl'
    for i in range(opt.epoch):
        allLoss=0
        for j,batch in enumerate(trainLoader):
            global_step=global_step+1
            model.train()
            optimizer.zero_grad()
            pred,loss=model(batch)
            allLoss=allLoss+loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if global_step%100==0:
                accuracy,f1score=test(model,devLoader,opt)
                if f1score>max_f1:
                    max_f1=f1score
                    torch.save(model,model_path)
        print('Epoch:{} allLoss:{}'.format(i,allLoss))
    print('Max F1:{}'.format(max_f1))
        

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default="bert_pt",type=str)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--lr_bert',default=2e-5,type=float)
    parser.add_argument('--optim',default='adam',type=str)
    parser.add_argument('--dropout',default=0.5,type=float)
    parser.add_argument('--dataset',default="restaurant14",type=str)
    parser.add_argument('--epoch',default=3,type=int)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--seed',default=1234,type=int)
    parser.add_argument('--initializer',default='xavier_uniform_',type=str)
    parser.add_argument('--l2reg',default=1e-5,type=float)
    parser.add_argument('--bert_path',type=str)
    parser.add_argument('--device',default='cuda:0',type=str)
    parser.add_argument('--bert_hidden_size',default=768,type=int)
    parser.add_argument('--class_num',default=3,type=int)
    opt=parser.parse_args()
    optimizers = {
        'adadelta': torch.optim.Adadelta, 
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,  
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    bert_paths={
        'rest14':'./data/ATE_data/rest_pt',
        'rest15':'./data/ATE_data/rest_pt',
        'rest16':'./data/ATE_data/rest_pt',
        'allRest':'./data/ATE_data/rest_pt'
    }
    vocabs={
        'rest14':Vocab4Rest141516Lap14,
        'rest15':Vocab4Rest141516Lap14,
        'rest16':Vocab4Rest141516Lap14,
        'allRest':Vocab4AllRest
    }
    model_class={
        'bert_pt':bert_pt,
        'bert_origin':bert_pt
    }
    needed_data={
        'bert_pt':['sents','labels'],
        'bert_origin':['sents','labels']
    }
    opt.vocab=vocabs[opt.dataset]
    if opt.model_name=="bert_pt":
        opt.bert_path=bert_paths[opt.dataset]
    opt.model=model_class[opt.model_name]
    opt.optim=optimizers[opt.optim]
    opt.initializer=initializers[opt.initializer]
    opt.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.needed_data=needed_data[opt.model_name]
    setSeed()
    vocab=buildVocab(opt)
    train(vocab,opt)
