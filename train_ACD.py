import torch
from data_utils import reset_params,myDataset,setSeed
from torch.utils.data import DataLoader
from test_ACD import test
from vocab_ACSC import WhitespaceTokenizer,buildVocab,Vocab4rest14DevSplit,Vocab4rest14_hard,Vocab4restLarge_restLargeHard
import argparse
import torch.nn as nn
from models_ACSC import ACD
import warnings
warnings.filterwarnings("ignore")

def train(opt,vocab):
    model=opt.model(vocab,opt)
    model=model.to(opt.device)
    bert_parameters=model.bert.named_parameters()
    no_decay=['bias','LaterNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters=[
        {'params':[p for n,p in bert_parameters if not any(nd in n for nd in no_decay)],'weight_decay':1e-5,'lr':opt.lr_bert},
        {'params':[p for n,p in bert_parameters if any(nd in n for nd in no_decay)],'weight_decay':0.0,'lr':opt.lr_bert},
        {'params':[p for n,p in model.predicts.named_parameters()],'weight_decay':1e-5,'lr':opt.lr},
        {'params':[p for n,p in model.attention.named_parameters()],'weight_decay':1e-5,'lr':opt.lr}
    ]
    optimizer=opt.optim(optimizer_grouped_parameters)
    trainLoader=DataLoader(myDataset(vocab.trainSent,vocab.trainY,opt.needed_data,vocab.trainAspect,vocab.trainAdjMatrix),batch_size=opt.batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    devLoader=DataLoader(myDataset(vocab.devSent,vocab.devY,opt.needed_data,vocab.devAspect,vocab.devAdjMatrix),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    max_acc=0
    global_step=0
    max_f1=0
    model_path='./pkl_files_ACSC/model_'+opt.dataset+'_'+opt.model_name+'.pkl'
    for i in range(opt.epoch):
        allLoss=0
        for num,batch in enumerate(trainLoader):
            global_step=global_step+1
            model.train()
            optimizer.zero_grad()
            pred,loss=model(batch)
            allLoss=allLoss+loss.item()
            loss.backward()
            optimizer.step()
            if global_step%10==0:
                accuracy,f1score=test(model,devLoader,opt,vocab)
                if accuracy>max_acc:
                    max_acc=accuracy
                    max_f1=f1score
                    torch.save(model,model_path)            
        print('Epoch:{} allLoss:{}'.format(i,allLoss))
    print('Max Accuracy:{} f1 score:{}'.format(max_acc,max_f1))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default='ACD_Bert',type=str)
    parser.add_argument('--dataset',default='rest14DevSplit',type=str)
    parser.add_argument('--optim',default='adam',type=str)
    parser.add_argument('--epoch',default=2,type=int)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--lr_bert',default=1e-5,type=float)
    parser.add_argument('--l2reg',default=1e-5,type=float)
    parser.add_argument('--initializer',default='xavier_uniform_',type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim',default=300,type=int)
    parser.add_argument('--hidden_size',default=300,type=int)
    parser.add_argument('--hidden_size_bert',default=768,type=int)
    parser.add_argument('--dropout',default=0.4,type=float)
    parser.add_argument('--bert_name',default="rest_pt",type=str)
    parser.add_argument('--device',default='cuda:0',type=str)
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
    vocab={
        'rest14DevSplit':Vocab4rest14DevSplit,
        'rest14_hard':Vocab4rest14_hard,
        'rest_large':Vocab4restLarge_restLargeHard,
        'rest_large_hard':Vocab4restLarge_restLargeHard,
    }
    needed_data={
        'ACD_Bert':['sents','aspect']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    model_class={
        'ACD_Bert':ACD.ACD_Bert
    }
    bert_path={
        'rest_pt':'./data/ATE_data/rest_pt'
    }
    opt.bert_path=bert_path[opt.bert_name]
    opt.model=model_class[opt.model_name]
    opt.optim=optimizers[opt.optim]
    opt.vocab=vocab[opt.dataset]
    opt.initializer=initializers[opt.initializer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.needed_data=needed_data[opt.model_name]
    setSeed()
    vocab=buildVocab(opt)
    train(opt,vocab)