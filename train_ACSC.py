import torch
import sys
from data_utils import reset_params,myDataset,setSeed,process_extractTerm
from torch.utils.data import DataLoader
from vocab_ACSC import buildVocab,Vocab4rest14DevSplit,Vocab4rest14_hard,Vocab4restLarge_restLargeHard,WhitespaceTokenizer
import argparse
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from test_ACSC import test
from combine_ATE_ACD import extractTerm_combine_ATE_ACD
from models_ACSC import CDT_Bert_extractTerm,CDT_categoryEmbedding_gcnCat_extractTerm
import warnings
warnings.filterwarnings("ignore")

def train(opt,vocab):
    setSeed(opt.seed)
    model=opt.model(vocab,opt)
    model=model.to(opt.device)
    optimizer=opt.optim(model.parameters(),lr=opt.lr,weight_decay=opt.l2reg)
    train_extractTerm_aspect,train_extractTerm_index,dev_extractTerm_aspect,dev_extractTerm_index=None,None,None,None
    if 'extractTerm_aspect' in opt.needed_data or 'extractTerm_index' in opt.needed_data:
        train_extractTerm_aspect,train_extractTerm_index=extractTerm_combine_ATE_ACD(vocab.trainSent,vocab.trainAspect,vocab,opt)
        dev_extractTerm_aspect,dev_extractTerm_index=extractTerm_combine_ATE_ACD(vocab.devSent,vocab.devAspect,vocab,opt)
        train_extractTerm_aspect=process_extractTerm(train_extractTerm_aspect)
        dev_extractTerm_aspect=process_extractTerm(dev_extractTerm_aspect)
    trainLoader=DataLoader(myDataset(vocab.trainSent,vocab.trainY,opt.needed_data,vocab.trainAspect,vocab.trainAdjMatrix,train_extractTerm_aspect,train_extractTerm_index),batch_size=opt.batch_size,shuffle=True,collate_fn=myDataset.collate_fn)
    devLoader=DataLoader(myDataset(vocab.devSent,vocab.devY,opt.needed_data,vocab.devAspect,vocab.devAdjMatrix,dev_extractTerm_aspect,dev_extractTerm_index),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    crossentry=nn.CrossEntropyLoss()
    t_total=len(trainLoader)*opt.epoch
    schedule=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(t_total*0.1),num_training_steps=t_total)
    max_acc=0
    global_step=0
    max_f1=0
    model_path="./pkl_files_ACSC/model_"+opt.dataset+"_"+opt.model_name+".pkl"
    for i in range(opt.epoch):
        allLoss=0
        for j,batch in enumerate(trainLoader):
            model.train()
            global_step=global_step+1
            optimizer.zero_grad()
            pred,loss=model(batch)
            allLoss=allLoss+loss.item()
            loss.backward()
            optimizer.step()
            schedule.step()
            if global_step%10==0:
                accuracy,f1score=test(model,devLoader,opt)
                if accuracy>max_acc:
                    max_acc=accuracy
                    max_f1=f1score
                    torch.save(model,model_path)
        print('Epoch:{} allLoss:{}'.format(i,allLoss))
        sys.stdout.flush()
    print('Max Accuracy:{} f1 score:{}'.format(max_acc,max_f1))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='AC_MIMLLN', type=str)
    parser.add_argument('--dataset', default='rest14DevSplit', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim', default=300, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--hidden_size_bert',default=768,type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--hasDev',default=False,type=bool)
    parser.add_argument('--dropout',default=0.5,type=float)
    parser.add_argument('--lstm_layers',default=3,type=int)
    parser.add_argument('--bert_name',default="bert-base-uncased",type=str)
    parser.add_argument('--k',default=2,type=int)
    opt = parser.parse_args()
    opt.UseMisc=1
    opt.UseACD=1
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
        'CDT_Bert_extractTerm':['sents','aspect','adjMatrix','labels','extractTerm_aspect','extractTerm_index'],
        'CDT_categoryEmbedding_gcnCat_extractTerm':['sents','adjMatrix','aspect','labels','extractTerm_aspect','extractTerm_index'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    model_class = {
        'CDT_Bert_extractTerm':CDT_Bert_extractTerm.CDT_Bert_extractTerm,
        'CDT_categoryEmbedding_gcnCat_extractTerm':CDT_categoryEmbedding_gcnCat_extractTerm.CDT_categoryEmbedding_gcnCat_extractTerm,
    }
    bert_path={
        'bert-base-uncased':'./data/pretrain_model/bert_base_uncased',
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
    print('opt',opt)
    vocab=buildVocab(opt)
    train(opt,vocab)
