from itertools import chain
import torch
import torch.nn as nn
from data_utils import estimate,cuda2cpu,setSeed,myDataset
from torch.utils.data import DataLoader
from vocab_ATE import buildVocab,Vocab4AllRest,Vocab4Rest141516Lap14
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

def test(model,dataLoader,opt):
    model.eval()
    with torch.no_grad():
        allLoss=0
        allResult=[]
        allLabel=[]
        allPred=[]
        for i,batch in enumerate(dataLoader):
            result=model.predict(batch[0]['sents'])
            labels=torch.LongTensor(batch[0]['labels']).to(opt.device)
            allResult.append(cuda2cpu(result))
            allLabel.append(cuda2cpu(labels))
        allLabel=list(chain.from_iterable(allLabel))
        allResult=list(chain.from_iterable(allResult))
        accuracy,precision,recall,f1score,cm=estimate(allLabel,allResult)
        print('[INFO]Test accuracy:{} F1:{} '.format(accuracy,f1score))
        sys.stdout.flush()
        return accuracy,f1score

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default='bert_pt',type=str)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--trainDataset',default='rest14',type=str)
    parser.add_argument('--dataset',default='res14',type=str)
    parser.add_argument('--device',default='cuda:0',type=str)
    parser.add_argument('--class_num',default=3,type=int)
    parser.add_argument('--bert_path',default='../data/ATE_data/rest_pt',type=str)
    opt=parser.parse_args()
    needed_data={
        'bert_pt':['sents','labels'],
        'bert_origin':['sents','labels']
    }
    vocabs={
        'rest14':Vocab4Rest141516Lap14,
        'rest15':Vocab4Rest141516Lap14,
        'rest16':Vocab4Rest141516Lap14,
        'allRest':Vocab4AllRest
    }
    opt.vocab=vocabs[opt.dataset]
    vocab=buildVocab(opt)
    opt.needed_data=needed_data[opt.model_name]
    model_path='./pkl_files_ATE/model_'+opt.trainDataset+'_'+opt.model_name+'.pkl'
    print('model_path',model_path)
    model=torch.load(model_path)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testY,opt.needed_data),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    test(model,testLoader,opt)