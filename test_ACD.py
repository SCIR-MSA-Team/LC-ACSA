from itertools import chain
import torch
from models_ACSC.ACD import ACD_Bert
from vocab_ACSC import WhitespaceTokenizer,buildVocab,Vocab4rest14DevSplit,Vocab4rest14_hard,Vocab4restLarge_restLargeHard
from data_utils import estimate,cuda2cpu,setSeed,myDataset
from torch.utils.data import DataLoader
import numpy as np
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

def test(model,dataLoader,opt,vocab):
    model.eval()
    with torch.no_grad():
        allLoss=0
        allResult=[]
        allLabel=[]
        allPred=[]
        for i,batch in enumerate(dataLoader):
            pred,loss=model(batch)
            aspects=[i['aspect'] for i in batch]
            aspects=[vocab.aspectOrder[aspect] for aspect in aspects]
            allLoss=allLoss+loss.item()
            _,result=torch.max(pred,1)
            allResult.append(cuda2cpu(result))
            allPred.append(cuda2cpu(pred))
            allLabel.append(cuda2cpu(aspects))
            allLoss=allLoss+loss.item()
        allLabel=list(chain.from_iterable(allLabel))
        allResult=list(chain.from_iterable(allResult))
        allPred=list(chain.from_iterable(allPred))
        accuracy,precision,recall,f1score,cm=estimate(allLabel,allResult)
        print('[INFO]TEST accuracy:{} F1:{} Loss:{}'.format(accuracy,f1score,allLoss))
        sys.stdout.flush()
        return accuracy,f1score

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default='ACD_Bert',type=str)
    parser.add_argument('--dataset',default='rest14DevSplit',type=str,help='twitter,rest14,laptop,rest15,rest16')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    opt=parser.parse_args()
    model_class={
        'ACD_Bert':ACD_Bert
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
    setSeed()
    opt.vocab=vocab[opt.dataset]
    opt.needed_data=needed_data[opt.model_name]
    vocab=buildVocab(opt)
    model_path="./pkl_files_ACSC/model_"+opt.dataset+"_"+opt.model_name+".pkl"
    model=torch.load(model_path)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testY,opt.needed_data,vocab.testAspect,vocab.testAdjMatrix),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    test(model,testLoader,opt,vocab)