from itertools import chain
import torch
from models_ACSC import CDT_Bert_extractTerm,CDT_categoryEmbedding_gcnCat_extractTerm
from combine_ATE_ACD import extractTerm_combine_ATE_ACD
import torch.nn as nn
from data_utils import estimate,cuda2cpu,setSeed,myDataset,process_extractTerm
from torch.utils.data import DataLoader
from vocab_ACSC import WhitespaceTokenizer,buildVocab,Vocab4rest14DevSplit,Vocab4rest14_hard,Vocab4restLarge_restLargeHard
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
        all_extractTerm_aspect=[]
        crossentry=nn.CrossEntropyLoss()
        for i,batch in enumerate(dataLoader):
            pred,loss=model(batch)
            _,result=torch.max(pred,1)
            labels=torch.LongTensor([i['labels'] for i in batch]).to(opt.device)
            extractTerm_aspect=[i['extractTerm_aspect'] for i in batch] if 'extractTerm_aspect' in opt.needed_data else None
            allResult.append(cuda2cpu(result))
            allLabel.append(cuda2cpu(labels))
            allPred.append(cuda2cpu(pred))
            allLoss=allLoss+loss.item()
            all_extractTerm_aspect.append(extractTerm_aspect)
        allLabel=list(chain.from_iterable(allLabel))
        allResult=list(chain.from_iterable(allResult))
        allPred=list(chain.from_iterable(allPred))
        accuracy,precision,recall,f1score,cm=estimate(allLabel,allResult)
        print('[INFO]TEST accuracy:{} F1:{} Loss:{}'.format(accuracy,f1score,allLoss))
        sys.stdout.flush()
        return accuracy,f1score

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',default='AC_MIMLLN',type=str)
    parser.add_argument('--dataset',default='rest14DevSplit',type=str,help='twitter,rest14,laptop,rest15,rest16')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--k',default=2,type=int)
    opt=parser.parse_args()
    opt.UseMisc=1
    opt.UseACD=1
    model_class={
        'CDT_Bert_extractTerm':CDT_Bert_extractTerm.CDT_Bert_extractTerm,
        'CDT_categoryEmbedding_gcnCat_extractTerm':CDT_categoryEmbedding_gcnCat_extractTerm.CDT_categoryEmbedding_gcnCat_extractTerm,
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
    opt.vocab=vocab[opt.dataset]
    opt.needed_data=needed_data[opt.model_name]
    print('opt',opt)
    setSeed(opt.seed)
    vocab=buildVocab(opt)
    model_path="./pkl_files_ACSC/model_"+opt.dataset+"_"+opt.model_name+".pkl"
    model=torch.load(model_path)
    test_extractTerm_aspect,test_extractTerm_index=None,None
    if 'extractTerm_aspect' in opt.needed_data or 'extractTerm_index' in opt.needed_data:
        test_extractTerm_aspect,test_extractTerm_index=extractTerm_combine_ATE_ACD(vocab.testSent,vocab.testAspect,vocab,opt)
        test_extractTerm_aspect=process_extractTerm(test_extractTerm_aspect)
    testLoader=DataLoader(myDataset(vocab.testSent,vocab.testY,opt.needed_data,vocab.testAspect,vocab.testAdjMatrix,test_extractTerm_aspect,test_extractTerm_index),batch_size=opt.batch_size,shuffle=False,collate_fn=myDataset.collate_fn)
    test(model,testLoader,opt)