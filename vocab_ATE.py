import torch
from transformers import BertTokenizer
from pickle import load,dump
from bs4 import BeautifulSoup
from data_utils import clean_str
import os
import json

class Vocab:
    def __init__(self,bert_path):
        self.tokenizer=BertTokenizer.from_pretrained(bert_path)

    def readFile(self,data_path):
        sentence=[]
        label=[]
        sent_data=open(data_path+'/'+'sentence.txt','r').readlines()
        target_data=open(data_path+'/'+'target.txt','r').readlines()
        assert len(sent_data)==len(target_data)
        for num,text in enumerate(sent_data):
            text=text.strip().split()
            text.insert(0,'[CLS]')
            text.append('[SEP]')
            target=[int(a) for a in target_data[num].strip().split()]
            target.insert(0,0)
            target.append(0)
            sentence.append(text)
            label.append(target)
        return sentence,label

    def sents2ids(self,sents):
        word_bert_bpe_words=[]
        for i,sent in enumerate(sents):
            count=0
            temp=dict()
            text_split=sent
            for num,word in enumerate(text_split):
                token_word=self.tokenizer.tokenize(word)
                temp[num]=list(range(count,count+len(token_word)))
                count=count+len(token_word)
            word_bert_bpe_words.append(temp)
        sents=[' '.join(sent) for sent in sents]
        token_texts=[self.tokenizer.tokenize(text) for text in sents]
        token_ids=[self.tokenizer.convert_tokens_to_ids(text) for text in token_texts]
        return token_ids,word_bert_bpe_words

class Vocab4Rest141516Lap14(Vocab):
    def __init__(self,bert_path,train_path,dev_path,test_path):
        super().__init__(bert_path)
        self.trainSent,self.trainY=self.readFile(train_path)
        self.testSent,self.testY=self.readFile(test_path)
        self.devSent,self.devY=self.readFile(dev_path) if dev_path!=None else self.testSent,self.testY

class Vocab4AllRest(Vocab):
    def __init__(self,bert_path,rest14_path,rest15_path,rest16_path):
        super().__init__(bert_path)
        self.trainSent,self.trainY=[],[]
        self.testSent,self.testY=[],[]
        rest14TrainSent,rest14TrainY=self.readFile(rest14_path+'/train')
        rest14TestSent,rest14TestY=self.readFile(rest14_path+'/test')
        rest15TrainSent,rest15TrainY=self.readFile(rest15_path+'/train')
        rest15TestSent,rest15TestY=self.readFile(rest15_path+'/test')
        rest16TrainSent,rest16TrainY=self.readFile(rest16_path+'/train')
        rest16TestSent,rest16TestY=self.readFile(rest16_path+'/test')
        self.trainSent=rest14TrainSent+rest15TrainSent+rest16TrainSent
        self.trainY=rest14TrainY+rest15TrainY+rest16TrainY
        self.testSent=rest14TestSent+rest15TestSent+rest16TestSent
        self.testY=rest14TestY+rest15TestY+rest16TestY
        self.devSent,self.devY=self.testSent,self.testY

def buildVocab(opt):
    train_path=None
    dev_path=None
    test_path=None
    if opt.dataset=="rest14":
        train_path='./data/ATE_data/rest14/train'
        test_path='./data/ATE_data/rest14/test'
    elif opt.dataset=="rest15":
        train_path='./data/ATE_data/rest15/train'
        test_path='./data/ATE_data/rest15/test'
    elif opt.dataset=="rest16":
        train_path='./data/ATE_data/rest16/train'
        test_path='./data/ATE_data/rest16/test'
    elif opt.dataset=="allRest":
        train_path='./data/ATE_data/rest14'
        dev_path='./data/ATE_data/rest15'
        test_path='./data/ATE_data/rest16'
    else:
        print('Please check the input dataset')
        exit()
    vocab_path='./pkl_files_ATE/vocab_'+opt.dataset+'.pkl'
    if os.path.exists(vocab_path):
        print('Load vocab.pkl')
        vocab=load(open(vocab_path,'rb'))
    else:
        print('Create vocab.pkl')
        vocab=opt.vocab(opt.bert_path,train_path,dev_path,test_path)
        dump(vocab,open(vocab_path,'wb'))
    return vocab