import pickle
from pickle import load, dump
import numpy as np
import os
import json
from data_utils import clean_str
from benepar.spacy_plugin import BeneparComponent
import spacy
from spacy.tokens import Doc
import argparse
from transformers import BertModel, BertTokenizer


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True]*len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class Vocab:
    def __init__(self):
        self.aspects = []
        self.words = ['pad']
        self.aspectOrder = None
        self.pad = 0
        self.length = 0
        self.polarity_num = self.getPolarity_num()
        self.polarity2id={"negative":0,"positive":1,"neutral":2,"conflict":3}
        self.word_vecs = None
        self.word2id = None
        self.inGlove = []
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def publicPart(self, labelTransform=True):
        for i in self.trainSent+self.devSent+self.testSent:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        for i in self.trainAspect+self.devAspect+self.testAspect:
            if i not in self.words:
                self.words.append(i)
        self.length=len(self.words)
        reverse=lambda x:dict(zip(x,range(self.length)))
        self.word2id=reverse(self.words)
        self.trainSent=self.words2indices(self.trainSent)
        self.devSent=self.words2indices(self.devSent) if self.devSent != None else None
        self.testSent=self.words2indices(self.testSent)
        self.trainAspect=self.words2indices4aspect(self.trainAspect)
        self.devAspect=self.words2indices4aspect(self.devAspect) if self.devAspect != None else None
        self.testAspect=self.words2indices4aspect(self.testAspect)
        self.word_vecs=np.random.uniform(-0.25,0.25,300*self.length).reshape(self.length,300)
        for i in self.trainAspect:
            if i not in self.aspects:
                self.aspects.append(i)
        self.aspectOrder=dict(zip(self.aspects,range(len(set(self.aspects)))))
        if labelTransform==True:
            self.trainY=[self.polarity2id[y] for y in self.trainY]
            self.devY=[self.polarity2id[y] for y in self.devY] if self.devY != None else None
            self.testY=[self.polarity2id[y] for y in self.testY]

    def getPolarity_num(self):
        return 3

    def words2indices(self,sents):
        return [[self.word2id[s] for s in w] for w in sents]

    def words2indices4aspect(self,aspect):
        return [self.word2id[aspe] for aspe in aspect]
    
    def indices2words(self,sents):
        return [[self.words[s] for s in w] for w in sents]

    def indices2words4aspect(self,aspect):
        return [self.words[aspe] for aspe in aspect]
    
    def dependency_adj_matrix(self,text):
        tokens=self.nlp(text)
        words=text.split()
        matrix=np.zeros((len(words),len(words))).astype('float32')
        assert len(words)==len(list(tokens))

        for token in tokens:
            matrix[token.i][token.i] = 1
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
        return matrix
    
    def load_pretrained_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.inGlove.append(word)
                    self.word_vecs[self.word2id[word]]=np.array([float(x) for x in line[1:]])

class Vocab4rest14DevSplit(Vocab):
    def __init__(self,pkl_path,dev_file=None,test_file=None):
        super().__init__()
        self.trainSent,self.trainAspect,self.trainAdjMatrix,self.trainY=None,None,None,None
        self.devSent,self.devAspect,self.devAdjMatrix,self.devY=None,None,None,None
        self.testSent,self.testAspect,self.testAdjMatrix,self.testY=None,None,None,None
        self.build(pkl_path)

    def read_data(self,data,index2word,sentence_map):
        sentence=[]
        aspect=[]
        label=[]
        adj_matrix=[]
        for i in data:
            token_text=i[0]
            text=' '.join([index2word[j] for j in token_text])
            text=sentence_map[text].lower().strip()
            aspect.append(index2word[i[2][0]])
            label.append(i[4])
            text=clean_str(text).split()
            sentence.append(text)
            adj_matrix.append(self.dependency_adj_matrix(' '.join(text)))
        return sentence,aspect,adj_matrix,label
    
    def build(self,pkl_path,sentence_path='./data/AC_MIMLLN_data/ABSA_DevSplits/dataset/sentence_map.txt'):
        with open(pkl_path,mode='rb') as file:
            data=pickle.load(file,encoding='utf-8')
            index2word=data['index_word']
            lines=[]
            with open(sentence_path,encoding='utf-8') as map_file:
                for line in map_file:
                    lines.append(line.strip('\r\n'))
            sentence_map={line.split('\t')[0]:line.split('\t')[1] for line in lines}
            self.trainSent,self.trainAspect,self.trainAdjMatrix,self.trainY=self.read_data(data["train"],index2word,sentence_map)
            self.devSent,self.devAspect,self.devAdjMatrix,self.devY=self.read_data(data["dev"],index2word,sentence_map)
            self.testSent,self.testAspect,self.testAdjMatrix,self.testY=self.read_data(data["test"],index2word,sentence_map)
            self.publicPart(labelTransform=False)

class Vocab4rest14_hard(Vocab4rest14DevSplit):
    def __init__(self,pkl_path,dev_file=None,test_file=None):
        super(Vocab4rest14_hard,self).__init__(pkl_path,dev_file,test_file)
        self.hardSent,self.hardAspect,self.hardAdjMatrix,self.hardY=[],[],[],[]
        self.extractHard()
        self.devSent,self.devAspect,self.devAdjMatrix,self.devY=self.hardSent,self.hardAspect,self.hardAdjMatrix,self.hardY
        self.testSent,self.testAspect,self.testAdjMatrix,self.testY=self.hardSent,self.hardAspect,self.hardAdjMatrix,self.hardY
        
    def extractHard(self):
        textCount=dict()
        for num,sent in enumerate(self.testSent):
            words=self.indices2words4aspect(sent)
            text=' '.join(words)
            if text not in textCount.keys():
                textCount[text]=[]
                textCount[text].append(self.testY[num])
            else:
                textCount[text].append(self.testY[num])
        yCount=dict()
        yCount[0]=0
        yCount[1]=0
        yCount[2]=0
        for num,sent in enumerate(self.testSent):
            words=self.indices2words4aspect(sent)
            text=' '.join(words)
            if len(set(textCount[text]))>1:
                self.hardSent.append(self.testSent[num])
                self.hardAspect.append(self.testAspect[num])
                self.hardAdjMatrix.append(self.testAdjMatrix[num])
                self.hardY.append(self.testY[num])
                yCount[self.testY[num]]=yCount[self.testY[num]]+1


class Vocab4restLarge_restLargeHard(Vocab):
    def __init__(self,train_file,dev_file,test_file):
        super().__init__()
        self.trainSent,self.trainAspect,self.trainAdjMatrix,self.trainY=self.readFile(train_file)
        self.testSent,self.testAspect,self.testAdjMatrix,self.testY=self.readFile(test_file)
        self.devSent,self.devAspect,self.devAdjMatrix,self.devY=self.testSent,self.testAspect,self.testAdjMatrix,self.testY
        self.publicPart()

    def readFile(self,fileName):
        sentence=[]
        aspect=[]
        adj_matrix=[]
        y=[]
        with open(fileName,'r',encoding='utf-8') as file:
            data=json.load(file)
            for i in data:
                text=clean_str(i['sentence']).lower().strip()
                text=text.split()
                sentence.append(text)
                aspect.append(clean_str(i['aspect']))
                y.append(i['sentiment'])
                adj_matrix.append(self.dependency_adj_matrix(' '.join(text)))
        return sentence,aspect,adj_matrix,y

class Vocab4Bert():
    def __init__(self,bert_path):
        self.tokenizer=BertTokenizer.from_pretrained(bert_path)
    
    def sents2ids(self,sents):
        word_bert_bpe_words=[]
        for i in sents:
            count=0
            temp=dict()
            text_split=i.split()
            for num,word in enumerate(text_split):
                token_word=self.tokenizer.tokenize(word)
                temp[num]=list(range(count,count+len(token_word)))
                count=count+len(token_word)
            word_bert_bpe_words.append(temp)
        token_texts=[self.tokenizer.tokenize(text) for text in sents]
        token_ids=[self.tokenizer.convert_tokens_to_ids(text) for text in token_texts]
        return token_ids,word_bert_bpe_words

def buildVocab(opt):
    train_file=None
    dev_file=None
    test_file=None
    if opt.dataset=="rest14DevSplit":
        train_file='./data/AC_MIMLLN_data/ABSA_DevSplits/dataset/Restaurants_category.pkl.new'
    elif opt.dataset=="rest14_hard":
        train_file='./data/AC_MIMLLN_data/ABSA_DevSplits/dataset/Restaurants_category.pkl.new'
    elif opt.dataset=="rest_large":
        train_file='./data/acsa-restaurant-large/acsa_train.json'
        test_file='./data/acsa-restaurant-large/acsa_test.json'
    elif opt.dataset=="rest_large_hard":
        train_file='./data/acsa-restaurant-large/acsa_hard_train.json'
        test_file='./data/acsa-restaurant-large/acsa_hard_test.json'
    else:
        print('[ERROR] Please check the vocab')
        exit()
    vocab_path="./pkl_files_ACSC/vocab_"+opt.dataset+".pkl"
    if os.path.exists(vocab_path):
        print('Load vocab.pkl')
        print('vocab_path',vocab_path)
        vocab=load(open(vocab_path,'rb'))
    else:
        print('Create vocab.pkl')
        vocab=opt.vocab(train_file,dev_file,test_file)
        vocab.load_pretrained_vec('./data/glove/glove.840B.300d.txt')
        dump(vocab,open(vocab_path,'wb'))
    return vocab

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='rest14DevSplit',type=str)
    vocab={
        'rest14DevSplit':Vocab4rest14DevSplit,
        'rest14_hard':Vocab4rest14_hard,
        'rest_large':Vocab4restLarge_restLargeHard,
        'rest_large_hard':Vocab4restLarge_restLargeHard,
        }
    opt=parser.parse_args()
    opt.vocab=vocab[opt.dataset]
    buildVocab(opt)