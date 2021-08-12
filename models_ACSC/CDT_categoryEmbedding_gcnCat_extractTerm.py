import torch
from models_ACSC.DynamicLSTM import DynamicLSTM
import torch.nn as nn
from models_ACSC.GNN import GraphConvolution
from data_utils import myDataset
import numpy as np

class CDT_categoryEmbedding_gcnCat_extractTerm(nn.Module):
    def __init__(self,vocab,opt):
        super().__init__()
        self.vocab=vocab
        self.opt=opt
        self.wordEmbedding=nn.Embedding.from_pretrained(torch.FloatTensor(vocab.word_vecs))
        self.wordEmbedding.requires_grad=True
        self.bilstm=DynamicLSTM(opt.embedding_dim*2,opt.hidden_size,batch_first=True,bidirectional=True)
        self.gcn1=GraphConvolution(opt.embedding_dim*3,opt.hidden_size*3)
        self.gcn2=GraphConvolution(opt.embedding_dim*3,opt.hidden_size*3)
        self.predict=nn.Linear(opt.hidden_size*3,vocab.polarity_num)
        self.dropout=nn.Dropout(opt.dropout)
        self.crossentry=nn.CrossEntropyLoss()

    def forward(self,data):
        sents_len=torch.LongTensor([len(i['sents']) for i in data]).to(self.opt.device)
        max_len=max([len(i['sents']) for i in data])
        category=torch.LongTensor([i['aspect'] for i in data]).to(self.opt.device)
        sents=myDataset.to_input_tensor([i['sents'] for i in data]).to(self.opt.device)
        labels=torch.LongTensor([i['labels'] for i in data]).to(self.opt.device)
        aspect_len=torch.LongTensor([len(i['extractTerm_aspect']) for i in data]).to(self.opt.device)
        extractTerm_index=myDataset.to_input_tensor([i['extractTerm_index'] for i in data]).to(self.opt.device)
        adj_matrix_temp=[i['adjMatrix'] for i in data]
        adj_matrix=[]
        for adj in adj_matrix_temp:
            adj_matrix.append(np.pad(adj,((0,max_len-len(adj)),(0,max_len-len(adj))),'constant'))
        adj_matrix=torch.LongTensor(adj_matrix).to(self.opt.device)
        wordEmbedding=self.wordEmbedding(sents)
        aspectEmbedding=self.wordEmbedding(category).unsqueeze(1).repeat(1,wordEmbedding.shape[1],1)
        embedding=torch.cat((wordEmbedding,aspectEmbedding),2)
        embedding=self.dropout(embedding)
        lstm_out,(_,_)=self.bilstm(embedding,sents_len)
        lstm_out=self.dropout(lstm_out)
        lstm_out_category=torch.cat((lstm_out,aspectEmbedding),2)
        gcn_out=self.gcn1(adj_matrix,lstm_out_category)
        gcn_out=self.gcn2(adj_matrix,gcn_out)
        gcn_mask=gcn_out*(extractTerm_index.unsqueeze(2).float())
        sum_gcn=torch.sum(gcn_mask,1)
        avg_gcn=sum_gcn/(aspect_len.unsqueeze(1).float())
        pred=self.predict(avg_gcn)
        loss=self.crossentry(pred,labels)
        return pred,loss