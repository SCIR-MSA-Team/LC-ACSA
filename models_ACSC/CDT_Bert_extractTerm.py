import torch
import torch.nn as nn
from data_utils import myDataset
from transformers import BertModel
from vocab_ACSC import *
from models_ACSC.GNN import GraphConvolution
import numpy as np

class CDT_Bert_extractTerm(nn.Module):
    def __init__(self,vocab,opt):
        super().__init__()
        self.vocab=vocab
        self.opt=opt
        self.bert=BertModel.from_pretrained(opt.bert_path)
        self.bert_vocab=Vocab4Bert(opt.bert_path)
        self.predict=nn.Linear(opt.hidden_size_bert,vocab.polarity_num)
        self.gcn1=GraphConvolution(opt.hidden_size_bert*2,opt.hidden_size_bert)
        self.gcn2=GraphConvolution(opt.hidden_size_bert,opt.hidden_size_bert)
        self.dropout=nn.Dropout(opt.dropout)
        self.crossentry=nn.CrossEntropyLoss()

    def forward(self,data):
        sents=[i['sents'] for i in data]
        max_len=max([len(sent) for sent in sents])
        word_sents=self.vocab.indices2words(sents)
        category=self.vocab.indices2words4aspect([i['aspect'] for i in data])
        adj_matrix_temp=[i['adjMatrix'] for i in data]
        adj_matrix=[]
        for adj in adj_matrix_temp:
            adj_matrix.append(np.pad(adj,((0,max_len-len(adj)),(0,max_len-len(adj))),'constant'))
        adj_matrix=torch.LongTensor(adj_matrix).to(self.opt.device)
        extractTerm_index=myDataset.to_input_tensor([i['extractTerm_index'] for i in data]).to(self.opt.device)
        aspect_len=torch.LongTensor([len(i['extractTerm_aspect']) for i in data]).to(self.opt.device)
        labels=torch.LongTensor([i['labels'] for i in data]).to(self.opt.device)
        for num,sent in enumerate(word_sents):
            sent.insert(0,'[CLS]')
            sent.append('[SEP]')
            sent.append(category[num])
        word_sents=[' '.join(sent) for sent in word_sents]
        token_ids,word_bert_bpe_words=self.bert_vocab.sents2ids(word_sents)
        attention_mask=[]
        token_type_ids=[]
        for num,token in enumerate(token_ids):
            attention_mask.append([1]*len(token))
            token_type_ids.append([0]*len(token))
        token_ids=myDataset.to_input_tensor(token_ids).to(self.opt.device)
        attention_mask=myDataset.to_input_tensor(attention_mask).to(self.opt.device)
        token_type_ids=myDataset.to_input_tensor(token_type_ids).to(self.opt.device)
        output=self.bert(input_ids=token_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        batch_embedding=[]
        catgeory_embedding=[]
        for batch_num in range(len(sents)):
            sequence_embedding_merge=[]
            for word_sequence in range(max_len):
                if word_sequence in list(range(len(sents[batch_num]))):
                    word_embedding=output[0][batch_num,word_bert_bpe_words[batch_num][word_sequence+1][0],:]
                else:
                    word_embedding=torch.zeros(self.opt.hidden_size_bert).to(self.opt.device)
                if word_sequence==len(sents[batch_num])-1:
                    catgeory_embedding.append(output[0][batch_num,word_bert_bpe_words[batch_num][len(sents[batch_num])+2][0],:])
                sequence_embedding_merge.append(word_embedding.unsqueeze(0))
            sequence_embedding_merge=torch.cat(sequence_embedding_merge,0)
            batch_embedding.append(sequence_embedding_merge.unsqueeze(0))
        batch_embedding=torch.cat(batch_embedding,0).to(self.opt.device)
        catgeory_embedding=torch.stack(catgeory_embedding).unsqueeze(1).repeat(1,max_len,1).to(self.opt.device)
        batch_embedding=torch.cat((batch_embedding,catgeory_embedding),2)
        batch_embedding=self.dropout(batch_embedding)
        gcn_out=self.gcn1(adj_matrix,batch_embedding)
        gcn_out=self.gcn2(adj_matrix,gcn_out)
        gcn_mask=gcn_out*(extractTerm_index.unsqueeze(2).float())
        gcn_sum=torch.sum(gcn_mask,1)
        gcn_ave=gcn_sum/(aspect_len.unsqueeze(1).float())
        pred=self.predict(gcn_ave)
        loss=self.crossentry(pred,labels)
        return pred,loss