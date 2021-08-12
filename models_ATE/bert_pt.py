import torch
import torch.nn as nn
from data_utils import myDataset
from pytorch_pretrained_bert import BertTokenizer,BertModel

class bert_pt(nn.Module):
    def __init__(self,vocab,opt):
        super().__init__()
        self.opt=opt
        self.vocab=vocab
        self.bert=BertModel.from_pretrained(opt.bert_path)
        self.linear=nn.Linear(opt.bert_hidden_size,opt.class_num,bias=False)
        self.crossentry=nn.CrossEntropyLoss()
        self.dropout=nn.Dropout(opt.dropout)
        opt.initializer(self.linear.weight.data,gain=1.414)
    
    def forward(self,data):
        labels=[i['labels'] for i in data]
        sents=[i['sents'] for i in data]
        max_sequence=max([len(sent) for sent in sents])
        sents,word_bert_bpe_words=self.vocab.sents2ids(sents)
        new_labels=[]
        for num,word_bert_bpe_word in enumerate(word_bert_bpe_words):
            new=[]
            for i in word_bert_bpe_word:
                new.extend([labels[num][i]]*len(word_bert_bpe_word[i]))
            new_labels.append(new)
        new_labels=torch.LongTensor(new_labels).to(self.opt.device)
        attention_mask=[]
        token_type_ids=[]
        for text in sents:
            attention_mask.append([1]*len(text))
            token_type_ids.append([0]*len(text))
        sents=myDataset.to_input_tensor(sents).to(self.opt.device)
        attention_mask=myDataset.to_input_tensor(attention_mask).to(self.opt.device)
        token_type_ids=myDataset.to_input_tensor(token_type_ids).to(self.opt.device)
        bert_out=self.bert(
            input_ids=sents,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output=bert_out[0][-1]
        pred=self.linear(output)
        pred=pred.squeeze(0)[1:-1]
        new_labels=new_labels.squeeze(0)[1:-1]
        loss=self.crossentry(pred,new_labels)
        return pred,loss
    
    def predict(self,sents):
        max_sequence=len(sents)
        sents,word_bert_bpe_words=self.vocab.sents2ids([sents])
        attention_mask=[]
        token_type_ids=[]
        for text in sents:
            attention_mask.append([1]*len(text))
            token_type_ids.append([0]*len(text))
        sents=myDataset.to_input_tensor(sents).to(self.opt.device)
        attention_mask=myDataset.to_input_tensor(attention_mask).to(self.opt.device)
        token_type_ids=myDataset.to_input_tensor(token_type_ids).to(self.opt.device)
        bert_out=self.bert(
            input_ids=sents,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output=bert_out[0][-1]
        pred=self.linear(output)
        pred=pred.squeeze(0)
        batch_embedding=[]
        for i in range(max_sequence):
            batch_embedding.append(pred[word_bert_bpe_words[0][i][0]])
        batch_embedding=torch.tensor([embedding.cpu().numpy() for embedding in batch_embedding]).to(self.opt.device)#[sequence_length,bert_hidden_size]
        _,result=torch.max(batch_embedding,1)
        return result