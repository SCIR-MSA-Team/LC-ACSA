import torch
import torch.nn as nn
from torch.nn import init
import vocab_ACSC
from transformers import BertModel,BertTokenizer
import numpy as np
import torch.nn.functional as F
from data_utils import myDataset,reset_params

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

def masked_softmax(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int = -1,
    memory_efficient: bool = False,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class AttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities

class ACD_Bert(nn.Module):
    def __init__(self,vocab,opt):
        super().__init__()
        self.vocab=vocab
        self.opt=opt
        self.bert=BertModel.from_pretrained(opt.bert_path)
        self.tokenizer=vocab_ACSC.Vocab4Bert(opt.bert_path)
        self.attention=nn.ModuleList([AttentionInHtt(opt.hidden_size_bert,opt.hidden_size) for i in range(len(self.vocab.aspects))])
        self.predicts=nn.ModuleList([nn.Linear(opt.hidden_size_bert,1) for i in range(len(vocab.aspects))])
        reset_params(self.predicts,opt.initializer)
        reset_params(self.attention,opt.initializer)

    def forward(self,batch):
        sents=[i['sents'] for i in batch]
        sents_tensor=myDataset.to_input_tensor(sents).to(self.opt.device)
        aspects=[i['aspect'] for i in batch]
        sents=self.vocab.indices2words(sents)
        for i in sents:
            i.insert(0,'[CLS]')
            i.append('[SEP]')
        max_length=max([len(sent) for sent in sents])
        sents=[' '.join(sent) for sent in sents]
        token_ids,word_bert_bpe_words=self.tokenizer.sents2ids(sents)
        attention_mask=[]
        token_type_ids=[]
        for token_id in token_ids:
            attention_mask.append([1]*len(token_id))
            token_type_ids.append([0]*len(token_id))
        token_ids=myDataset.to_input_tensor(token_ids).to(self.opt.device)
        attention_mask=myDataset.to_input_tensor(attention_mask).to(self.opt.device)
        token_type_ids=myDataset.to_input_tensor(token_type_ids).to(self.opt.device)
        output=self.bert(input_ids=token_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        bpe_embedding=[]
        for batch in range(output[0].shape[0]):
            batch_embedding=[]
            for word_index in range(1,sents_tensor.shape[1]+1):
                if word_index in word_bert_bpe_words[batch]:
                    batch_embedding.append(output[0][batch,word_bert_bpe_words[batch][word_index][0],:].unsqueeze(0))
                else:
                    batch_embedding.append(torch.zeros(self.opt.hidden_size_bert).unsqueeze(0).to(self.opt.device))
            batch_embedding=torch.cat(batch_embedding,0)
            bpe_embedding.append(batch_embedding.unsqueeze(0))
        bpe_embedding=torch.cat(bpe_embedding,0)
        mask=(sents_tensor!=0).float()
        attens=[func(bpe_embedding,mask) for func in self.attention]
        r=[torch.bmm(atten.unsqueeze(1),bpe_embedding).squeeze(1) for atten in attens]
        y=torch.stack([func(r[num]) for num,func in enumerate(self.predicts)]).squeeze(-1).permute(1,0)
        zeros=torch.zeros(token_ids.shape[0],len(self.vocab.aspects))
        for i in range(len(aspects)):
            zeros[i][self.vocab.aspectOrder[aspects[i]]]=1
        crossentry=nn.BCEWithLogitsLoss()
        zeros=zeros.to(self.opt.device)
        loss=crossentry(y,zeros)
        return y,loss
    
    def predict(self,sents,aspects):
        sents=[sents]
        sents_tensor=torch.LongTensor(sents).to(self.opt.device)
        aspects=[aspects]
        sents=self.vocab.indices2words(sents)
        for i in sents:
            i.insert(0,'[CLS]')
            i.append('[SEP]')
        max_length=max([len(sent) for sent in sents])
        sents=[' '.join(sent) for sent in sents]
        token_ids,word_bert_bpe_words=self.tokenizer.sents2ids(sents)
        attention_mask=[]
        token_type_ids=[]
        for token_id in token_ids:
            attention_mask.append([1]*len(token_id))
            token_type_ids.append([0]*len(token_id))
        token_ids=torch.LongTensor(token_ids).to(self.opt.device)
        attention_mask=torch.LongTensor(attention_mask).to(self.opt.device)
        token_type_ids=torch.LongTensor(token_type_ids).to(self.opt.device)
        output=self.bert(input_ids=token_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        bpe_embedding=[]
        for batch in range(output[0].shape[0]):
            batch_embedding=[]
            for word_index in range(1,sents_tensor.shape[1]+1):
                if word_index in word_bert_bpe_words[batch]:
                    batch_embedding.append(output[0][batch,word_bert_bpe_words[batch][word_index][0],:].unsqueeze(0))
                else:
                    batch_embedding.append(torch.zeros(self.opt.hidden_size_bert).unsqueeze(0).to(self.opt.device))
            batch_embedding=torch.cat(batch_embedding,0)
            bpe_embedding.append(batch_embedding.unsqueeze(0))
        bpe_embedding=torch.cat(bpe_embedding,0)
        mask=(sents_tensor!=0).float()
        attens=[func(bpe_embedding,mask) for func in self.attention]
        attens=torch.stack(attens).squeeze(1)
        atten=attens[self.vocab.aspectOrder[aspects[0]]]
        return atten