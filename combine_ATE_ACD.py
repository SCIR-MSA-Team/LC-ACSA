import torch
import numpy as np
import copy
from calculate_similarity import calculate_similarity,similarity,getGlove
from predict_ATE import predict as predict_ATE
from predict_ACD import predict_ACD

def connectTermWithCategory4ATE(opt,vocab,word2id,word_vecs,aspects_ATE,PreDefinecategory,category2id,ATE_input_aspect):
    similarity=[calculate_similarity(word2id,word_vecs,asp,PreDefinecategory,300) for asp in aspects_ATE]
    ATE_connect_result=[]
    for num,sim in enumerate(similarity):
        category_aspect=[[] for i in range(len(vocab.aspects))]
        if len(sim)>0:
            for num2,si in enumerate(sim):
                cate=np.argmax(si)
                category_aspect[cate].append(aspects_ATE[num][num2])
            ATE_connect_result.append(category_aspect[category2id[vocab.words[ATE_input_aspect[num]]]])
        else:
            ATE_connect_result.append([])
    return ATE_connect_result

def findMostConnectedTerm4ACD(word2id,word_vecs,aspects_ACD,ACD_input_aspects):
    similarity=[calculate_similarity(word2id,word_vecs,asp,[input_aspects],300) for (asp,input_aspects) in zip(aspects_ACD,ACD_input_aspects)]
    ACD_connect_result=[]
    for num,sim in enumerate(similarity):
        ACD_connect_result.append([aspects_ACD[num][np.argmax(sim)]])
    return ACD_connect_result

def mapDataset2ATEpath(dataset):
    paths={
        'rest14DevSplit':'./pkl_files_ATE/model_allRest_bert_pt.pkl',
        'rest14_hard':'./pkl_files_ATE/model_allRest_bert_pt.pkl',
        'rest_large':'./pkl_files_ATE/model_allRest_bert_pt.pkl',
        'rest_large_hard':'./pkl_files_ATE/model_allRest_bert_pt.pkl',
    }
    return paths[dataset]

def mapDataset2ACDpath(dataset):
    paths={
        'rest14DevSplit':'./pkl_files_ACSC/model_rest14DevSplit_ACD_Bert.pkl',
        'rest14_hard':'./pkl_files_ACSC/model_rest14DevSplit_ACD_Bert.pkl',
        'rest_large':'./pkl_files_ACSC/model_rest_large_ACD_Bert.pkl',
        'rest_large_hard':'./pkl_files_ACSC/model_rest_large_hard_ACD_Bert.pkl',
    }
    return paths[dataset]

def extractTerm_combine_ATE_ACD(sents,aspects,vocab,opt):
    word_sents=vocab.indices2words(sents)
    CLS_sents_ATE=copy.deepcopy(word_sents)
    for i in CLS_sents_ATE:
        i.insert(0,'[CLS]')
        i.append('[SEP]')
    model_ATE_path=mapDataset2ATEpath(opt.dataset)
    model_ATE=torch.load(model_ATE_path)
    model_ACD_path=mapDataset2ACDpath(opt.dataset)
    word2id,word_vecs=vocab.word2id,vocab.word_vecs
    PreDefinecategory=[vocab.words[asp] for asp in vocab.aspects]
    category2id={}
    for num,cate in enumerate(vocab.aspectOrder.keys()):
        category2id[vocab.words[cate]]=num
    model_ACD=torch.load(model_ACD_path)
    results_ATE,aspects_ATE=predict_ATE(model_ATE,CLS_sents_ATE)
    all_term_index=[i.cpu().numpy()[1:-1] for i in results_ATE]
    ATE_connect_result=connectTermWithCategory4ATE(opt,vocab,word2id,word_vecs,aspects_ATE,PreDefinecategory,category2id,aspects)
    final_result=ATE_connect_result
    if opt.UseACD==1:
        CantMatch=[]
        idmap={}
        for num,ate_result in enumerate(ATE_connect_result):
            if ate_result==[]:
                idmap[num]=len(CantMatch)
                CantMatch.append(num)
        ACD_input_Sents=[sents[inde] for inde in CantMatch]
        ACD_input_aspects=[aspects[inde] for inde in CantMatch]
        results_ACD,aspects_ACD=predict_ACD(model_ACD,ACD_input_Sents,ACD_input_aspects,opt.k,vocab)
        ACD_input_aspects=[vocab.words[i] for i in ACD_input_aspects]
        ACD_connect_result=findMostConnectedTerm4ACD(word2id,word_vecs,aspects_ACD,ACD_input_aspects)
        final_result=[ATE_connect_result[inde] if inde not in CantMatch else ACD_connect_result[idmap[inde]] for inde in range(len(sents))]
        all_term_index=[all_term_index[inde] if inde not in CantMatch else results_ACD[idmap[inde]] for inde in range(len(sents))]
    final_term_index=[]
    for num,sent in enumerate(word_sents):
        temp_term=final_result[num]
        term=[]
        for num2,te in enumerate(temp_term):
            if(te.split()==1):
                term.append(te)
            else:
                term.extend(te.split())
        term_index=[1 if sent[inde] in term else 0 for inde in range(len(sent))]
        assert len(term_index)==len(sent)
        final_term_index.append(term_index)
    return final_result,final_term_index