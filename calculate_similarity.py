import math
import numpy as np

def similarity(a,b):
    numerator=sum(x*y for x,y in zip(a,b))
    denominator_a=math.sqrt(sum(x**2 for x in a))
    denominator_b=math.sqrt(sum(x**2 for x in b))
    return numerator/(denominator_a*denominator_b)

def getGlove(trainPredictAspect,devPredictAspect,testPredictAspect,category,glove_path,embedding_dim):
    words=[]
    inGlove=[]
    for i in trainPredictAspect+devPredictAspect+testPredictAspect:
        for j in i:
            if len(j.split())==1:
                if j not in words:
                    words.append(j)
            else:
                j_split=j.split()
                for k in j_split:
                    if k not in words:
                        words.append(k)
    for i in category:
        if i not in words:
            words.append(i)
    reverse=lambda x:dict(zip(x,range(len(x))))
    word2id=reverse(words)
    word_vecs=np.zeros((len(words),300))
    with open(glove_path) as fp:
        for line in fp.readlines():
            line=line.split(" ")
            word=line[0]
            if word in words:
                inGlove.append(word)
                word_vecs[word2id[word]]=np.array([float(x) for x in line[1:]])
    for i in words:
        if i not in inGlove:
            print(i)
    return word2id,word_vecs

def calculate_similarity(word2id,word_vecs,predict_aspect,category,embedding_dim=300):
    similarity_value=[]
    if predict_aspect!=[]:
        for aspect in predict_aspect:
            temp=[]
            aspect_split=aspect.split()
            aspect_embedding=np.zeros(embedding_dim)
            if len(aspect_split)==1:
                aspect_embedding=word_vecs[word2id[aspect]]
            else:
                for asp in aspect_split:
                    aspect_embedding=word_vecs[word2id[asp]]+aspect_embedding
            for cate in category:
                sim=similarity(aspect_embedding,word_vecs[word2id[cate]])
                temp.append(sim)
            similarity_value.append(temp)
    return similarity_value
    