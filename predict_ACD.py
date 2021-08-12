import torch

def getAspectResult(sents,result):
    aspect=[]
    assert len(sents)==len(result)
    for index,value in enumerate(result):
        if value==1:
            temp=[sents[index]]
            tempIndex=index+1
            while tempIndex<len(sents) and result[tempIndex]==2:
                temp.append(sents[tempIndex])
                tempIndex=tempIndex+1
            aspect.append(' '.join(temp))
    return aspect

def predict_ACD(model,sents,aspects,k,vocab):
    with torch.no_grad():
        wordSents=vocab.indices2words(sents)
        wordaspects=[]
        results=[]
        for num,sent in enumerate(sents):
            attens=model.predict(sent,aspects[num])
            index=[]
            for i in range(k):
                max_value=-999
                value_index=-1
                for num2,value in enumerate(attens):
                    if value>max_value:
                        max_value=value
                        value_index=num2
                index.append(value_index)
                attens[value_index]=-999
            result=[1 if inde in index else 0 for inde in range(len(attens))]
            results.append(result)
            wordaspects.append(getAspectResult(wordSents[num],result))
        return results,wordaspects