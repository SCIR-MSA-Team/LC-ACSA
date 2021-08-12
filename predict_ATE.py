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

def predict(model,sents):
    model.eval()
    results=[]
    aspects=[]
    with torch.no_grad():
        for sent in sents:
            result=model.predict(sent)
            assert len(sent)==len(result)
            results.append(result)
            aspect=getAspectResult(sent[1:-1],result[1:-1])
            aspects.append(aspect)
        return results,aspects
