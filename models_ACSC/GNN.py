import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from data_utils import mask_logits

class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        super(GraphConvolution,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
            
    def forward(self,adjacency,input_feature):
        support=torch.matmul(input_feature,self.weight)
        d=torch.sum(adjacency,2, keepdim=True)+1
        output=torch.bmm(adjacency.float(),support.float())/d.float()
        if self.use_bias:
            output=output+self.bias
        return output
        