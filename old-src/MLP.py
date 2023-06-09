import torch
from torch import nn
from torch.nn import Linear,Dropout,Sequential,LayerNorm
from torch.nn.init import xavier_normal_,zeros_,constant_
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid=len(nl)-1
        sumdrop=np.sum(dropout_p)
        modules=[]
        for i in range(1,nhid):
            modules.append(actfun(nl[i-1],nl[i]))
            if table_norm: modules.append(LayerNorm(nl[i]))
            if sumdrop>=0.0001: modules.append(Dropout(p=dropout_p[i-1]))
            #bias = not(i==nhid-1)
            linear=Linear(nl[i],nl[i+1])
            if i==nhid-1: 
                zeros_(linear.weight)
            else:
                xavier_normal_(linear.weight)
            if i==nhid-1: zeros_(linear.bias)
            modules.append(linear)
        self.resblock=Sequential(*modules)

    def forward(self, x):
        return self.resblock(x) + x

#==================for get the atomic energy=======================================
class NNMod(torch.nn.Module):
   def __init__(self,outputneuron,nblock,nl,dropout_p,actfun,RBF=None,table_norm=True):
      """
      nl: is the neural network structure;
      outputneuron: the number of output neuron of neural network
      """
      super(NNMod,self).__init__()
      # create the structure of the nn     
      self.outputneuron=outputneuron
      sumdrop=np.sum(dropout_p)
      with torch.no_grad():
          nl.append(nl[1])
          nhid=len(nl)-1
          modules=[]
          linear=Linear(nl[0],nl[1])
          xavier_normal_(linear.weight)
          modules.append(linear)
          for iblock in range(nblock):
              modules.append( * [ResBlock(nl,dropout_p,actfun,table_norm=table_norm)])
          modules.append(actfun(nl[nhid-1],nl[nhid]))
          if table_norm: modules.append(LayerNorm(nl[nhid]))
          #if table_norm: modules.append(LayerNorm(nl[nhid]))
          if RBF is not None: modules.append(RBF(nl[nhid],nl[nhid]))
          linear=Linear(nl[nhid],self.outputneuron)
          #zeros_(linear.weight)
          modules.append(linear)
      self.nets = Sequential(*modules)

#   @pysnooper.snoop('out',depth=2)   for debug
   def forward(self,density):    
      # elements: dtype: LongTensor store the index of elements of each center atom
      return self.nets(density)
