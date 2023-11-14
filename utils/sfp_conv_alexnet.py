import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.sfp_quant import *
#np.set_printoptions(threshold=np.inf)

def conv2d_Q_alexnet(w_bit, Kw, Ka):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self,in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, stride=1, 
                padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)    
      #print(self.Kw)
      #print(self.Ka)

    def forward(self, input, order=None):
      self.input_q = self.quantize_fn(input/self.Ka) #量化并缩放激活值
      self.weight_q = self.quantize_fn(self.weight/self.Kw)  #量化并缩放权重
      self.bias_q = self.bias / self.Ka / self.Kw
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias_q, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return Conv2d_Q

def linear_Q_alexnet(w_bit, Kw, Ka):      #全连接层
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, Kw=Kw, Ka=Ka, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)
      #print(self.Kw)
      #print(self.Ka)

    def forward(self, input):
      self.input_q = self.quantize_fn(input/self.Ka)
      self.weight_q = self.quantize_fn(self.weight/self.Kw)
      self.bias_q = self.bias /self.Kw /self.Ka
      out =  F.linear(self.input_q, self.weight_q, self.bias_q)*self.Kw*self.Ka
      return out
  return Linear_Q
