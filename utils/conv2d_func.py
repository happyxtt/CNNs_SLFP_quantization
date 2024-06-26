import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.sfp_quant import *
#np.set_printoptions(threshold=np.inf)

def conv2d_Q(q_bit, Kw, Ka):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self, in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, 
    stride=1, padding=0, dilation=1, groups=1, bias=False):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)

    def forward(self, input, order=None):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw) 
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return Conv2d_Q

def conv2d_Q_bias(q_bit, Kw, Ka):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self,in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, stride=1, padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, groups, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)    
      # print(self.Kw)
      # print(self.Ka)

    def forward(self, input, order=None):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw) 
      self.bias_q = self.bias / self.Ka / self.Kw
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias_q, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka*self.Kw
      return self.output
  return Conv2d_Q

def linear_Q(q_bit, Kw, Ka):   
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, Kw=Kw, Ka=Ka, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.q_bit = q_bit
      self.quantize_weight = weight_quantize_func(q_bit=q_bit)
      self.quantize_act = act_quantize_func(q_bit=q_bit)
      self.Kw = torch.tensor(Kw)
      self.Ka = torch.tensor(Ka)

    def forward(self, input):
      self.input_q = self.quantize_act(input/self.Ka)
      self.weight_q = self.quantize_weight(self.weight/self.Kw)
      self.bias_q = self.bias /self.Kw /self.Ka
      out =  F.linear(self.input_q, self.weight_q, self.bias_q)*self.Kw*self.Ka
      return out
  return Linear_Q

if __name__ == '__main__':
  a = torch.rand(1, 3, 32, 32)
  Conv2d = conv2d_Q_fn(q_bit=1)
  conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
  img = torch.randn(1, 256, 56, 56)
  print(img.max().item(), img.min().item())
  out = conv(img)
  print(out.max().item(), out.min().item())