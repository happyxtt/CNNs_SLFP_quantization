import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def STLFunction():  
  class stl(torch.autograd.Function):  # 和继承nn.Module比，autograd.Function还能自定义backward

    @staticmethod             
    def forward(ctx, x):           # 用ctx而不是self。保存上下文，forward里的变量backward能继续用
        out = torch.where(torch.abs(x) <= 1, x, torch.sign(x) * (torch.log(torch.abs(x)) + 1))
        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE,do nothing in backward (quantized)
      grad_input = grad_output.clone()  #.clone: 使拷贝的类型和等号左边一致（不然是和grad_output一致）
      grad_input = torch.where(torch.abs(grad_output) <= 1, 1, 1/torch.abs(grad_output)) * grad_output
      return grad_input
  return stl().apply


class STL(nn.Module):    
  def __init__(self):
    super(STL, self).__init__()
    self.stl = STLFunction()

  def forward(self, x):
    stlout = self.stl(x)   
    return stlout

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)