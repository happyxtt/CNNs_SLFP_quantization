import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#np.set_printoptions(threshold=np.inf)


def uniform_quantize(k):  #k-bit定点量化tensor
  class qfn(torch.autograd.Function):  # 和继承nn.Module比，autograd.Function还能自定义backward

    @staticmethod             
    def forward(ctx, input):           # 用ctx而不是self。保存上下文，forward里的变量backward能继续用
      
        if k == 32:
            out = input
        #elif k == 1:
        # out = torch.sign(input)
        elif k == 8:
            #out = input.clone()
            '''
            n = float(2 ** k - 1)
            out = torch.round(input / n) * n
            '''

            
            N = 8  #sfp<3,n>
            sign = torch.sign(input)
            input = torch.abs(input)
            input[input <= 0.117] = 1e-10
            input[(input >= 0.117) & (input < 0.125)] = 0.125
            input = torch.clamp(input, 0, 15)
            scaling_factor = pow(2, torch.floor(torch.log2(input)))
            out = torch.mul(sign, torch.mul(torch.round(torch.div(input, scaling_factor)*N)/N , scaling_factor))
            

            '''
            N = 16  # SLFP<3,n>
            sign = torch.sign(input)
            input = torch.abs(input)
            input[input <= 0.0625] = 1e-10
            input[(input >= 0.0625) & (input < 0.125)] = 0.125
            input = torch.clamp(input, 0, 15.3216)   #14.672   15.3216
            scaling_factor = pow(2, torch.floor(torch.log2(input)))
            out = pow(2,torch.floor(torch.log2(input)) + torch.round(torch.log2(torch.div(input,scaling_factor))*N)/N)
            out = torch.mul(sign, out)     
            '''
        
            '''
            #####  sfp<4,3>
            
            N = 8   #8
            sign = torch.sign(input)
            input = torch.clamp(torch.abs(input), 0.00390625, 240)  #240    
            scaling_factor = pow(2, torch.floor(torch.log2(input)))
            out = torch.mul(sign, torch.mul(torch.round(torch.div(input, scaling_factor)*N)/N , scaling_factor))
            '''
          
            '''
            ####  slfp<4,3>
            N = 8
            sign = torch.sign(input)                            
            input = torch.clamp(torch.abs(input), 0.00390625, 234.753) #234.753, 245.14644
            scaling_factor = pow(2, torch.floor(torch.log2(input)))
            out = pow(2,torch.floor(torch.log2(input)) + torch.round(torch.log2(torch.div(input,scaling_factor))*N)/N)
            out = torch.mul(sign, out)
            '''
        return out

    @staticmethod
    def backward(ctx, grad_output):     # STE,do nothing in backward (quantized)
      grad_input = grad_output.clone()  #.clone: 使拷贝的类型和等号左边一致（不然是和grad_output一致）
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):    #量化权重
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit                  # 传给量化模块 
    self.uniform_q = uniform_quantize(k=w_bit) # uniform_q()：执行量化

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 8:
      weight_q = self.uniform_q(x)  # W_sfp 
      
    return weight_q

def conv2d_Q_mobilenet(w_bit, Kw, Ka):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self, in_channels, out_channels, kernel_size, Kw = Kw, Ka = Ka, stride=1, 
                padding=0, dilation=1, groups=1, bias=False):
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
      self.output = F.conv2d(self.input_q, self.weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)*self.Ka * self.Kw
      return self.output
  return Conv2d_Q

def linear_Q_fn(w_bit):      #全连接层
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      self.input_q = self.quantize_fn(input/0.6350061355098602)
      self.weight_q = self.quantize_fn(self.weight/0.05926013761951077)
      self.bias_q = self.bias /0.6350061355098602 /0.05926013761951077
      out =  F.linear(self.input_q, self.weight_q, self.bias)*0.6350061355098602*0.05926013761951077
      return out
  return Linear_Q

##原本的没有缩放的卷积
def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):  
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,   #dilation: controls the spacing between the kernel points. 扩大感受野，=1 --> 标准卷积
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      input_q = self.quantize_fn(input)  #
      weight_q = self.quantize_fn(self.weight)  #
      output = F.conv2d(input_q, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
      return output 
  return Conv2d_Q


if __name__ == '__main__':
  import numpy as np

  a = torch.rand(1, 3, 32, 32)

  Conv2d = conv2d_Q_fn(w_bit=1)
  conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

  img = torch.randn(1, 256, 56, 56)
  print(img.max().item(), img.min().item())
  out = conv(img)
  print(out.max().item(), out.min().item())
