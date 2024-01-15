import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import torchsummary
import math
import sys
sys.path.append('..')
from utils.sfp_quant import *   
from utils.slfp_conv_shufflenetv2 import *

"""shufflenetv2 in pytorch: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/shufflenetv2.py

[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)

def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class ShuffleUnit(nn.Module):

    def __init__(self, wbit ,in_channels, out_channels, stride, Kw, Ka):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv2d_0 = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_1 = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[1], Ka = Ka[1])
        Conv2d_2 = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[2], Ka = Ka[2])
        Conv2d_3 = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[3], Ka = Ka[3])
        Conv2d_4 = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[4], Ka = Ka[4])

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                Conv2d_3(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_4(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x

class ShuffleNetV2(nn.Module):

    def __init__(self, wbit,ratio=1, class_num=100):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        #########  Ka, Kw  ########
        #Kw = np.arange(60, dtype=float)
        #Kw = (Kw + 10)/10
        #Ka = Kw
        ## shufflenet 的 Ka Kw
        ka = [2.7537312507629395, 7.555527210235596, 4.81114387512207, 11.094647407531738, 7.555527210235596, 15.242570877075195, 8.979893684387207, 7.511760234832764, 11.678345680236816, 8.53162956237793, 7.105198383331299, 9.260470390319824, 10.607419967651367, 6.446119785308838, 8.823225975036621, 12.3921537399292, 8.17046070098877, 6.96644401550293, 12.3921537399292, 10.547478675842285, 4.910508155822754, 5.171605110168457, 6.637495040893555, 5.024521350860596, 4.97715950012207, 7.018732070922852, 4.330652236938477, 4.495896339416504, 6.299440860748291, 4.963755130767822, 4.165472030639648, 6.0865607261657715, 4.284519672393799, 4.1714701652526855, 6.806074142456055, 4.733305931091309, 4.395836353302002, 7.121100902557373, 5.61025333404541, 5.25752592086792, 8.024835586547852, 5.76422119140625, 5.705770015716553, 7.765516757965088, 5.76422119140625, 8.19332504272461, 4.528226852416992, 4.810886859893799, 6.3730363845825195, 4.781790733337402, 4.570272922515869, 6.727193832397461, 4.446099281311035, 4.341022491455078, 6.815759658813477, 4.050592422485352]+ [1]*10 # 57个数，56层，最后一个是最后一层的输出
        Ka = np.array(ka)/15.5
        kw = [0.7558735013008118, 0.49423471093177795, 0.5136241316795349, 0.60989910364151, 0.5862343907356262, 0.6468716263771057, 0.49205535650253296, 0.4595864415168762, 0.5098061561584473, 0.33036527037620544, 0.35325777530670166, 0.35556748509407043, 0.5581929087638855, 0.3229704797267914, 0.4010358154773712, 0.29787009954452515, 0.288500040769577, 0.25207117199897766, 0.317177951335907, 0.3287501931190491, 0.16998499631881714, 0.30774733424186707, 0.27996766567230225, 0.2027178555727005, 0.30330315232276917, 0.23482947051525116, 0.22493161261081696, 0.2888806462287903, 0.19320374727249146, 0.24502170085906982, 0.2639356255531311, 0.22498929500579834, 0.2088075429201126, 0.27295875549316406, 0.2054140269756317, 0.20450830459594727, 0.24333742260932922, 0.1962268203496933, 0.28548988699913025, 0.2538895308971405, 0.23348627984523773, 0.16584959626197815, 0.23481036722660065, 0.22298164665699005, 0.23057013750076294, 0.25494927167892456, 0.14327891170978546, 0.22803305089473724, 0.14990486204624176, 0.14703013002872467, 0.24455490708351135, 0.1472909301519394, 0.16691987216472626, 0.22616896033287048, 0.14078955352306366, 0.0846848413348198] + [1]*10 
        Kw = np.array(kw)/15.5     
        ##############

        Conv2d = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw, Ka = Ka)
        Conv_last = conv2d_Q_shufflenet(w_bit = wbit, Kw = Kw[55], Ka = Ka[55])
        self.pre = nn.Sequential(
            Conv2d(3, 24, 3, padding=1, Kw = Kw[0], Ka = Ka[0]),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(wbit, 24, out_channels[0], 3, 6, Kw, Ka)
        self.stage3 = self._make_stage(wbit, out_channels[0], out_channels[1], 7, 6, Kw[14:], Ka[14:])
        self.stage4 = self._make_stage(wbit, out_channels[1], out_channels[2], 3, 6, Kw[40:], Ka[40:])
        self.conv5 = nn.Sequential(
            Conv_last(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], class_num)

    def get_layer_inputs(self):
        return self.layer_inputs
    
    def get_layer_outputs(self):
        return self.layer_outputs
    
    def reset_layer_inputs_outputs(self):
        self.layer_inputs = {}
        self.layer_outputs = {}

    def get_layer_weights(self):
        return self.layer_weights
    
    def reset_layer_weights(self):
        self.layer_weights = {}

    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        '''
        self.layer_inputs[0] = self.pre[0].input_q
        self.layer_weights[0] = self.pre[0].weight_q
        #self.layer_outputs[0] = self.pre[1].output

        self.layer_inputs[1] = self.stage2[0]. residual[0].input_q
        self.layer_weights[1] = self.stage2[0].residual[0].weight_q
        #self.layer_outputs[1] = self.stage2[0].residual[2].output

        self.layer_inputs[2] = self.stage2[0]. residual[3].input_q
        self.layer_weights[2] = self.stage2[0].residual[3].weight_q
        #self.layer_outputs[2] = self.stage2[0].residual[4].output

        self.layer_inputs[3] = self.stage2[0]. residual[5].input_q
        self.layer_weights[3] = self.stage2[0].residual[5].weight_q
        #self.layer_outputs[3] = self.stage2[0].residual[7].output

        self.layer_inputs[4] = self.stage2[0]. shortcut[0].input_q
        self.layer_weights[4] = self.stage2[0].shortcut[0].weight_q
        #self.layer_outputs[4] = self.stage2[0].shortcut[1].output

        self.layer_inputs[5] = self.stage2[0]. shortcut[2].input_q
        self.layer_weights[5] = self.stage2[0].shortcut[2].weight_q
        #self.layer_outputs[5] = self.stage2[0].shortcut[4].output

        self.layer_inputs[6] = self.stage2[1]. residual[0].input_q
        self.layer_weights[6] = self.stage2[1].residual[0].weight_q
        #self.layer_outputs[6] = self.stage2[1].residual[2].output

        self.layer_inputs[7] = self.stage2[1]. residual[3].input_q
        self.layer_weights[7] = self.stage2[1].residual[3].weight_q
        #self.layer_outputs[7] = self.stage2[1].residual[4].output

        self.layer_inputs[8] = self.stage2[1]. residual[5].input_q
        self.layer_weights[8] = self.stage2[1].residual[5].weight_q
        #self.layer_outputs[8] = self.stage2[1].residual[7].output

        self.layer_inputs[9] = self.stage2[2]. residual[0].input_q
        self.layer_weights[9] = self.stage2[2].residual[0].weight_q
        #self.layer_outputs[9] = self.stage2[2].residual[2].output

        self.layer_inputs[10] = self.stage2[2]. residual[3].input_q
        self.layer_weights[10] = self.stage2[2].residual[3].weight_q
        #self.layer_outputs[10] = self.stage2[2].residual[4].output

        self.layer_inputs[11] = self.stage2[2]. residual[5].input_q
        self.layer_weights[11] = self.stage2[2].residual[5].weight_q
        #self.layer_outputs[11] = self.stage2[2].residual[7].output

        self.layer_inputs[12] = self.stage2[3]. residual[0].input_q
        self.layer_weights[12] = self.stage2[3].residual[0].weight_q
        #self.layer_outputs[12] = self.stage2[3].residual[2].output

        self.layer_inputs[13] = self.stage2[3]. residual[3].input_q
        self.layer_weights[13] = self.stage2[3].residual[3].weight_q
        #self.layer_outputs[13] = self.stage2[3].residual[4].output

        self.layer_inputs[14] = self.stage2[3]. residual[5].input_q
        self.layer_weights[14] = self.stage2[3].residual[5].weight_q
        #self.layer_outputs[14] = self.stage2[3].residual[7].output

        self.layer_inputs[15] = self.stage3[0]. residual[0].input_q
        self.layer_weights[15] = self.stage3[0].residual[0].weight_q
        #self.layer_outputs[15] = self.stage3[0].residual[2].output

        self.layer_inputs[16] = self.stage3[0]. residual[3].input_q
        self.layer_weights[16] = self.stage3[0].residual[3].weight_q
        #self.layer_outputs[16] = self.stage3[0].residual[4].output

        self.layer_inputs[17] = self.stage3[0]. residual[5].input_q
        self.layer_weights[17] = self.stage3[0].residual[5].weight_q
        #self.layer_outputs[17] = self.stage3[0].residual[7].output

        self.layer_inputs[18] = self.stage3[0]. shortcut[0].input_q
        self.layer_weights[18] = self.stage3[0].shortcut[0].weight_q
        #self.layer_outputs[18] = self.stage3[0].shortcut[2].output

        self.layer_inputs[19] = self.stage3[0]. shortcut[2].input_q
        self.layer_weights[19] = self.stage3[0].shortcut[2].weight_q
        #self.layer_outputs[19] = self.stage3[0].shortcut[2].output

        self.layer_inputs[20] = self.stage3[1]. residual[0].input_q
        self.layer_weights[20] = self.stage3[1].residual[0].weight_q
        #self.layer_outputs[20] = self.stage3[1].residual[2].output

        self.layer_inputs[21] = self.stage3[1]. residual[3].input_q
        self.layer_weights[21] = self.stage3[1].residual[3].weight_q
        #self.layer_outputs[21] = self.stage3[1].residual[4].output

        self.layer_inputs[22] = self.stage3[1]. residual[5].input_q
        self.layer_weights[22] = self.stage3[1].residual[5].weight_q
        #self.layer_outputs[22] = self.stage3[1].residual[7].output

        self.layer_inputs[23] = self.stage3[2]. residual[0].input_q
        self.layer_weights[23] = self.stage3[2].residual[0].weight_q
        #self.layer_outputs[23] = self.stage3[2].residual[2].output

        self.layer_inputs[24] = self.stage3[2]. residual[3].input_q
        self.layer_weights[24] = self.stage3[2].residual[3].weight_q
        #self.layer_outputs[24] = self.stage3[2].residual[4].output

        self.layer_inputs[25] = self.stage3[2]. residual[5].input_q
        self.layer_weights[25] = self.stage3[2].residual[5].weight_q
        #self.layer_outputs[25] = self.stage3[2].residual[7].output

        self.layer_inputs[26] = self.stage3[3]. residual[0].input_q
        self.layer_weights[26] = self.stage3[3].residual[0].weight_q
        #self.layer_outputs[26] = self.stage3[3].residual[2].output

        self.layer_inputs[27] = self.stage3[3]. residual[3].input_q
        self.layer_weights[27] = self.stage3[3].residual[3].weight_q
        #self.layer_outputs[27] = self.stage3[3].residual[4].output

        self.layer_inputs[28] = self.stage3[3]. residual[5].input_q
        self.layer_weights[28] = self.stage3[3].residual[5].weight_q
        #self.layer_outputs[28] = self.stage3[3].residual[7].output

        self.layer_inputs[29] = self.stage3[4]. residual[0].input_q
        self.layer_weights[29] = self.stage3[4].residual[0].weight_q
        #self.layer_outputs[29] = self.stage3[4].residual[2].output

        self.layer_inputs[30] = self.stage3[4]. residual[3].input_q
        self.layer_weights[30] = self.stage3[4].residual[3].weight_q
        #self.layer_outputs[30] = self.stage3[4].residual[4].output

        self.layer_inputs[31] = self.stage3[4]. residual[5].input_q
        self.layer_weights[31] = self.stage3[4].residual[5].weight_q
        #self.layer_outputs[31] = self.stage3[4].residual[7].output

        self.layer_inputs[32] = self.stage3[5]. residual[0].input_q
        self.layer_weights[32] = self.stage3[5].residual[0].weight_q
        #self.layer_outputs[32] = self.stage3[5].residual[2].output

        self.layer_inputs[33] = self.stage3[5]. residual[3].input_q
        self.layer_weights[33] = self.stage3[5].residual[3].weight_q
        #self.layer_outputs[33] = self.stage3[5].residual[4].output

        self.layer_inputs[34] = self.stage3[5]. residual[5].input_q
        self.layer_weights[34] = self.stage3[5].residual[5].weight_q
        #self.layer_outputs[34] = self.stage3[5].residual[7].output

        self.layer_inputs[35] = self.stage3[6]. residual[0].input_q
        self.layer_weights[35] = self.stage3[6].residual[0].weight_q

        self.layer_inputs[36] = self.stage3[6]. residual[3].input_q
        self.layer_weights[36] = self.stage3[6].residual[3].weight_q

        self.layer_inputs[37] = self.stage3[6]. residual[5].input_q
        self.layer_weights[37] = self.stage3[6].residual[5].weight_q

        self.layer_inputs[38] = self.stage3[7]. residual[0].input_q
        self.layer_weights[38] = self.stage3[7].residual[0].weight_q

        self.layer_inputs[39] = self.stage3[7]. residual[3].input_q
        self.layer_weights[39] = self.stage3[7].residual[3].weight_q

        self.layer_inputs[40] = self.stage3[7]. residual[5].input_q
        self.layer_weights[40] = self.stage3[7].residual[5].weight_q

        self.layer_inputs[41] = self.stage4[0]. residual[0].input_q
        self.layer_weights[41] = self.stage4[0].residual[0].weight_q

        self.layer_inputs[42] = self.stage4[0]. residual[3].input_q
        self.layer_weights[42] = self.stage4[0].residual[3].weight_q

        self.layer_inputs[43] = self.stage4[0]. residual[5].input_q
        self.layer_weights[43] = self.stage4[0].residual[5].weight_q

        self.layer_inputs[44] = self.stage4[0]. shortcut[0].input_q
        self.layer_weights[44] = self.stage4[0].shortcut[0].weight_q

        self.layer_inputs[45] = self.stage4[0]. shortcut[2].input_q
        self.layer_weights[45] = self.stage4[0].shortcut[2].weight_q

        self.layer_inputs[46] = self.stage4[1]. residual[0].input_q
        self.layer_weights[46] = self.stage4[1].residual[0].weight_q

        self.layer_inputs[47] = self.stage4[1]. residual[3].input_q
        self.layer_weights[47] = self.stage4[1].residual[3].weight_q

        self.layer_inputs[48] = self.stage4[1]. residual[5].input_q
        self.layer_weights[48] = self.stage4[1].residual[5].weight_q

        self.layer_inputs[49] = self.stage4[2]. residual[0].input_q
        self.layer_weights[49] = self.stage4[2].residual[0].weight_q

        self.layer_inputs[50] = self.stage4[2]. residual[3].input_q
        self.layer_weights[50] = self.stage4[2].residual[3].weight_q

        self.layer_inputs[51] = self.stage4[2]. residual[5].input_q
        self.layer_weights[51] = self.stage4[2].residual[5].weight_q

        self.layer_inputs[52] = self.stage4[3]. residual[0].input_q
        self.layer_weights[52] = self.stage4[3].residual[0].weight_q

        self.layer_inputs[53] = self.stage4[3]. residual[3].input_q
        self.layer_weights[53] = self.stage4[3].residual[3].weight_q

        self.layer_inputs[54] = self.stage4[3]. residual[5].input_q
        self.layer_weights[54] = self.stage4[3].residual[5].weight_q

        self.layer_inputs[55] = self.conv5[0].input_q
        self.layer_weights[55] = self.conv5[0].weight_q
        self.layer_outputs[55] = x
        '''
        return x

    def _make_stage(self, wbit, in_channels, out_channels, repeat, begin_num, Kw, Ka):
        layers = []
        layers.append(ShuffleUnit(wbit, in_channels, out_channels, 2, Kw[1:6], Ka[1:6]))

        while repeat:
            layers.append(ShuffleUnit(wbit, out_channels, out_channels, 1, Kw[begin_num: begin_num + 10], Ka[begin_num: begin_num + 10]))
            begin_num = begin_num + 3
            repeat -= 1

        return nn.Sequential(*layers)

def shufflenetv2():
    return ShuffleNetV2()

    
if __name__ == "__main__":
    """Testing
    """
    #model = ShuffleNetV2()
    #print(model)

    features = []
    def hook(self, input, output):
        #print(output.data.cpu().numpy().shape)
        features.append(output.data.cpu().numpy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = MobileNetV3_Large(32,32).to(device)
    model = ShuffleNetV2(32).to(device)
    print(model)
    torchsummary.summary(model, (3,32,32),device = "cuda")