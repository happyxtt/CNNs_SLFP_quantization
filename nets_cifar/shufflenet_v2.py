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
from utils.activation_func import *
from utils.conv2d_func import *

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

    def __init__(self, qbit ,in_channels, out_channels, stride, Kw, Ka):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv2d_0 = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_1 = conv2d_Q(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
        Conv2d_2 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
        Conv2d_3 = conv2d_Q(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])
        Conv2d_4 = conv2d_Q(q_bit = qbit, Kw = Kw[4], Ka = Ka[4])

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU()
            )
            self.shortcut = nn.Sequential(
                Conv2d_3(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_4(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU()
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
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

    def __init__(self, qbit, ratio=1, class_num=100):
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
     
        ## relu
        ka = [2.7537312507629395, 5.234059810638428, 4.2632646560668945, 9.554951667785645, 5.234059810638428, 11.346541404724121, 7.264077186584473, 6.350336074829102, 9.16678237915039, 5.25019645690918, 3.478346347808838, 5.273358345031738, 6.550426959991455, 6.841591835021973, 6.551841735839844, 8.429676055908203, 3.6293373107910156, 6.033638000488281, 8.429676055908203, 8.35006332397461, 3.1058642864227295, 2.4753012657165527, 4.7964186668396, 3.1863455772399902, 2.5042595863342285, 4.836341857910156, 3.550802707672119, 2.6628284454345703, 4.511188983917236, 3.9728715419769287, 2.248741626739502, 4.604737758636475, 2.9986534118652344, 2.4434714317321777, 5.667801856994629, 3.9212722778320312, 2.7176830768585205, 5.555120944976807, 3.8605222702026367, 2.682180404663086, 5.483858585357666, 5.311583518981934, 2.37758731842041, 7.243487358093262, 5.311583518981934, 6.235666275024414, 2.933143377304077, 2.8617892265319824, 6.526449680328369, 3.827115774154663, 2.4493494033813477, 6.2418341636657715, 3.028607130050659, 2.5422754287719727, 7.466217994689941, 3.8511102199554443]+ [1]*10 
        Ka = np.array(ka)/15
        #Ka = np.ones_like(Ka)
        kw = [1.3029288053512573, 0.8977667689323425, 1.2451014518737793, 1.0910767316818237, 1.0148509740829468, 1.4486286640167236, 0.7227526307106018, 0.9099447131156921, 0.8158362507820129, 0.5097854733467102, 0.9196422696113586, 0.871035635471344, 0.9180927276611328, 0.6798731684684753, 0.9902427792549133, 0.4500037431716919, 0.336839497089386, 0.6030286550521851, 0.47783952951431274, 0.7545949220657349, 0.2900848984718323, 0.6755377054214478, 0.49563902616500854, 0.3502323627471924, 0.4965823292732239, 0.5626934766769409, 0.4495861530303955, 0.5228708386421204, 0.706261932849884, 0.4907167851924896, 0.48858827352523804, 0.49457845091819763, 0.3859078288078308, 0.4841776490211487, 0.4749361276626587, 0.625221848487854, 0.39950987696647644, 0.45098355412483215, 0.6184139847755432, 0.3726619482040405, 0.42748335003852844, 0.4202514886856079, 0.27142152190208435, 0.36484283208847046, 0.47397205233573914, 0.5499554872512817, 0.2817867696285248, 0.39557644724845886, 0.4021666347980499, 0.3414369523525238, 0.37255141139030457, 0.36935344338417053, 0.41747722029685974, 0.29923439025878906, 0.34784701466560364, 0.2244890332221985] + [1]*10 
        Kw = np.array(kw)/15
        #Kw = np.ones_like(Kw)
        #############

        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Conv_last = conv2d_Q(q_bit = qbit, Kw = Kw[55], Ka = Ka[55])
        Linear = linear_Q(q_bit=qbit, Kw = Kw[56], Ka = Ka[56])
       

        self.pre = nn.Sequential(
            Conv2d(3, 24, 3, padding=1, Kw = Kw[0], Ka = Ka[0]),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(qbit, 24, out_channels[0], 3, 6, Kw, Ka)
        self.stage3 = self._make_stage(qbit, out_channels[0], out_channels[1], 7, 6, Kw[14:], Ka[14:])
        self.stage4 = self._make_stage(qbit, out_channels[1], out_channels[2], 3, 6, Kw[40:], Ka[40:])
        self.conv5 = nn.Sequential(
            Conv_last(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU()
        )

        self.fc = Linear(out_channels[3], class_num)

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

        """ Get the input, output and weight of each layer to calculate the scale, uncommoent when pre_reference is used
        """
        '''
        self.layer_inputs[0] = self.pre[0].input_q
        self.layer_weights[0] = self.pre[0].weight_q
        self.layer_inputs[1] = self.stage2[0]. residual[0].input_q
        self.layer_weights[1] = self.stage2[0].residual[0].weight_q
        self.layer_inputs[2] = self.stage2[0]. residual[4].input_q
        self.layer_weights[2] = self.stage2[0].residual[4].weight_q
        self.layer_inputs[3] = self.stage2[0]. residual[6].input_q
        self.layer_weights[3] = self.stage2[0].residual[6].weight_q
        self.layer_inputs[4] = self.stage2[0]. shortcut[0].input_q
        self.layer_weights[4] = self.stage2[0].shortcut[0].weight_q
        self.layer_inputs[5] = self.stage2[0]. shortcut[2].input_q
        self.layer_weights[5] = self.stage2[0].shortcut[2].weight_q
        self.layer_inputs[6] = self.stage2[1]. residual[0].input_q
        self.layer_weights[6] = self.stage2[1].residual[0].weight_q
        self.layer_inputs[7] = self.stage2[1]. residual[4].input_q
        self.layer_weights[7] = self.stage2[1].residual[4].weight_q
        self.layer_inputs[8] = self.stage2[1]. residual[6].input_q
        self.layer_weights[8] = self.stage2[1].residual[6].weight_q
        self.layer_inputs[9] = self.stage2[2]. residual[0].input_q
        self.layer_weights[9] = self.stage2[2].residual[0].weight_q
        self.layer_inputs[10] = self.stage2[2]. residual[4].input_q
        self.layer_weights[10] = self.stage2[2].residual[4].weight_q
        self.layer_inputs[11] = self.stage2[2]. residual[6].input_q
        self.layer_weights[11] = self.stage2[2].residual[6].weight_q
        self.layer_inputs[12] = self.stage2[3]. residual[0].input_q
        self.layer_weights[12] = self.stage2[3].residual[0].weight_q
        self.layer_inputs[13] = self.stage2[3]. residual[4].input_q
        self.layer_weights[13] = self.stage2[3].residual[4].weight_q
        self.layer_inputs[14] = self.stage2[3]. residual[6].input_q
        self.layer_weights[14] = self.stage2[3].residual[6].weight_q
        self.layer_inputs[15] = self.stage3[0]. residual[0].input_q
        self.layer_weights[15] = self.stage3[0].residual[0].weight_q
        self.layer_inputs[16] = self.stage3[0]. residual[4].input_q
        self.layer_weights[16] = self.stage3[0].residual[4].weight_q
        self.layer_inputs[17] = self.stage3[0]. residual[6].input_q
        self.layer_weights[17] = self.stage3[0].residual[6].weight_q
        self.layer_inputs[18] = self.stage3[0]. shortcut[0].input_q
        self.layer_weights[18] = self.stage3[0].shortcut[0].weight_q
        self.layer_inputs[19] = self.stage3[0]. shortcut[2].input_q
        self.layer_weights[19] = self.stage3[0].shortcut[2].weight_q
        self.layer_inputs[20] = self.stage3[1]. residual[0].input_q
        self.layer_weights[20] = self.stage3[1].residual[0].weight_q
        self.layer_inputs[21] = self.stage3[1]. residual[4].input_q
        self.layer_weights[21] = self.stage3[1].residual[4].weight_q
        self.layer_inputs[22] = self.stage3[1]. residual[6].input_q
        self.layer_weights[22] = self.stage3[1].residual[6].weight_q
        self.layer_inputs[23] = self.stage3[2]. residual[0].input_q
        self.layer_weights[23] = self.stage3[2].residual[0].weight_q
        self.layer_inputs[24] = self.stage3[2]. residual[4].input_q
        self.layer_weights[24] = self.stage3[2].residual[4].weight_q
        self.layer_inputs[25] = self.stage3[2]. residual[6].input_q
        self.layer_weights[25] = self.stage3[2].residual[6].weight_q
        self.layer_inputs[26] = self.stage3[3]. residual[0].input_q
        self.layer_weights[26] = self.stage3[3].residual[0].weight_q
        self.layer_inputs[27] = self.stage3[3]. residual[4].input_q
        self.layer_weights[27] = self.stage3[3].residual[4].weight_q
        self.layer_inputs[28] = self.stage3[3]. residual[6].input_q
        self.layer_weights[28] = self.stage3[3].residual[6].weight_q
        self.layer_inputs[29] = self.stage3[4]. residual[0].input_q
        self.layer_weights[29] = self.stage3[4].residual[0].weight_q
        self.layer_inputs[30] = self.stage3[4]. residual[4].input_q
        self.layer_weights[30] = self.stage3[4].residual[4].weight_q
        self.layer_inputs[31] = self.stage3[4]. residual[6].input_q
        self.layer_weights[31] = self.stage3[4].residual[6].weight_q
        self.layer_inputs[32] = self.stage3[5]. residual[0].input_q
        self.layer_weights[32] = self.stage3[5].residual[0].weight_q
        self.layer_inputs[33] = self.stage3[5]. residual[4].input_q
        self.layer_weights[33] = self.stage3[5].residual[4].weight_q
        self.layer_inputs[34] = self.stage3[5]. residual[6].input_q
        self.layer_weights[34] = self.stage3[5].residual[6].weight_q
        self.layer_inputs[35] = self.stage3[6]. residual[0].input_q
        self.layer_weights[35] = self.stage3[6].residual[0].weight_q
        self.layer_inputs[36] = self.stage3[6]. residual[4].input_q
        self.layer_weights[36] = self.stage3[6].residual[4].weight_q
        self.layer_inputs[37] = self.stage3[6]. residual[6].input_q
        self.layer_weights[37] = self.stage3[6].residual[6].weight_q
        self.layer_inputs[38] = self.stage3[7]. residual[0].input_q
        self.layer_weights[38] = self.stage3[7].residual[0].weight_q
        self.layer_inputs[39] = self.stage3[7]. residual[4].input_q
        self.layer_weights[39] = self.stage3[7].residual[4].weight_q
        self.layer_inputs[40] = self.stage3[7]. residual[6].input_q
        self.layer_weights[40] = self.stage3[7].residual[6].weight_q
        self.layer_inputs[41] = self.stage4[0]. residual[0].input_q
        self.layer_weights[41] = self.stage4[0].residual[0].weight_q
        self.layer_inputs[42] = self.stage4[0]. residual[4].input_q
        self.layer_weights[42] = self.stage4[0].residual[4].weight_q
        self.layer_inputs[43] = self.stage4[0]. residual[6].input_q
        self.layer_weights[43] = self.stage4[0].residual[6].weight_q
        self.layer_inputs[44] = self.stage4[0]. shortcut[0].input_q
        self.layer_weights[44] = self.stage4[0].shortcut[0].weight_q
        self.layer_inputs[45] = self.stage4[0]. shortcut[2].input_q
        self.layer_weights[45] = self.stage4[0].shortcut[2].weight_q
        self.layer_inputs[46] = self.stage4[1]. residual[0].input_q
        self.layer_weights[46] = self.stage4[1].residual[0].weight_q
        self.layer_inputs[47] = self.stage4[1]. residual[4].input_q
        self.layer_weights[47] = self.stage4[1].residual[4].weight_q
        self.layer_inputs[48] = self.stage4[1]. residual[6].input_q
        self.layer_weights[48] = self.stage4[1].residual[6].weight_q
        self.layer_inputs[49] = self.stage4[2]. residual[0].input_q
        self.layer_weights[49] = self.stage4[2].residual[0].weight_q
        self.layer_inputs[50] = self.stage4[2]. residual[4].input_q
        self.layer_weights[50] = self.stage4[2].residual[4].weight_q
        self.layer_inputs[51] = self.stage4[2]. residual[6].input_q
        self.layer_weights[51] = self.stage4[2].residual[6].weight_q
        self.layer_inputs[52] = self.stage4[3]. residual[0].input_q
        self.layer_weights[52] = self.stage4[3].residual[0].weight_q
        self.layer_inputs[53] = self.stage4[3]. residual[4].input_q
        self.layer_weights[53] = self.stage4[3].residual[4].weight_q
        self.layer_inputs[54] = self.stage4[3]. residual[6].input_q
        self.layer_weights[54] = self.stage4[3].residual[6].weight_q
        self.layer_inputs[55] = self.conv5[0].input_q
        self.layer_weights[55] = self.conv5[0].weight_q
        self.layer_outputs[55] = x
        '''
        return x

    def _make_stage(self, qbit, in_channels, out_channels, repeat, begin_num, Kw, Ka):
        layers = []
        layers.append(ShuffleUnit(qbit, in_channels, out_channels, 2, Kw[1:6], Ka[1:6]))

        while repeat:
            layers.append(ShuffleUnit(qbit, out_channels, out_channels, 1, Kw[begin_num: begin_num + 10], Ka[begin_num: begin_num + 10]))
            begin_num = begin_num + 3
            repeat -= 1
        return nn.Sequential(*layers)


#########  shufflenet v2 swish  #########

class ShuffleUnit_swish_0(nn.Module):

    def __init__(self, qbit ,in_channels, out_channels, stride, Kw, Ka):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv2d_0 = conv2d_Q(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_1 = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
        Conv2d_2 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
        Conv2d_3 = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])
        Conv2d_4 = conv2d_Q(q_bit = qbit, Kw = Kw[4], Ka = Ka[4])

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))
            )
            self.shortcut = nn.Sequential(
                Conv2d_3(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_4(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels)
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



class ShuffleUnit_swish(nn.Module):

    def __init__(self, qbit ,in_channels, out_channels, stride, Kw, Ka):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv2d_0 = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_1 = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
        Conv2d_2 = conv2d_Q(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])
        Conv2d_3 = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[3], Ka = Ka[3])
        Conv2d_4 = conv2d_Q(q_bit = qbit, Kw = Kw[4], Ka = Ka[4])

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))
            )
            self.shortcut = nn.Sequential(
                Conv2d_3(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_4(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels)
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

class ShuffleNetV2_swish(nn.Module):

    def __init__(self, qbit, ratio=1, class_num=100):
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

        #########  Ka, Kw  ######## swish
        ka = [2.7537312507629395, 8.352974891662598, 44.345428466796875, 7.256276607513428, 8.351006507873535, 4.622588634490967, 10.289244651794434, 1.9524964094161987, 2.988591432571411, 7.3726806640625, 2.07716965675354, 3.2872111797332764, 10.760262489318848, 3.0348165035247803, 3.212768316268921, 8.540712356567383, 6.730961322784424, 3.265659809112549, 8.540712356567383, 2.9971935749053955, 1.2801672220230103, 1.3836190700531006, 1.5459139347076416, 2.508695602416992, 2.5070836544036865, 3.474975824356079, 3.193751573562622, 2.9492719173431396, 2.3656506538391113, 3.6935536861419678, 3.697955846786499, 4.56043815612793, 2.868847370147705, 3.1386780738830566, 2.4623422622680664, 4.1813740730285645, 5.110937595367432, 2.3686935901641846, 4.969298362731934, 7.121790409088135, 3.9425771236419678, 7.146529197692871, 7.207535266876221, 4.599490642547607, 7.146529197692871, 6.649366855621338, 5.563633918762207, 6.241123676300049, 2.8004133701324463, 4.686285495758057, 3.690547466278076, 3.3939099311828613, 5.134063720703125, 3.796161651611328, 3.3674376010894775, 6.249766826629639]+ [1]*10 
        Ka = np.array(ka)/15.5
        # ka = [2.7537312507629395, 8.352974891662598/2, 44.345428466796875/15, 7.256276607513428, 8.351006507873535/2, 
        # 4.622588634490967, 10.289244651794434/2, 1.9524964094161987*3, 2.988591432571411, 7.3726806640625,
        # 2.07716965675354, 3.2872111797332764, 10.760262489318848/2, 3.0348165035247803, 3.212768316268921, 
        # 8.540712356567383, 6.730961322784424/2, 3.265659809112549, 8.540712356567383, 2.9971935749053955 * 2, 
        # 1.2801672220230103, 1.3836190700531006, 1.5459139347076416 * 2, 2.508695602416992, 2.5070836544036865, 
        # 3.474975824356079, 3.193751573562622, 2.9492719173431396, 2.3656506538391113, 3.6935536861419678, 
        # 3.697955846786499 / 3, 4.56043815612793, 2.868847370147705, 3.1386780738830566, 2.4623422622680664, 
        # 4.1813740730285645, 5.110937595367432, 2.3686935901641846, 4.969298362731934, 7.121790409088135, 
        # 3.9425771236419678, 7.146529197692871, 7.207535266876221, 4.599490642547607, 7.146529197692871, 
        # 6.649366855621338, 5.563633918762207, 6.241123676300049, 2.8004133701324463, 4.686285495758057, 
        # 3.690547466278076, 3.3939099311828613, 5.134063720703125, 3.796161651611328, 3.3674376010894775, 6.249766826629639]+ [1]*10 
        # Ka = np.array(ka)/15.5
        #Ka = np.ones_like(Ka)
        kw = [1.0240788459777832, 0.7284968495368958, 0.7170019149780273, 0.9115654826164246, 0.8345966935157776, 0.8403480648994446, 0.8074891567230225, 0.6712865233421326, 0.62143474817276, 0.8184850811958313, 0.7097687125205994, 0.6678714156150818, 0.6980628967285156, 0.6053593754768372, 0.5824851393699646, 0.6489204168319702, 0.43810752034187317, 0.7855631709098816, 0.3688235878944397, 0.7101003527641296, 0.40075504779815674, 0.5029330253601074, 0.5184293389320374, 0.49896061420440674, 0.5157762765884399, 0.4322760999202728, 0.6555637717247009, 0.3579387366771698, 0.8796291947364807, 0.6924411654472351, 0.4203939139842987, 0.5337778925895691, 0.6155830025672913, 0.5148077011108398, 0.49643006920814514, 0.605701744556427, 0.3512621819972992, 0.4653396010398865, 0.41797566413879395, 0.2847897410392761, 0.48266831040382385, 0.5118687748908997, 0.23804710805416107, 0.4624210596084595, 0.25393539667129517, 0.4072876274585724, 0.37856799364089966, 0.43110376596450806, 0.3780010938644409, 0.41204845905303955, 0.36244434118270874, 0.4905353784561157, 0.3262300193309784, 0.22004690766334534, 0.41724178194999695, 0.5333138108253479]+ [1]*10 
        Kw = np.array(kw)/15.5 
        #Kw = np.ones_like(Kw)


        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Conv2d_with_swish = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw, Ka = Ka)
        Conv_last = conv2d_Q_with_swish(q_bit = qbit, Kw = Kw[55], Ka = Ka[55])
        Linear = linear_Q_with_swish(q_bit=qbit, Kw = Kw[56], Ka = Ka[56])
       

        self.pre = nn.Sequential(
            Conv2d(3, 24, 3, padding=1, Kw = Kw[0], Ka = Ka[0]),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage_0(qbit, 24, out_channels[0], 3, 6, Kw, Ka)
        self.stage3 = self._make_stage(qbit, out_channels[0], out_channels[1], 7, 6, Kw[14:], Ka[14:])
        self.stage4 = self._make_stage(qbit, out_channels[1], out_channels[2], 3, 6, Kw[40:], Ka[40:])
        self.conv5 = nn.Sequential(
            Conv_last(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3])
        )

        self.fc = Linear(out_channels[3], class_num)

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

        """ Get the input, output and weight of each layer to calculate the scale, uncommoent when pre_reference is used
        """
        '''
        self.layer_inputs[0] = self.pre[0].input_q
        self.layer_weights[0] = self.pre[0].weight_q
        self.layer_inputs[1] = self.stage2[0]. residual[0].input_q
        self.layer_weights[1] = self.stage2[0].residual[0].weight_q
        self.layer_inputs[2] = self.stage2[0]. residual[2].input_q
        self.layer_weights[2] = self.stage2[0].residual[2].weight_q
        self.layer_inputs[3] = self.stage2[0]. residual[4].input_q
        self.layer_weights[3] = self.stage2[0].residual[4].weight_q
        self.layer_inputs[4] = self.stage2[0]. shortcut[0].input_q
        self.layer_weights[4] = self.stage2[0].shortcut[0].weight_q
        self.layer_inputs[5] = self.stage2[0]. shortcut[2].input_q
        self.layer_weights[5] = self.stage2[0].shortcut[2].weight_q
        self.layer_inputs[6] = self.stage2[1]. residual[0].input_q
        self.layer_weights[6] = self.stage2[1].residual[0].weight_q
        self.layer_inputs[7] = self.stage2[1]. residual[2].input_q
        self.layer_weights[7] = self.stage2[1].residual[2].weight_q
        self.layer_inputs[8] = self.stage2[1]. residual[4].input_q
        self.layer_weights[8] = self.stage2[1].residual[4].weight_q
        self.layer_inputs[9] = self.stage2[2]. residual[0].input_q
        self.layer_weights[9] = self.stage2[2].residual[0].weight_q
        self.layer_inputs[10] = self.stage2[2]. residual[2].input_q
        self.layer_weights[10] = self.stage2[2].residual[2].weight_q
        self.layer_inputs[11] = self.stage2[2]. residual[4].input_q
        self.layer_weights[11] = self.stage2[2].residual[4].weight_q
        self.layer_inputs[12] = self.stage2[3]. residual[0].input_q
        self.layer_weights[12] = self.stage2[3].residual[0].weight_q
        self.layer_inputs[13] = self.stage2[3]. residual[2].input_q
        self.layer_weights[13] = self.stage2[3].residual[2].weight_q
        self.layer_inputs[14] = self.stage2[3]. residual[4].input_q
        self.layer_weights[14] = self.stage2[3].residual[4].weight_q
        self.layer_inputs[15] = self.stage3[0]. residual[0].input_q
        self.layer_weights[15] = self.stage3[0].residual[0].weight_q
        self.layer_inputs[16] = self.stage3[0]. residual[2].input_q
        self.layer_weights[16] = self.stage3[0].residual[2].weight_q
        self.layer_inputs[17] = self.stage3[0]. residual[4].input_q
        self.layer_weights[17] = self.stage3[0].residual[4].weight_q
        self.layer_inputs[18] = self.stage3[0]. shortcut[0].input_q
        self.layer_weights[18] = self.stage3[0].shortcut[0].weight_q
        self.layer_inputs[19] = self.stage3[0]. shortcut[2].input_q
        self.layer_weights[19] = self.stage3[0].shortcut[2].weight_q
        self.layer_inputs[20] = self.stage3[1]. residual[0].input_q
        self.layer_weights[20] = self.stage3[1].residual[0].weight_q
        self.layer_inputs[21] = self.stage3[1]. residual[2].input_q
        self.layer_weights[21] = self.stage3[1].residual[2].weight_q
        self.layer_inputs[22] = self.stage3[1]. residual[4].input_q
        self.layer_weights[22] = self.stage3[1].residual[4].weight_q
        self.layer_inputs[23] = self.stage3[2]. residual[0].input_q
        self.layer_weights[23] = self.stage3[2].residual[0].weight_q
        self.layer_inputs[24] = self.stage3[2]. residual[2].input_q
        self.layer_weights[24] = self.stage3[2].residual[2].weight_q
        self.layer_inputs[25] = self.stage3[2]. residual[4].input_q
        self.layer_weights[25] = self.stage3[2].residual[4].weight_q
        self.layer_inputs[26] = self.stage3[3]. residual[0].input_q
        self.layer_weights[26] = self.stage3[3].residual[0].weight_q
        self.layer_inputs[27] = self.stage3[3]. residual[2].input_q
        self.layer_weights[27] = self.stage3[3].residual[2].weight_q
        self.layer_inputs[28] = self.stage3[3]. residual[4].input_q
        self.layer_weights[28] = self.stage3[3].residual[4].weight_q
        self.layer_inputs[29] = self.stage3[4]. residual[0].input_q
        self.layer_weights[29] = self.stage3[4].residual[0].weight_q
        self.layer_inputs[30] = self.stage3[4]. residual[2].input_q
        self.layer_weights[30] = self.stage3[4].residual[2].weight_q
        self.layer_inputs[31] = self.stage3[4]. residual[4].input_q
        self.layer_weights[31] = self.stage3[4].residual[4].weight_q
        self.layer_inputs[32] = self.stage3[5]. residual[0].input_q
        self.layer_weights[32] = self.stage3[5].residual[0].weight_q
        self.layer_inputs[33] = self.stage3[5]. residual[2].input_q
        self.layer_weights[33] = self.stage3[5].residual[2].weight_q
        self.layer_inputs[34] = self.stage3[5]. residual[4].input_q
        self.layer_weights[34] = self.stage3[5].residual[4].weight_q
        self.layer_inputs[35] = self.stage3[6]. residual[0].input_q
        self.layer_weights[35] = self.stage3[6].residual[0].weight_q
        self.layer_inputs[36] = self.stage3[6]. residual[2].input_q
        self.layer_weights[36] = self.stage3[6].residual[2].weight_q
        self.layer_inputs[37] = self.stage3[6]. residual[4].input_q
        self.layer_weights[37] = self.stage3[6].residual[4].weight_q
        self.layer_inputs[38] = self.stage3[7]. residual[0].input_q
        self.layer_weights[38] = self.stage3[7].residual[0].weight_q
        self.layer_inputs[39] = self.stage3[7]. residual[2].input_q
        self.layer_weights[39] = self.stage3[7].residual[2].weight_q
        self.layer_inputs[40] = self.stage3[7]. residual[4].input_q
        self.layer_weights[40] = self.stage3[7].residual[4].weight_q
        self.layer_inputs[41] = self.stage4[0]. residual[0].input_q
        self.layer_weights[41] = self.stage4[0].residual[0].weight_q
        self.layer_inputs[42] = self.stage4[0]. residual[2].input_q
        self.layer_weights[42] = self.stage4[0].residual[2].weight_q
        self.layer_inputs[43] = self.stage4[0]. residual[4].input_q
        self.layer_weights[43] = self.stage4[0].residual[4].weight_q
        self.layer_inputs[44] = self.stage4[0]. shortcut[0].input_q
        self.layer_weights[44] = self.stage4[0].shortcut[0].weight_q
        self.layer_inputs[45] = self.stage4[0]. shortcut[2].input_q
        self.layer_weights[45] = self.stage4[0].shortcut[2].weight_q
        self.layer_inputs[46] = self.stage4[1]. residual[0].input_q
        self.layer_weights[46] = self.stage4[1].residual[0].weight_q
        self.layer_inputs[47] = self.stage4[1]. residual[2].input_q
        self.layer_weights[47] = self.stage4[1].residual[2].weight_q
        self.layer_inputs[48] = self.stage4[1]. residual[4].input_q
        self.layer_weights[48] = self.stage4[1].residual[4].weight_q
        self.layer_inputs[49] = self.stage4[2]. residual[0].input_q
        self.layer_weights[49] = self.stage4[2].residual[0].weight_q
        self.layer_inputs[50] = self.stage4[2]. residual[2].input_q
        self.layer_weights[50] = self.stage4[2].residual[2].weight_q
        self.layer_inputs[51] = self.stage4[2]. residual[4].input_q
        self.layer_weights[51] = self.stage4[2].residual[4].weight_q
        self.layer_inputs[52] = self.stage4[3]. residual[0].input_q
        self.layer_weights[52] = self.stage4[3].residual[0].weight_q
        self.layer_inputs[53] = self.stage4[3]. residual[2].input_q
        self.layer_weights[53] = self.stage4[3].residual[2].weight_q
        self.layer_inputs[54] = self.stage4[3]. residual[4].input_q
        self.layer_weights[54] = self.stage4[3].residual[4].weight_q
        self.layer_inputs[55] = self.conv5[0].input_q
        self.layer_weights[55] = self.conv5[0].weight_q
        self.layer_outputs[55] = x
        '''
        return x

    def _make_stage_0(self, qbit, in_channels, out_channels, repeat, begin_num, Kw, Ka):
        layers = []
        layers.append(ShuffleUnit_swish_0(qbit, in_channels, out_channels, 2, Kw[1:6], Ka[1:6]))

        while repeat:
            layers.append(ShuffleUnit_swish(qbit, out_channels, out_channels, 1, Kw[begin_num: begin_num + 10], Ka[begin_num: begin_num + 10]))
            begin_num = begin_num + 3
            repeat -= 1
        return nn.Sequential(*layers)

    def _make_stage(self, qbit, in_channels, out_channels, repeat, begin_num, Kw, Ka):
        layers = []
        layers.append(ShuffleUnit_swish(qbit, in_channels, out_channels, 2, Kw[1:6], Ka[1:6]))

        while repeat:
            layers.append(ShuffleUnit_swish(qbit, out_channels, out_channels, 1, Kw[begin_num: begin_num + 10], Ka[begin_num: begin_num + 10]))
            begin_num = begin_num + 3
            repeat -= 1
        return nn.Sequential(*layers)
    
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
    model = ShuffleNetV2_swish(32).to(device)
    print(model)
    #torchsummary.summary(model, (3,32,32),device = "cuda")