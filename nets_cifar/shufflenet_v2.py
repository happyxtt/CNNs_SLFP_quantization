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
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU(),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU()
            )
            self.shortcut = nn.Sequential(
                Conv2d_3(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_4(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU()
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                Conv2d_0(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                layerout_quantize_func(q_bit=qbit),
                nn.ReLU(),
                Conv2d_1(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                Conv2d_2(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                layerout_quantize_func(q_bit=qbit),
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

        #########  Ka, Kw  ######## swish
        # ka = [2.7085108757019043, 5.417021751403809, 4.177095413208008, 9.934863090515137, 5.417021751403809, 10.834044456481934, 7.336031913757324, 6.1688432693481445, 8.724061965942383, 5.187358379364014, 3.5125045776367188, 5.656854152679443, 6.44196081161499, 6.727171421051025, 6.1688432693481445, 8.0, 4.0, 6.1688432693481445, 8.0, 9.110308647155762, 4.0, 2.8284270763397217, 5.187358379364014, 4.362030982971191, 3.3635857105255127, 5.907304763793945, 4.0, 3.8304128646850586, 6.1688432693481445, 4.362030982971191, 2.9536523818969727, 6.1688432693481445, 3.668015956878662, 3.0844216346740723, 5.907304763793945, 4.555154323577881, 3.5125045776367188, 7.336031913757324, 4.756828784942627, 3.5125045776367188, 7.660826206207275, 5.907304763793945, 3.0844216346740723, 11.313708305358887, 5.907304763793945, 10.374716758728027, 4.555154323577881, 4.756828784942627, 9.513656616210938, 5.656854152679443, 3.668015956878662, 14.672063827514648, 7.025008678436279, 4.362030982971191, 15.32165241241455]+ [1]*10 
        # Ka = np.array(ka)/15.5
        # Ka = np.ones_like(Ka)
        # kw = [1.2968395948410034, 0.9170039296150208, 1.241857886314392, 1.0905078649520874, 1.0, 1.4768261909484863, 0.7384130358695984, 0.9170039296150208, 0.8052451014518738, 0.5, 0.9170039296150208, 0.8781260251998901, 0.9170039296150208, 0.6771277189254761, 1.0, 0.4585019648075104, 0.33856385946273804, 0.5946035385131836, 0.4788016676902771, 0.7711053490638733, 0.28469714522361755, 0.6771277189254761, 0.5, 0.3535533547401428, 0.5, 0.5693943500518799, 0.4585019648075104, 0.522136926651001, 0.7071067094802856, 0.5, 0.4788016676902771, 0.5, 0.38555267453193665, 0.4788016676902771, 0.4788016676902771, 0.6209288835525513, 0.4026225805282593, 0.4585019648075104, 0.6209288835525513, 0.3692065477371216, 0.4204481840133667, 0.4204481840133667, 0.27262693643569946, 0.3692065477371216, 0.4788016676902771, 0.5452538132667542, 0.28469714522361755, 0.4026225805282593, 0.4026225805282593, 0.33856385946273804, 0.3692065477371216, 0.3692065477371216, 0.4204481840133667, 0.2973017692565918, 0.3535533547401428] + [1]*10 
        # Kw = np.array(kw)/15.5   
        # Kw = np.ones_like(Kw)  
     
        ## relu
        ka = [2.7537312507629395, 5.234059810638428, 4.2632646560668945, 9.554951667785645, 5.234059810638428, 11.346541404724121, 7.264077186584473, 6.350336074829102, 9.16678237915039, 5.25019645690918, 3.478346347808838, 5.273358345031738, 6.550426959991455, 6.841591835021973, 6.551841735839844, 8.429676055908203, 3.6293373107910156, 6.033638000488281, 8.429676055908203, 8.35006332397461, 3.1058642864227295, 2.4753012657165527, 4.7964186668396, 3.1863455772399902, 2.5042595863342285, 4.836341857910156, 3.550802707672119, 2.6628284454345703, 4.511188983917236, 3.9728715419769287, 2.248741626739502, 4.604737758636475, 2.9986534118652344, 2.4434714317321777, 5.667801856994629, 3.9212722778320312, 2.7176830768585205, 5.555120944976807, 3.8605222702026367, 2.682180404663086, 5.483858585357666, 5.311583518981934, 2.37758731842041, 7.243487358093262, 5.311583518981934, 6.235666275024414, 2.933143377304077, 2.8617892265319824, 6.526449680328369, 3.827115774154663, 2.4493494033813477, 6.2418341636657715, 3.028607130050659, 2.5422754287719727, 7.466217994689941, 3.8511102199554443]+ [1]*10 
        Ka = np.array(ka)/15
        # Ka = np.ones_like(Ka)
        kw = [1.3029288053512573, 0.8977667689323425, 1.2451014518737793, 1.0910767316818237, 1.0148509740829468, 1.4486286640167236, 0.7227526307106018, 0.9099447131156921, 0.8158362507820129, 0.5097854733467102, 0.9196422696113586, 0.871035635471344, 0.9180927276611328, 0.6798731684684753, 0.9902427792549133, 0.4500037431716919, 0.336839497089386, 0.6030286550521851, 0.47783952951431274, 0.7545949220657349, 0.2900848984718323, 0.6755377054214478, 0.49563902616500854, 0.3502323627471924, 0.4965823292732239, 0.5626934766769409, 0.4495861530303955, 0.5228708386421204, 0.706261932849884, 0.4907167851924896, 0.48858827352523804, 0.49457845091819763, 0.3859078288078308, 0.4841776490211487, 0.4749361276626587, 0.625221848487854, 0.39950987696647644, 0.45098355412483215, 0.6184139847755432, 0.3726619482040405, 0.42748335003852844, 0.4202514886856079, 0.27142152190208435, 0.36484283208847046, 0.47397205233573914, 0.5499554872512817, 0.2817867696285248, 0.39557644724845886, 0.4021666347980499, 0.3414369523525238, 0.37255141139030457, 0.36935344338417053, 0.41747722029685974, 0.29923439025878906, 0.34784701466560364, 0.2244890332221985] + [1]*10 
        Kw = np.array(kw)/15
        ##############

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
            layerout_quantize_func(q_bit=qbit),
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
        return x

    def _make_stage(self, qbit, in_channels, out_channels, repeat, begin_num, Kw, Ka):
        layers = []
        layers.append(ShuffleUnit(qbit, in_channels, out_channels, 2, Kw[1:6], Ka[1:6]))

        while repeat:
            layers.append(ShuffleUnit(qbit, out_channels, out_channels, 1, Kw[begin_num: begin_num + 10], Ka[begin_num: begin_num + 10]))
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
    model = ShuffleNetV2(32).to(device)
    print(model)
    torchsummary.summary(model, (3,32,32),device = "cuda")