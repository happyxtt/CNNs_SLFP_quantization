import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append('..')
from utils.sfp_quant import *
from utils.activation_func import *
from utils.conv2d_func import *

__all__ = ['SqueezeNet']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, qbit, Kw, Ka):
        super(Fire, self).__init__()

        Conv2d_1 = conv2d_Q_bias(q_bit = qbit, Kw = Kw[0], Ka = Ka[0])
        Conv2d_2 = conv2d_Q_bias(q_bit = qbit, Kw = Kw[1], Ka = Ka[1])
        Conv2d_3 = conv2d_Q_bias(q_bit = qbit, Kw = Kw[2], Ka = Ka[2])

        self.inplanes = inplanes
        self.squeeze = Conv2d_1(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = Conv2d_2(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = Conv2d_3(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, qbit, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()

        inout = [2.640000104904175, 28.194124221801758, 72.97775268554688, 72.97775268554688, 77.43336486816406, 120.30152893066406, 120.30152893066406, 135.91839599609375, 180.7229766845703, 180.7229766845703, 150.3900604248047, 442.6010437011719, 442.6010437011719, 482.10418701171875, 619.6080322265625, 619.6080322265625, 487.75390625, 919.07763671875, 919.07763671875, 597.53125, 786.5740966796875, 786.5740966796875, 632.507080078125, 973.8804931640625, 973.8804931640625, 715.134033203125]
        Ka = np.array(inout)/15.5
        weight=[0.791490912437439, 1.0884634256362915, 0.9738085865974426, 0.8482335209846497, 0.8622108101844788, 1.059234857559204, 0.5848156213760376, 1.0154176950454712, 0.7202360033988953, 0.8102350831031799, 2.0325794219970703, 0.6379887461662292, 0.877097487449646, 0.6971914172172546, 0.6247027516365051, 0.642976701259613, 0.735572338104248, 0.5566408634185791, 0.4962397813796997, 0.5997017025947571, 0.5008355379104614, 0.6644789576530457, 0.6134956479072571, 0.5012431144714355, 0.5272226333618164, 0.2842995524406433]
        Kw = np.array(weight)/15.5

        Conv2d_0 = conv2d_Q_bias(q_bit = qbit, Ka = Ka[0], Kw = Kw[0])
        Conv2d_final = conv2d_Q_bias(q_bit = qbit, Ka = Ka[25], Kw = Kw[25])
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                Conv2d_0(3, 96, kernel_size=7, stride=2),  #1
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64, qbit = qbit, Kw = Kw[1:], Ka = Ka[1:]),  
                Fire(128, 16, 64, 64, qbit = qbit, Kw = Kw[4:], Ka = Ka[4:]),
                Fire(128, 32, 128, 128, qbit = qbit, Kw = Kw[7:], Ka = Ka[7:]),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128, qbit = qbit, Kw = Kw[10:], Ka = Ka[10:]),
                Fire(256, 48, 192, 192, qbit = qbit, Kw = Kw[13:], Ka = Ka[13:]),
                Fire(384, 48, 192, 192, qbit = qbit, Kw = Kw[16:], Ka = Ka[16:]),
                Fire(384, 64, 256, 256, qbit = qbit, Kw = Kw[19:], Ka = Ka[19:]),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256, qbit = qbit, Kw = Kw[22:], Ka = Ka[22:]),
            )
        # Final convolution is initialized differently form the rest
        final_conv = Conv2d_final(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
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
        x = self.features(x)
        x = self.classifier(x)
        '''
        self.layer_inputs[0] = self.features[0].input_q
        self.layer_weights[0] = self.features[0].weight_q
        self.layer_inputs[1] = self.features[3].squeeze.input_q
        self.layer_weights[1] = self.features[3].squeeze.weight_q
        self.layer_inputs[2] = self.features[3].expand1x1.input_q
        self.layer_weights[2] = self.features[3].expand1x1.weight_q
        self.layer_inputs[3] = self.features[3].expand3x3.input_q
        self.layer_weights[3] = self.features[3].expand3x3.weight_q
        self.layer_inputs[4] = self.features[4].squeeze.input_q
        self.layer_weights[4] = self.features[4].squeeze.weight_q
        self.layer_inputs[5] = self.features[4].expand1x1.input_q
        self.layer_weights[5] = self.features[4].expand1x1.weight_q
        self.layer_inputs[6] = self.features[4].expand3x3.input_q
        self.layer_weights[6] = self.features[4].expand3x3.weight_q
        self.layer_inputs[7] = self.features[5].squeeze.input_q
        self.layer_weights[7] = self.features[5].squeeze.weight_q
        self.layer_inputs[8] = self.features[5].expand1x1.input_q
        self.layer_weights[8] = self.features[5].expand1x1.weight_q
        self.layer_inputs[9] = self.features[5].expand3x3.input_q
        self.layer_weights[9] = self.features[5].expand3x3.weight_q
        self.layer_inputs[10] = self.features[7].squeeze.input_q
        self.layer_weights[10] = self.features[7].squeeze.weight_q
        self.layer_inputs[11] = self.features[7].expand1x1.input_q
        self.layer_weights[11] = self.features[7].expand1x1.weight_q
        self.layer_inputs[12] = self.features[7].expand3x3.input_q
        self.layer_weights[12] = self.features[7].expand3x3.weight_q
        self.layer_inputs[13] = self.features[8].squeeze.input_q
        self.layer_weights[13] = self.features[8].squeeze.weight_q
        self.layer_inputs[14] = self.features[8].expand1x1.input_q
        self.layer_weights[14] = self.features[8].expand1x1.weight_q
        self.layer_inputs[15] = self.features[8].expand3x3.input_q
        self.layer_weights[15] = self.features[8].expand3x3.weight_q
        self.layer_inputs[16] = self.features[9].squeeze.input_q
        self.layer_weights[16] = self.features[9].squeeze.weight_q
        self.layer_inputs[17] = self.features[9].expand1x1.input_q
        self.layer_weights[17] = self.features[9].expand1x1.weight_q
        self.layer_inputs[18] = self.features[9].expand3x3.input_q
        self.layer_weights[18] = self.features[9].expand3x3.weight_q
        self.layer_inputs[19] = self.features[10].squeeze.input_q
        self.layer_weights[19] = self.features[10].squeeze.weight_q
        self.layer_inputs[20] = self.features[10].expand1x1.input_q
        self.layer_weights[20] = self.features[10].expand1x1.weight_q
        self.layer_inputs[21] = self.features[10].expand3x3.input_q
        self.layer_weights[21] = self.features[10].expand3x3.weight_q
        self.layer_inputs[22] = self.features[12].squeeze.input_q
        self.layer_weights[22] = self.features[12].squeeze.weight_q
        self.layer_inputs[23] = self.features[12].expand1x1.input_q
        self.layer_weights[23] = self.features[12].expand1x1.weight_q
        self.layer_inputs[24] = self.features[12].expand3x3.input_q
        self.layer_weights[24] = self.features[12].expand3x3.weight_q
        self.layer_inputs[25] = self.classifier[1].input_q
        self.layer_weights[25] = self.classifier[1].weight_q
        '''
        return x.view(x.size(0), self.num_classes)




if __name__=='__main__':
    # model check
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SqueezeNet(qbit = 32).to(device)
    #model = MobileNetV1(ch_in=3, n_classes=10).to(device)
    #print(model)
    #summary(model, input_size=(3, 224, 224), device='cuda')

    #model = torch.load("../ckpt/resnet-50.pth", map_location='cpu')  # 注意指定map_location以确保在CPU上加载
    # 遍历模型的每一层并打印详细信息
    #for key in model:
    #    print(key)

    print(model)