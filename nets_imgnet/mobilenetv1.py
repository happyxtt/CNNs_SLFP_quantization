import torch.nn as nn
from torchsummary import summary
import torch
import sys
sys.path.append('..')
from utils.sfp_quant import *
from utils.activation_func import *
from utils.conv2d_func import *


class MobileNetV1_Q(nn.Module):
    def __init__(self, ch_in, qbit):
        super(MobileNetV1_Q, self).__init__()
        
        ka = [2.640000104904175, 2.6023073196411133, 6.629735469818115, 8.06674575805664, 13.16812801361084, 3.5005202293395996, 5.474634170532227, 3.467971086502075, 2.6531498432159424, 2.276766061782837, 4.367635250091553, 2.7206525802612305, 5.651697635650635, 2.0327985286712646, 2.945751905441284, 1.9591712951660156, 3.280294418334961, 1.7093303203582764, 3.6466710567474365, 1.7202441692352295, 6.958395004272461, 2.871131658554077, 12.649026870727539, 2.9706058502197266, 6.570033073425293, 2.18925142288208, 2.2897119522094727, 9.842595100402832]
        Ka = np.array(ka)/15.5

        kw = [0.8589458465576172, 1.9635683298110962, 1.2365360260009766, 0.5438900589942932, 1.2541989088058472, 0.9803679585456848, 1.236991286277771, 0.28881916403770447, 1.0297598838806152, 0.8297733068466187, 0.7577158212661743, 0.23283751308918, 0.5724515318870544, 0.6798462867736816, 0.5818396806716919, 0.8004610538482666, 0.5788508057594299, 0.589914083480835, 0.721644401550293, 0.7284839749336243, 0.8877108097076416, 0.4677695035934448, 0.8636178374290466, 0.306072473526001, 0.8925297856330872, 0.21044661104679108, 0.6525866389274597, 0.918532133102417]
        Kw = np.array(kw)/15.5

        Conv2d = conv2d_Q(q_bit = qbit, Kw = Kw, Ka = Ka)
        Linear = linear_Q(q_bit=qbit, Kw = Kw[27], Ka = Ka[27])

        def conv_dw(inp, oup, stride, Kw, Ka):
            return nn.Sequential(
                # dw
                Conv2d(inp, inp, 3, Kw[0], Ka[0], stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pw
                Conv2d(inp, oup, 1, Kw[1], Ka[1], 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        def conv_bn(inp, oup, stride, Kw, Ka):
                return nn.Sequential(
                    Conv2d(inp, oup, 3, Kw, Ka, stride, 1,  bias=False), 
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                    )
        
        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2, Kw[0], Ka[0]),
            conv_dw(32, 64, 1, Kw[1:], Ka[1:]),
            conv_dw(64, 128, 2, Kw[3:], Ka[3:]),
            conv_dw(128, 128, 1, Kw[5:], Ka[5:]),
            conv_dw(128, 256, 2, Kw[7:], Ka[7:]),
            conv_dw(256, 256, 1, Kw[9:], Ka[9:]),
            conv_dw(256, 512, 2, Kw[11:], Ka[11:]),
            conv_dw(512, 512, 1, Kw[13:], Ka[13:]),
            conv_dw(512, 512, 1, Kw[15:], Ka[15:]),
            conv_dw(512, 512, 1, Kw[17:], Ka[17:]),
            conv_dw(512, 512, 1, Kw[19:], Ka[19:]),
            conv_dw(512, 512, 1, Kw[21:], Ka[21:]),
            conv_dw(512, 1024, 2, Kw[23:], Ka[23:]),
            conv_dw(1024, 1024, 1, Kw[25:], Ka[25:]),
            nn.AvgPool2d(7)
        )
        #self.fc = Linear(1024, 1000)
        self.fc = nn.Linear(1024, 1000)

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
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        '''
        self.layer_inputs[0] = self.model[0][0].input_q
        self.layer_weights[0] = self.model[0][0].weight_q

        self.layer_inputs[1] = self.model[1][0].input_q
        self.layer_weights[1] = self.model[1][0].weight_q

        self.layer_inputs[2] = self.model[1][3].input_q
        self.layer_weights[2] = self.model[1][3].weight_q

        self.layer_inputs[3] = self.model[2][0].input_q
        self.layer_weights[3] = self.model[2][0].weight_q

        self.layer_inputs[4] = self.model[2][3].input_q
        self.layer_weights[4] = self.model[2][3].weight_q

        self.layer_inputs[5] = self.model[3][0].input_q
        self.layer_weights[5] = self.model[3][0].weight_q

        self.layer_inputs[6] = self.model[3][3].input_q
        self.layer_weights[6] = self.model[3][3].weight_q

        self.layer_inputs[7] = self.model[4][0].input_q
        self.layer_weights[7] = self.model[4][0].weight_q

        self.layer_inputs[8] = self.model[4][3].input_q
        self.layer_weights[8] = self.model[4][3].weight_q

        self.layer_inputs[9] = self.model[5][0].input_q
        self.layer_weights[9] = self.model[5][0].weight_q

        self.layer_inputs[10] = self.model[5][3].input_q
        self.layer_weights[10] = self.model[5][3].weight_q

        self.layer_inputs[11] = self.model[6][0].input_q
        self.layer_weights[11] = self.model[6][0].weight_q

        self.layer_inputs[12] = self.model[6][3].input_q
        self.layer_weights[12] = self.model[6][3].weight_q
        
        self.layer_inputs[13] = self.model[7][0].input_q
        self.layer_weights[13] = self.model[7][0].weight_q

        self.layer_inputs[14] = self.model[7][3].input_q
        self.layer_weights[14] = self.model[7][3].weight_q

        self.layer_inputs[15] = self.model[8][0].input_q
        self.layer_weights[15] = self.model[8][0].weight_q

        self.layer_inputs[16] = self.model[8][3].input_q
        self.layer_weights[16] = self.model[8][3].weight_q

        self.layer_inputs[17] = self.model[9][0].input_q
        self.layer_weights[17] = self.model[9][0].weight_q

        self.layer_inputs[18] = self.model[9][3].input_q
        self.layer_weights[18] = self.model[9][3].weight_q

        self.layer_inputs[19] = self.model[10][0].input_q
        self.layer_weights[19] = self.model[10][0].weight_q

        self.layer_inputs[20] = self.model[10][3].input_q
        self.layer_weights[20] = self.model[10][3].weight_q

        self.layer_inputs[21] = self.model[11][0].input_q
        self.layer_weights[21] = self.model[11][0].weight_q

        self.layer_inputs[22] = self.model[11][3].input_q
        self.layer_weights[22] = self.model[11][3].weight_q

        self.layer_inputs[23] = self.model[12][0].input_q
        self.layer_weights[23] = self.model[12][0].weight_q

        self.layer_inputs[24] = self.model[12][3].input_q
        self.layer_weights[24] = self.model[12][3].weight_q

        self.layer_inputs[25] = self.model[13][0].input_q
        self.layer_weights[25] = self.model[13][0].weight_q

        self.layer_inputs[26] = self.model[13][3].input_q
        self.layer_weights[26] = self.model[13][3].weight_q

        self.layer_inputs[27] = self.fc.input_q
        self.layer_weights[27] = self.fc.weight_q
        self.layer_outputs[27] = x
        '''
        return x

if __name__=='__main__':
    # model check
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1_Q(ch_in=3, qbit=32, abit=32).to(device)
    #model = MobileNetV1(ch_in=3, n_classes=10).to(device)
    #print(model)
    #summary(model, input_size=(3, 224, 224), device='cuda')
