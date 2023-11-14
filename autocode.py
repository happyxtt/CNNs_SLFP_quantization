import numpy as np

def convkeys_squeezenet():
    total_num =  1
    for fire_num in [3,4,5,7,8,9,10,12]:
        #for sub_num in [0, 3]:
            print("        self.layer_inputs[" + str(total_num) + "] = self.features[" + str(fire_num) + "].squeeze.input_q")
            print("        self.layer_weights[" + str(total_num) + "] = self.features[" + str(fire_num) + "].squeeze.weight_q")
            total_num += 1
            print("        self.layer_inputs[" + str(total_num) + "] = self.features[" + str(fire_num) + "].expand1x1.input_q")
            print("        self.layer_weights[" + str(total_num) + "] = self.features[" + str(fire_num) + "].expand1x1.weight_q")
            total_num += 1
            print("        self.layer_inputs[" + str(total_num) + "] = self.features[" + str(fire_num) + "].expand3x3.input_q")          
            print("        self.layer_weights[" + str(total_num) + "] = self.features[" + str(fire_num) + "].expand3x3.weight_q")
            total_num += 1

def convkeys_resnet50():
    total_num = 43
    layer_num = 4
    for sequential_num in [0,1,2]:
        if (sequential_num == 0):
            print("        self.inputs[" + str(total_num) + "] = self.layer" + str(layer_num) + "[" + str(sequential_num) + "].downsample[0].input_q")
            print("        self.weights[" + str(total_num) + "] = self.layer" + str(layer_num) + "[" + str(sequential_num) + "].downsample[0].weight_q")
            total_num += 1
        for conv_num in [1,2,3]:
            print("        self.inputs[" + str(total_num) + "] = self.layer" + str(layer_num) + "[" + str(sequential_num) + "].conv" + str(conv_num) + ".input_q")
            print("        self.weights[" + str(total_num) + "] = self.layer" + str(layer_num) + "[" + str(sequential_num) + "].conv" + str(conv_num) + ".weight_q")
            total_num += 1

          

def K_mobilenet_txt():
    total_num =  1
    for sequential_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        for sub_num in [0, 3]:
            print("        self.layer_inputs[" + str(total_num) + "] = self.model[" + str(sequential_num) + "][" + str(sub_num) + "].input_q")
            print("        self.layer_weights[" + str(total_num) + "] = self.model[" + str(sequential_num) + "][" + str(sub_num) + "].weight_q")
            total_num += 1

def extract_num():
    file  = open('max_weight_resnet.txt', 'r')
    lines = file.readlines()
    # 初始化一个空的数组来保存数字
    numbers = []
    # 遍历每行内容并提取数字
    for line in lines:
        # 使用split()方法将每行的文本按空格分割成单词
        words = line.split()

        # 遍历每个单词，将其尝试转换为浮点数，并添加到数组中
        for word in words:
            try:
                number = float(word)
                numbers.append(number)
            except ValueError:
                pass
    numbers_without_layer = []
    for i in range(54):
        numbers_without_layer.append(numbers[2*i+1]) 
    # 打印提取出的数字数组
    print(numbers_without_layer)

def backup_mobilenet():
    inout = [2.640000104904175, 2.6023073196411133, 6.629735469818115, 8.06674575805664, 13.16812801361084, 3.5005202293395996, 5.474634170532227, 3.467971086502075, 2.6531498432159424, 2.276766061782837, 4.367635250091553, 2.7206525802612305, 5.651697635650635, 2.0327985286712646, 2.945751905441284, 1.9591712951660156, 3.280294418334961, 1.7093303203582764, 3.6466710567474365, 1.7202441692352295, 6.958395004272461, 2.871131658554077, 12.649026870727539, 2.9706058502197266, 6.570033073425293, 2.18925142288208, 2.2897119522094727, 9.842595100402832]
    weight = [0.8589458465576172, 1.9635683298110962, 1.2365360260009766, 0.5438900589942932, 1.2541989088058472, 0.9803679585456848, 1.236991286277771, 0.28881916403770447, 1.0297598838806152, 0.8297733068466187, 0.7577158212661743, 0.23283751308918, 0.5724515318870544, 0.6798462867736816, 0.5818396806716919, 0.8004610538482666, 0.5788508057594299, 0.589914083480835, 0.721644401550293, 0.7284839749336243, 0.8877108097076416, 0.4677695035934448, 0.8636178374290466, 0.306072473526001, 0.8925297856330872, 0.21044661104679108, 0.6525866389274597, 0.918532133102417]
    weight = np.array(weight)/15.5
    print(weight)

def backup_squeezenet():
    inout = [2.640000104904175, 28.194124221801758, 72.97775268554688, 72.97775268554688, 77.43336486816406, 120.30152893066406, 120.30152893066406, 135.91839599609375, 180.7229766845703, 180.7229766845703, 150.3900604248047, 442.6010437011719, 442.6010437011719, 482.10418701171875, 619.6080322265625, 619.6080322265625, 487.75390625, 919.07763671875, 919.07763671875, 597.53125, 786.5740966796875, 786.5740966796875, 632.507080078125, 973.8804931640625, 973.8804931640625, 715.134033203125]
    Ka = np.array(inout)/15.5
    weight=[0.791490912437439, 1.0884634256362915, 0.9738085865974426, 0.8482335209846497, 0.8622108101844788, 1.059234857559204, 0.5848156213760376, 1.0154176950454712, 0.7202360033988953, 0.8102350831031799, 2.0325794219970703, 0.6379887461662292, 0.877097487449646, 0.6971914172172546, 0.6247027516365051, 0.642976701259613, 0.735572338104248, 0.5566408634185791, 0.4962397813796997, 0.5997017025947571, 0.5008355379104614, 0.6644789576530457, 0.6134956479072571, 0.5012431144714355, 0.5272226333618164, 0.2842995524406433]
    Kw = np.array(weight)/15.5
    print("Ka = ", Ka)
    print("Kw = ", Kw)

def backup_resnet():
    inout = [2.640000104904175, 6.8493123054504395, 6.8493123054504395, 2.0926101207733154, 2.3465774059295654, 4.545978546142578, 2.3446199893951416, 2.8475520610809326, 4.749269485473633, 2.107717990875244, 3.3412084579467773, 4.412791728973389, 4.412791728973389, 3.8282792568206787, 2.9802281856536865, 5.069820404052734, 1.4619481563568115, 2.186246395111084, 5.0605292320251465, 2.0890896320343018, 2.204008102416992, 5.053404808044434, 2.407410144805908, 3.188458204269409, 4.624925136566162, 4.624925136566162, 3.9921064376831055, 2.503716230392456, 3.886512041091919, 3.0490880012512207, 1.9895399808883667, 4.729367256164551, 1.8484134674072266, 1.7739477157592773, 4.359723091125488, 2.481842279434204, 2.022366762161255, 5.081398963928223, 3.197451591491699, 1.9158319234848022, 5.182647705078125, 2.850689172744751, 3.7739882469177246, 4.207239627838135, 4.207239627838135, 2.8491551876068115, 3.0215585231781006, 15.216011047363281, 3.1868929862976074, 1.979512095451355, 16.78635597229004, 2.9933321475982666, 2.6009302139282227, 7.67310094833374]
    Ka = np.array(inout)/15.5
    weight = [0.7817208766937256, 0.987881064414978, 0.7266281247138977, 0.46786433458328247, 0.3936349153518677, 0.2617597281932831, 0.5201045870780945, 0.29462704062461853, 0.19206704199314117, 0.2855665683746338, 0.2751551568508148, 0.5662445425987244, 0.3531537353992462, 0.29927510023117065, 0.3916732370853424, 0.25216183066368103, 0.2997848093509674, 0.30379050970077515, 0.23830968141555786, 0.2555960714817047, 0.35215842723846436, 0.28143224120140076, 0.2209654152393341, 0.2956201732158661, 0.34601572155952454, 0.3425379693508148, 0.2007666528224945, 0.32124170660972595, 0.29417240619659424, 0.2634257674217224, 0.4968879222869873, 0.2714691460132599, 0.21002456545829773, 0.3537616431713104, 0.2390037477016449, 0.27921295166015625, 0.3126426041126251, 0.2721982002258301, 0.19188867509365082, 0.316133052110672, 0.39949774742126465, 0.2235630750656128, 0.32883593440055847, 0.6412832736968994, 0.3415152430534363, 0.3992723524570465, 0.3546474874019623, 0.700333833694458, 0.22574764490127563, 0.24268335103988647, 0.4540838599205017, 0.14155906438827515, 0.279774934053421, 0.7371371984481812]
    Kw = np.array(weight)/15.5
    print("Ka = ", Ka)
    print("Kw = ", Kw)


if __name__ == '__main__':  
    backup_resnet()