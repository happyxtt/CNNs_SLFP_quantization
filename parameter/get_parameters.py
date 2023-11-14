#state_dict = torch.load('ckpt/cifar100/mobilenet_s4.t7')

import torch
import numpy as np

# 加载.t7文件
model = torch.load('ckpt/cifar100/mobilenet_fp32_scale_SLFP33_finetune.t7', map_location=torch.device('cpu')) 

# 初始化一个字典用于保存参数信息
parameter_info = {}

# 初始化一个列表用于保存参数数据
parameters = []

# 提取权重、偏置和BN层参数数据
for key, value in model.items():
    if 'weight' in key or 'bias' in key:
        parameter_info[key] = value.size()  # 存储参数形状
        parameters.append(value.numpy().flatten())  # 存储参数数据

# 将参数堆叠成一个大数组
parameters_array = np.concatenate(parameters)

# 保存参数数据为一个txt文件
np.savetxt('all_parameters.txt', parameters_array)

# 保存参数信息字典为一个npy文件
np.save('parameter_info.npy', parameter_info)


