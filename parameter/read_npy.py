import numpy as np

# 加载参数信息字典
parameter_info = np.load('parameter_info.npy', allow_pickle=True).item()

# 打印参数信息
for key, value in parameter_info.items():
    print(f"Parameter: {key}, Shape: {value}")