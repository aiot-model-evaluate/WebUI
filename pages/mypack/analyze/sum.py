import pandas as pd
import os

weights = {
    'infer_average_inference_delay.csv': 0.285,
    'infer_throughput.csv': 0.285,
    'infer_power.csv': 0.15,
    'infer_energy_efficiency.csv': 0.28,
    
    'train_throughput.csv': 0.384,
    'train_power.csv': 0.258,
    'train_energy_efficiency.csv': 0.358,
}

def weighted_sum(directory, prefix, weights):
    sum_df = None
    sum_weight = None
    for filename, weight in weights.items():
        if filename.startswith(prefix):
            file_path = os.path.join(directory, 'norm', 'norm_' + filename)
            df = pd.read_csv(file_path, index_col=0)
            if sum_df is None:
                sum_df = df * weight
                sum_weight = weight
            else:
                sum_df += df * weight
                sum_weight += weight
    return sum_df / sum_weight

directory_path = '.\..\Table'

infer_sum = weighted_sum(directory_path, 'infer_', weights)
train_sum = weighted_sum(directory_path, 'train_', weights)

path1 = '.\..\Table\sum'
if not os.path.exists(path1):
    os.makedirs(path1)

infer_sum.to_csv(os.path.join(path1, 'infer_weighted_sum.csv'))
train_sum.to_csv(os.path.join(path1, 'train_weighted_sum.csv'))

# 在第一行第一列写入"Model"
with open(os.path.join(path1, 'infer_weighted_sum.csv'), 'r+') as f:
    # 按行读取文件
    lines = f.readlines()
    # 在第一行第一列写入"Model"
    temp = "Model" + lines[0]
    lines[0] = temp
    # 清空文件
    f.seek(0)
    f.truncate()
    # 写入新的内容
    f.writelines(lines)
    
# 在第一行第一列写入"Model"
with open(os.path.join(path1, 'train_weighted_sum.csv'), 'r+') as f:
    # 按行读取文件
    lines = f.readlines()
    # 在第一行第一列写入"Model"
    temp = "Model" + lines[0]
    lines[0] = temp
    # 清空文件
    f.seek(0)
    f.truncate()
    # 写入新的内容
    f.writelines(lines)

print("Inference Weighted Sum:\n", infer_sum)
print("Training Weighted Sum:\n", train_sum)
