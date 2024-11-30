import os
import sys
import csv
import pandas as pd
from matplotlib import pyplot as plt

# 绘制雷达图
def draw(data, labels, save_path, mode, device):
    num_vars = len(data[0])
    angles = [n / float(num_vars) * 2 * 3.141592653 for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    ax.set_theta_offset(3.141592653)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels)
    
    ax.set_rscale('log')
    
    for i in range(len(data)):
        values = data[i]
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # 设置标题
    if mode == "infer":
        plt.title(device + " Inference" + " Radar Chart")
    elif mode == "train":
        plt.title(device + " Training" + " Radar Chart")    
    
    plt.savefig(save_path)
    plt.close()

def generate_radar_chart(csv_file_path, save_path, mode):
    pd_data = pd.read_csv(csv_file_path)
    # 读取第一行返回为列表
    print("--------------{mode}----------------".format(mode=mode))
    devices = pd_data.columns.values.tolist()[1:]
    print(devices)
    # 读取第一列返回为列表
    models = pd_data.iloc[:, 0].values.tolist()
    print(models)
    # 调用draw函数绘制雷达图
    for device in devices:
        data = []
        path = save_path + device + ".png"
        # 将该设备对应列的数据加入data
        for model in models:
            data.append(pd_data.loc[pd_data['Model'] == model, device].values.tolist())
        draw([data], models, path, mode, device)
        
if __name__ == '__main__':
    infer_input_path = "./../Table/sum/infer_weighted_sum.csv"
    train_input_path = "./../Table/sum/train_weighted_sum.csv"
    
    generate_radar_chart(infer_input_path,"./../Diagram/Weighted/old_radar_chart/infer_radar_chart/", "infer")
    generate_radar_chart(train_input_path,"./../Diagram/Weighted/old_radar_chart/train_radar_chart/", "train")
    
    
    