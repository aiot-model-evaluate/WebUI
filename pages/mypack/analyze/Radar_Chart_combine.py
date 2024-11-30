import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# 绘制雷达图
def draw(data, labels, save_path, mode):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 完成图形的闭合
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # 偏移使图形从正上方开始
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels)
    
    ax.set_rscale('log')  # 设置对数刻度
    
    # 绘制每个设备的数据
    for device_data, device_name in data:
        values = device_data + device_data[:1]  # 数据闭合
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=device_name)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.title(f"{mode.capitalize()} Radar Chart")
    plt.savefig(save_path)
    plt.close()

def generate_radar_chart(csv_file_path, save_path, mode):
    pd_data = pd.read_csv(csv_file_path)
    devices = pd_data.columns[1:].tolist()
    models = pd_data['Model'].tolist()
    
    all_device_data = []
    
    for device in devices:
        device_data = []
        for model in models:
            model_values = pd_data.loc[pd_data['Model'] == model, device].dropna()
            if not model_values.empty:
                device_data.append(model_values.iloc[0])
            else:
                device_data.append(0)
        
        all_device_data.append((device_data, device))

    path = os.path.join(save_path, f"{mode}_radar_chart.png")
    draw(all_device_data, models, path, mode)

if __name__ == '__main__':
    infer_input_path = "./../Table/sum/infer_weighted_sum.csv"
    train_input_path = "./../Table/sum/train_weighted_sum.csv"
    
    generate_radar_chart(infer_input_path,"./../Diagram/weighted/conbined_radar_chart/infer_conbined_radar_chart/", "infer")
    generate_radar_chart(train_input_path,"./../Diagram/weighted/conbined_radar_chart/train_conbined_radar_chart/", "train")
