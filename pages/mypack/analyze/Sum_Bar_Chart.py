import os
import sys
import csv
import pandas as pd
from matplotlib import pyplot as plt

train_weight = {
    "Bert": 0.2,
    "LSTM": 0.2,
    "Resnet": 0.2,
    "Unet": 0.2,
    "yolo_v10":0.2
}

infer_weight = {
    "Bert": 0.1,
    "GLM4": 0.1,
    "LDM": 0.1,
    "llama3": 0.1,
    "LSTM": 0.1,
    "Qwen2": 0.1,
    "Resnet": 0.1,
    "Unet": 0.1,
    "yolo_v10": 0.1
}
    
def generate_Bar_chart(csv_file_path, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    pd_data = pd.read_csv(csv_file_path)
    # 读取第一行返回为列表
    print("--------------{mode}----------------".format(mode=mode))
    devices = pd_data.columns.values.tolist()[1:]
    print(devices)
    # 读取第一列返回为列表
    models = pd_data.iloc[:, 0].values.tolist()
    print(models)
    if mode == "train":
        score = {}
        weight = train_weight
        # 读取数据(一列列的读取)
        volumes = pd_data.iloc[:, 1:].values
        for i in range(len(devices)):
            score[devices[i]] = sum(volumes[:, i] * [weight[model] for model in models])
        # 画图(设备为x轴，分数为y轴)
        plt.figure(figsize=(10, 5))
        plt.bar(score.keys(), score.values(), color='green', width=0.5)
        plt.title("Train Score")
        plt.xlabel("Devices")
        plt.ylabel("Score")
        plt.savefig(os.path.join(save_path, "train_score.png"))
        plt.close()
    
    elif mode == "infer":
        score = {}
        weight = infer_weight
        # 读取数据(一列列的读取)
        volumes = pd_data.iloc[:, 1:].values
        for i in range(len(devices)):
            score[devices[i]] = sum(volumes[:, i] * [weight[model] for model in models])
        # 画图(设备为x轴，分数为y轴)
        plt.figure(figsize=(10, 5))
        plt.bar(score.keys(), score.values(), color='green', width=0.5)
        plt.title("Infer Score")
        plt.xlabel("Devices")
        plt.ylabel("Score")
        plt.savefig(os.path.join(save_path, "infer_score.png"))
        plt.close()
        
if __name__ == '__main__':
    infer_input_path = "./../Table/sum/infer_weighted_sum.csv"
    train_input_path = "./../Table/sum/train_weighted_sum.csv"
    
    generate_Bar_chart(infer_input_path,"./../Diagram/weighted/Bar_chart/infer_Bar_chart/", "infer")
    generate_Bar_chart(train_input_path,"./../Diagram/weighted/Bar_chart/train_Bar_chart/", "train")
