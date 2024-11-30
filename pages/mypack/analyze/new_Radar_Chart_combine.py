import os
import sys
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

class Radar_Chart:
    # 初始化类(需要指定)
    # 使用is_sumed参数来指定是否按照模型进行加权求和以此生成总结图(is_sumed=True则生成总结图，is_sumed=False则生成单个模型图)
    def __init__(self, csv_file_path, save_path, is_sumed=False):
        # 以进行了训练的模型为key，对应的数据为pandas的DataFrame(横坐标为设备，纵坐标为)
        self.train_data_list = {}
        # 以进行了训练的模型为key，对应的数据为pandas的DataFrame(横坐标为设备，纵坐标为)
        self.infer_data_list = {}
        self.csv_file_path = csv_file_path
        self.save_path = save_path
        self.is_sumed = is_sumed
        self.csv_file_list = self.load_files(self.csv_file_path)
        self.train_models = []
        self.infer_models = []
        self.train_devices = []
        self.infer_devices = []
    
    # 获取所有归一化之后的csv文件
    def load_files(self, csv_file_path):
        csv_file_list = []
        # 遍历文件夹下所有文件
        for root, dirs, files in os.walk(csv_file_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_file_list.append(os.path.join(root, file))
        return csv_file_list
    
    # 绘制图像
    def draw_radar_chart(self):
        # 将每个性能参数表格中所有模型的数据取平均值，设备名不变
        train_data_list = pd.DataFrame()
        infer_data_list = pd.DataFrame()
        for csv_file in self.csv_file_list:
            pd_data = pd.read_csv(csv_file)
            
            filename = os.path.basename(csv_file).split(".")[0]
            # 模式
            mode = filename.split("_")[1]
            # 性能参数名称
            param = ""
            for i in range(2, len(filename.split("_"))):
                param += filename.split("_")[i] + "_"
            param = param[:-1]
            # 读取第一行返回为列表
            devices = pd_data.columns.values.tolist()[1:]
            
            # 对每一列数据取平均值
            if mode == "train":
                train_data_list[param] = pd_data[devices].mean()
                
            elif mode == "infer":
                infer_data_list[param] = pd_data[devices].mean()
        
        # 如果路径不存在则创建路径
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # 绘制雷达图
        print("draw infer radar chart")
        self.draw(pd_data=infer_data_list, save_path=self.save_path, mode="infer")
        print("draw train radar chart")
        self.draw(pd_data=train_data_list, save_path=self.save_path, mode="train")    
                
    # 绘制雷达图
    def draw(self, pd_data, save_path, mode):
        # 对pd_data的进行转置
        num_vars = len(pd_data.columns)
        labels = pd_data.columns.tolist() 
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 完成图形的闭合
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)  # 偏移使图形从正上方开始
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], labels)
        
        ax.set_rscale('linear')  # 设置线性刻度
        
        # pd_data的列名为设备名，pd_data的数据为设备的性能参数数据
        # 将多个设备的数据绘制到同一个雷达图中，每个设备的数据用不同的颜色表示
        # 雷达图的每个角度表示一个性能参数
        # 雷达图的每一圈表示一个设备
        
        for i in range(len(pd_data.index)):
            values = pd_data.iloc[i].values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=pd_data.index[i])
            ax.fill(angles, values, alpha=0.1) 

        if mode == "infer":
            path = os.path.join(save_path, "infer")
        elif mode == "train":
            path = os.path.join(save_path, "train")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(mode)
        plt.savefig(path + "\\"  + mode + ".png")
        plt.close()
    
if __name__ == '__main__':
    csv_file_path = ".\\..\\Table\\norm"
    save_path = ".\\..\\Diagram\\Radar_Chart\\combined"
    tool = Radar_Chart(csv_file_path=csv_file_path, save_path=save_path, is_sumed=False)
    tool.draw_radar_chart()