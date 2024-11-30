# 递归遍历指定路径下的所有文件，对每个文件进行分析，生成结果

import os
import json
import csv
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class generater:
    # 指定待分析数据的根目录与存储结果的根目录
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path
        # 遍历指定路径下的所有文件
        self.filelist = self.findfile()
        self.csv_list, self.json_list = self.align()
        # UTF-8编码
        with open("./../file_list.txt", "w", encoding="utf-8") as f:
            for file in self.filelist:
                f.write(file + "\n")
        self.total_mem = {"A100":'40,960 MiB',"4090":'24,564 MiB',"Jetson-Xanier-NX":'6,833 MiB',"MR-100":'32,768 MiB',"BI-150":'32,768 MiB',"MLU370-M8":'42,396 MiB',"910B":'65,536MiB',"910A":'32,768MiB'}
    
    # 获取文件路径列表
    def findfile(self):
        filelist = []
        # 递归遍历指定路径下的所有文件
        for root, dirs, files in os.walk(self.path):
            for file in files:
                filelist.append(os.path.join(root, file))
        return filelist
    
    # 对齐数据
    def align(self):
        csv_list = []
        json_list = []
        for file in self.filelist:
            if file.endswith(".csv"):
                csv_list.append(file)
            elif file.endswith(".json"):
                json_list.append(file)
        return csv_list, json_list
    
    # 生成结果
    def genarate(self):
        print("generating result...")
        for i in tqdm(range(len(self.csv_list))):
            csv_file = self.csv_list[i]
            json_file = self.json_list[i]
            tokens = csv_file.split("\\")
            device = tokens[-3]
            model = tokens[-2]
            mode = tokens[-1].split(".")[0]
            if mode == "train":
                self.train_analyze(csv_file, json_file, self.save_path, device, model)
            elif mode == "inference":
                self.infer_analyze(csv_file, json_file, self.save_path, device, model)
    
    # 处理训练数据            
    def train_analyze(self, csv_file, json_file, save_path, device, model):
        # 读取csv文件
        # time(s),GPU_Utilization(%),Memory_Utilization(%),Used_Memory(Bytes),Power_Usage(mW),Temperature(C)
        # 0.0,0,3.1308520599250933,806420480,13561,45
        data = pd.read_csv(csv_file)
        # 读取json文件
        with open(json_file, "r") as f:
            json_data = json.load(f)
            
        Items = int(json_data["Items"])                      # 训练参数(来自json文件)

        # 目标参数
        # 训练时间(单位：s)
        training_time = float(json_data["total_time(s)"])      
        # 能耗(单位：mW * s)
        power_consumption = 0                                  
        for i in range(0, len(data) - 1):
            time1 = data.loc[i, "time(s)"]
            time2 = data.loc[i + 1, "time(s)"]
            d_time = time2 - time1
            power_consumption += data.loc[i, "Power_Usage(mW)"] * d_time 
        # 平均功率(单位：mW)
        average_power = power_consumption / training_time
        # 吞吐量 (单位：Items/s)
        troughput = Items / training_time
        # 能效比 (单位：Items/W * e6)
        energy_efficiency = Items / power_consumption * 1000
        # 平均温度(单位：摄氏度)
        Average_Temperature = data["Temperature(C)"].mean()
        # GPU利用率(平均值)
        GPU_Utilization = data["GPU_Utilization(%)"].mean()
        # 单卡显存利用率(平均值)
        Memory_Utilization = data["Memory_Utilization(%)"].mean()
        # 显存总量(单位：Bytes)
        Total_Memory = self.total_mem[device]
        
        
        # save_path
        save_path = os.path.join(self.save_path, device, model)
        # 判断路径是否存在，不存在则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + "/train.report", "w", newline="", encoding="utf-8") as f:
            # 按照txt格式保存结果
            f.write("训练时间: " + str(training_time) + " s\n")
            f.write("平均功率: " + str(average_power) + " mW\n")
            f.write("吞吐量: " + str(troughput) + " Items/s\n")
            f.write("能效比: " + str(energy_efficiency) + " Items/mW\n")
            f.write("平均温度: " + str(Average_Temperature) + " °C\n")
            f.write("GPU利用率: " + str(GPU_Utilization) + " %\n")
            f.write("单卡显存利用率: " + str(Memory_Utilization) + " %\n")
            f.write("显存总量: " + Total_Memory + "\n")
        
             
    # 处理推理数据    
    def infer_analyze(self, csv_file, json_file, save_path, device, model):
        # 读取csv文件
        data = pd.read_csv(csv_file)
        # 读取json文件
        with open(json_file, "r") as f:
            json_data = json.load(f)
            
        Items = int(json_data["Items"])                        # 训练参数(来自json文件)
        total_time = float(json_data["total_time(s)"])         # 推理时间(来自json文件)
        
        # 目标参数
        # 平均前向推理速度
        Average_forward_speed = Items / total_time
        # 数据加载速度
        load_time = float(json_data["load_time(s)"]) 
        # 平均前向传播延迟
        Average_forward_delay = float(json_data["Average_forward_delay(s)"]) 
        # 能耗(mW * s)
        power_consumption = 0                                  
        for i in range(0, len(data) - 1):
            time1 = data.loc[i, "time(s)"]
            time2 = data.loc[i + 1, "time(s)"]
            d_time = time2 - time1
            power_consumption += data.loc[i, "Power_Usage(mW)"] * d_time 
            
        # 平均功率(mW)
        average_power = power_consumption / total_time
        # 能效比(Items/mW)
        energy_efficiency = Items / power_consumption * 1000
        # 平均温度(摄氏度)
        Average_Temperature = data["Temperature(C)"].mean()
        # GPU利用率(平均值)
        GPU_Utilization = data["GPU_Utilization(%)"].mean()
        # 单卡显存利用率(平均值)
        Memory_Utilization = data["Memory_Utilization(%)"].mean()
        # 显存总量(Bytes)
        Total_Memory = self.total_mem[device]
        
        # save_path
        save_path = os.path.join(self.save_path, device, model)
        # 判断路径是否存在，不存在则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + "/infer.report", "w", newline="", encoding="utf-8") as f:
            # 按照txt格式保存结果
            f.write("吞吐量: " + str(Average_forward_speed) + " Items/s\n")
            f.write("数据加载速度: " + str(load_time) + " s\n")
            f.write("平均前向传播延迟: " + str(Average_forward_delay) + " s\n")
            f.write("平均功率: " + str(average_power) + " mW\n")
            f.write("能效比: " + str(energy_efficiency) + " Items/mW\n")
            f.write("平均温度: " + str(Average_Temperature) + " °C\n")
            f.write("GPU利用率: " + str(GPU_Utilization) + " %\n")
            f.write("单卡显存利用率: " + str(Memory_Utilization) + " %\n")
            f.write("显存总量: " + Total_Memory + "\n")
            
    def shoe_items(self):
        # 构建字典
        train_items_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_items_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        print("showing items...")
        for file in tqdm(self.json_list):
            with open(file, "r") as f:
                json_data = json.load(f)
                device = file.split("\\")[-3]
                model = file.split("\\")[-2]
                mode = file.split("\\")[-1].split(".")[0]
                Items = json_data["Items"]
                if mode == "train":
                    train_items_dict[device][model] = Items
                elif mode == "inference":
                    infer_items_dict[device][model] = Items
                    
        # 保存为csv文件
        train_df = pd.DataFrame(train_items_dict)
        infer_df = pd.DataFrame(infer_items_dict)
        train_df.to_csv("./../train_items.csv")
        infer_df.to_csv("./../infer_items.csv")
            
    def show_table(self):
        save_path = ".\..\Table"
        # 判断路径是否存在，没有就创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 构建吞吐量的字典
        train_throughput_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_throughput_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建平均功率的字典
        train_power_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_power_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建能效比的字典
        train_energy_efficiency_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}  
        infer_energy_efficiency_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建GPU利用率的字典
        train_GPU_Utilization_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_GPU_Utilization_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建显存利用率的字典
        train_Memory_Utilization_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_Memory_Utilization_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建平均温度的字典
        train_average_temperature_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        infer_average_temperature_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        # 构建平均前向推理时延的字典
        infer_average_inference_delay_dict = {"4090":{}, "A100":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        
        # 遍历self.save_path下的所有文件
        print("gerating table...")
        file_list = []
        for root, dirs, files in os.walk(self.save_path):
            for file in files:
                file_list.append(os.path.join(root, file))
        for file in tqdm(file_list):
            tokens = file.split("\\")
            device = tokens[-3]
            model = tokens[-2]
            mode = tokens[-1].split(".")[0]
            if mode == "train":
                with open(file, "r", encoding="UTF-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "吞吐量" in line:
                            train_throughput_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "平均功率" in line:
                            train_power_dict[device][model] = float(line.split(":")[1].split(" ")[1]) / 1000
                        elif "能效比" in line:
                            train_energy_efficiency_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "GPU利用率" in line:
                            train_GPU_Utilization_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "单卡显存利用率" in line:
                            train_Memory_Utilization_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "平均温度" in line:
                            train_average_temperature_dict[device][model] = float(line.split(":")[1].split(" ")[1])
            elif mode == "infer":
                with open(file, "r", encoding="UTF-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "吞吐量" in line:
                            infer_throughput_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "平均功率" in line:
                            infer_power_dict[device][model] = float(line.split(":")[1].split(" ")[1]) / 1000
                        elif "能效比" in line:
                            infer_energy_efficiency_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "GPU利用率" in line:
                            infer_GPU_Utilization_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "单卡显存利用率" in line:
                            infer_Memory_Utilization_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "平均温度" in line:
                            infer_average_temperature_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                        elif "平均前向传播延迟" in line:
                            infer_average_inference_delay_dict[device][model] = float(line.split(":")[1].split(" ")[1])
                                
        # 保存为csv文件
        train_throughput_df = pd.DataFrame(train_throughput_dict)
        infer_throughput_df = pd.DataFrame(infer_throughput_dict)
        train_power_df = pd.DataFrame(train_power_dict)
        infer_power_df = pd.DataFrame(infer_power_dict)
        train_energy_efficiency_df = pd.DataFrame(train_energy_efficiency_dict)
        infer_energy_efficiency_df = pd.DataFrame(infer_energy_efficiency_dict)
        train_GPU_Utilization_df = pd.DataFrame(train_GPU_Utilization_dict)
        infer_GPU_Utilization_df = pd.DataFrame(infer_GPU_Utilization_dict)
        train_Memory_Utilization_df = pd.DataFrame(train_Memory_Utilization_dict)
        infer_Memory_Utilization_df = pd.DataFrame(infer_Memory_Utilization_dict)
        train_average_temperature_df = pd.DataFrame(train_average_temperature_dict)
        infer_average_temperature_df = pd.DataFrame(infer_average_temperature_dict)
        infer_average_inference_delay_df = pd.DataFrame(infer_average_inference_delay_dict)
        
        # 保存为csv文件
        train_throughput_df.to_csv(save_path + "/train_throughput.csv")
        infer_throughput_df.to_csv(save_path + "/infer_throughput.csv")
        train_power_df.to_csv(save_path + "/train_power.csv")
        infer_power_df.to_csv(save_path + "/infer_power.csv")
        train_energy_efficiency_df.to_csv(save_path + "/train_energy_efficiency.csv")
        infer_energy_efficiency_df.to_csv(save_path + "/infer_energy_efficiency.csv")
        train_GPU_Utilization_df.to_csv(save_path + "/train_GPU_Utilization.csv")
        infer_GPU_Utilization_df.to_csv(save_path + "/infer_GPU_Utilization.csv")
        train_Memory_Utilization_df.to_csv(save_path + "/train_Memory_Utilization.csv")
        infer_Memory_Utilization_df.to_csv(save_path + "/infer_Memory_Utilization.csv")
        train_average_temperature_df.to_csv(save_path + "/train_average_temperature.csv")
        infer_average_temperature_df.to_csv(save_path + "/infer_average_temperature.csv")
        infer_average_inference_delay_df.to_csv(save_path + "/infer_average_inference_delay.csv")
        
    # 生成图表(diagram)
    # 生成功率-时间图，GPU利用率-时间图，温度-时间图
    def show_diagram(self):
        save_path = ".\..\Diagram\Origin"
        LLM_save_path = ".\..\Diagram\LLM"
        LLM = ["Qwen2","GLM4","llama3"]
        # 判断路径是否存在，没有就创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 读取".\..\Temp_File"下的所有csv文件
        csv_file_list = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".csv"):
                    csv_file_list.append(os.path.join(root, file))
        
        # 生成原始图表
        print("generating origin diagram...")
        for file in tqdm(csv_file_list):
            device = file.split("\\")[-3]
            model = file.split("\\")[-2]
            mode = file.split("\\")[-1].split(".")[0]
            data = pd.read_csv(file)
            time = data["time(s)"]
            GPU_Utilization = data["GPU_Utilization(%)"]
            Temperature = data["Temperature(C)"]
            Power = data["Power_Usage(mW)"]
            # GPU利用率-时间图
            plt.figure()
            plt.plot(time, GPU_Utilization, color="red", label="GPU Utilization")
            plt.xlabel("Time(s)")
            plt.ylabel("GPU Utilization(%)")
            plt.title(device + "-" + model + "-" + mode + "-GPU Utilization-Time")
            plt.legend()
            # if model in LLM:
            #     plt.savefig(LLM_save_path + "/" + device + "-" + model + "-" + mode + "-GPU Utilization-Time.png")
            plt.savefig(save_path + "/" + device + "-" + model + "-" + mode + "-GPU Utilization-Time.png")
            plt.close()
            # 温度-时间图
            plt.figure()
            plt.plot(time, Temperature, color="blue", label="Temperature")
            plt.xlabel("Time(s)")
            plt.ylabel("Temperature(C)")
            plt.title(device + "-" + model + "-" + mode + "-Temperature-Time")
            plt.legend()
            # if model in LLM:
            #     plt.savefig(LLM_save_path + "/" + device + "-" + model + "-" + mode + "-Temperature-Time.png")
            plt.savefig(save_path + "/" + device + "-" + model + "-" + mode + "-Temperature-Time.png")
            plt.close()
            # 功率-时间图(单位换算为W)
            plt.figure()
            plt.plot(time, Power / 1000, color="green", label="Power")
            plt.xlabel("Time(s)")
            plt.ylabel("Power(W)")
            plt.title(device + "-" + model + "-" + mode + "-Power-Time")
            plt.legend()
            # if model in LLM:
            #     plt.savefig(LLM_save_path + "/" + device + "-" + model + "-" + mode + "-Power-Time.png")
            plt.savefig(save_path + "/" + device + "-" + model + "-" + mode + "-Power-Time.png")
            plt.close()
        
        # 生成大模型图表(三个大模型的各个参数分别绘制到一个图表中)
        time_dict = {"A100":{}, "4090":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        GPU_Utilization_dict = {"A100":{}, "4090":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        Temperature_dict = {"A100":{}, "4090":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        Power_dict = {"A100":{}, "4090":{}, "Jetson-Xanier-NX":{}, "MR-100":{}, "BI-150":{}, "MLU370-M8":{}, "910B":{}, "910A":{}}
        for file in csv_file_list:
            device = file.split("\\")[-3]
            model = file.split("\\")[-2]
            mode = file.split("\\")[-1].split(".")[0]
            if model in LLM:
                data = pd.read_csv(file)
                time = data["time(s)"]
                GPU_Utilization = data["GPU_Utilization(%)"]
                Temperature = data["Temperature(C)"]
                Power = data["Power_Usage(mW)"]
                time_dict[device][model] = time
                GPU_Utilization_dict[device][model] = GPU_Utilization
                Temperature_dict[device][model] = Temperature
                Power_dict[device][model] = Power
        # GPU利用率-时间图
        print("generating LLM GPU Utilization-Time diagram...")
        for device in tqdm(time_dict.keys()):
            if device == "Jetson-Xanier-NX":
                continue
            else:
                # 要求使用不同颜色的线条区分开三个大模型，绘制到一个图表中
                plt.figure()
                for model in time_dict[device].keys():
                    plt.plot(time_dict[device][model], GPU_Utilization_dict[device][model], label=model)
                plt.xlabel("Time(s)")
                plt.ylabel("GPU Utilization(%)")
                plt.title(device + "-GPU Utilization-Time")
                if any(model for model in time_dict[device].keys()):
                    plt.legend()
                plt.savefig(LLM_save_path + "/" + device + "-GPU Utilization-Time.png")
                plt.close()

        # 温度-时间图
        print("generating LLM Temperature-Time diagram...")
        for device in tqdm(time_dict.keys()):
            if device == "Jetson-Xanier-NX":
                continue
            else:
                plt.figure()
                for model in time_dict[device].keys():
                    plt.plot(time_dict[device][model], Temperature_dict[device][model], label=model)
                plt.xlabel("Time(s)")
                plt.ylabel("Temperature(C)")
                plt.title(device + "-Temperature-Time")
                if any(model for model in time_dict[device].keys()):
                    plt.legend()
                plt.savefig(LLM_save_path + "/" + device + "-Temperature-Time.png")
                plt.close()

        # 功率-时间图(单位换算为W)
        print("generating LLM Power-Time diagram...")
        for device in tqdm(time_dict.keys()):
            if device == "Jetson-Xanier-NX":
                continue
            else:
                plt.figure()
                for model in time_dict[device].keys():
                    plt.plot(time_dict[device][model], Power_dict[device][model] / 1000, label=model)
                plt.xlabel("Time(s)")
                plt.ylabel("Power(W)")
                plt.title(device + "-Power-Time")
                if any(model for model in time_dict[device].keys()):
                    plt.legend()
                plt.savefig(LLM_save_path + "/" + device + "-Power-Time.png")
                plt.close()
                          
if __name__ == "__main__":
    path = ".\..\Temp_File"
    save_path = ".\..\Output"
    tool = generater(path, save_path)
    tool.genarate()
    tool.shoe_items()
    tool.show_table()
    tool.show_diagram()