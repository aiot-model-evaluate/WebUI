# 该文件负责读取原始记录文件，将其转换为标准格式，以便后续处理
# 提示使用者提供输入文件路径，输出文件路径，指定运行模式(训练/推理)

import os
import csv
import sys
import argparse

# 获取当前文件路径
PATH = os.path.dirname(os.path.abspath(__file__))

class Format2Standard:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.device = self.select_device()
        self.model = self.select_model()
        self.mode = self.select_mode()
        self.record = self.record()
        # 存储路径(self.output_path/self.device/self.model/self.mode.csv)
        self.output_path = os.path.join(self.output_path, self.device, self.model, self.mode + '.csv')
        self.temp_data = []
        print(f"device: {self.device}, model: {self.model}, mode: {self.mode}")

    # 选择设备
    def select_device(self):
        device_list = ['A100','4090','Jetson Xanier NX','MR-100','BI-150','MLU370-M8','910B', '910A']
        print("\n---------------------------------------------------------------------")
        print("You can select device from below:")
        for device in device_list:
            print(f"choose {device} by input {device_list.index(device)}")
        device = input("Please select the device you want to use:")
        print(f"Your choice is {device_list[int(device)]}")
        return device_list[int(device)]
    
    # 选择模型    
    def select_model(self):
        model_list = ['llama3','GLM4','Qwen2','Bert','LDM','Unet','Resnet','LSTM','yolo_v10']
        print("\n---------------------------------------------------------------------")
        print("You can select model from below:")
        for model in model_list:
            print(f"choose {model} by input {model_list.index(model)}")
        model = input("Please select the model you want to use:")
        print(f"Your choice is {model_list[int(model)]}")
        return model_list[int(model)]
        
    # 选择模式
    def select_mode(self):
        mode_list = ['train','inference']    
        print("\n---------------------------------------------------------------------")
        print("You can select mode from below:")
        for mode in mode_list:
            print(f"choose {mode} by input {mode_list.index(mode)}")
        mode = input("Please select the mode you want to use:")
        print(f"Your choice is {mode_list[int(mode)]}")
        return mode_list[int(mode)]
    
    # 收集用户手动输入的数据
    def record(self):
        record = {}
        if (self.mode == 'train'):
            record['Items'] = input("Please input the items in training:")
            record['total_time(s)'] = input("Please input the total time in training:")
        elif (self.mode == 'inference'):
            record['Items'] = input("Please input the items in inference:")
            record['load_time(s)'] = input("Please input the load time in inference:")
            record['Average_forward_delay(s)'] = input("Please input the average forward delay in inference:")
            record['total_time(s)'] = input("Please input the total time in inference:")
        return record
    # 转换
    def convert(self):
        # 读取原始记录文件
        temp_data = []
        # 键值对+json格式(pynvml monitor)
        if(self.device in ['A100','4090']):
            with open(self.input_path, 'r') as f:
                data = f.readlines()
            baseline = data[0]
            base_Power_Usage = int(baseline.split("'Power Usage (mW)': ")[1].split(",")[0])
            Total_memory = int(baseline.split("'Total Memory (bytes)': ")[1].split(",")[0])
            # line : "1722309756.131607: {'GPU Utilization (%)': 1, 'Memory Utilization (%)': 0, 'Total Memory (bytes)': 25757220864, 'Used Memory (bytes)': 913375232, 'Free Memory (bytes)': 24843845632, 'Power Usage (mW)': 31346, 'Temperature (C)': 56}"
            start_time = float(data[1].split(':')[0])
            for line in data[1:]:
                try:
                    time_now = float(line.split(':')[0]) - start_time
                    GPU_Utilization = int(line.split("'GPU Utilization (%)': ")[1].split(",")[0])
                    Used_Memory = int(line.split("'Used Memory (bytes)': ")[1].split(",")[0])
                    Memory_Utilization = 100 * (Used_Memory / Total_memory)
                    Power_Usage = int(line.split("'Power Usage (mW)': ")[1].split(",")[0])
                    Temperature = int(line.split("'Temperature (C)': ")[1].split("}")[0])
                    temp_data.append([time_now, GPU_Utilization, Memory_Utilization, Used_Memory, Power_Usage, Temperature])
                except:
                    break
                
        # 键值对+json格式(jtop monitor)
        elif(self.device in ['Jetson Xanier NX']):
            temp_data = []
            with open(self.input_path, 'r') as f:
                data = f.readlines()
            baseline = data[0]
            base_Power_Usage = int(baseline.split("'Power Usage (mW)': ")[1].split(",")[0])
            Total_memory = int(baseline.split("'Total Memory (bytes)': ")[1].split(",")[0])
            
            # line : "1723155675.4301682: {'GPU Utilization (%)': 0.0, 'Total Memory (bytes)': 6997092, 'Used Memory (bytes)': 2849104, 'Free Memory (bytes)': 3333644, 'Power Usage (mW)': 6440, 'Temperature (C)': 38.0}"
            start_time = float(data[1].split(':')[0])
            for line in data[1:]:
                try:
                    time_now = float(line.split(':')[0]) - start_time
                    GPU_Utilization = float(line.split("'GPU Utilization (%)': ")[1].split(",")[0])
                    Used_Memory = int(line.split("'Used Memory (bytes)': ")[1].split(",")[0])
                    Memory_Utilization = 100 * (Used_Memory / Total_memory)
                    Power_Usage = int(line.split("'Power Usage (mW)': ")[1].split(",")[0]) - base_Power_Usage
                    Temperature = float(line.split("'Temperature (C)': ")[1].split("}")[0])
                    temp_data.append([time_now, GPU_Utilization, Memory_Utilization, Used_Memory, Power_Usage, Temperature])
                except:
                    break
            
        # 键值对+json格式(ixsmi monitor)        
        elif(self.device in ['MR-100','BI-150','910B','910A']):
            with open(self.input_path, 'r') as f:
                data = f.readlines()
            baseline = data[0]
            Total_memory = int(baseline.split("'Total Memory (MiB)': ")[1].split(",")[0].replace("'", "")) * 1024 * 1024   # MiB -> Bytes
            base_Power_Usage = int(float(baseline.split("'Power Usage (W)': ")[1].split(",")[0].replace("'", "")))
            
            # line : "1723106336.666979: {'GPU Utilization (%)': '98', 'Memory Utilization (%)': 1.715087890625, 'Total Memory (MiB)': '32768', 'Used Memory (MiB)': '562', 'Free Memory (MiB)': '32206', 'Power Usage (W)': '102', 'Temperature (C)': '38'}"        
            start_time = float(data[1].split(':')[0])
            for line in data[1:]:
                try:
                    time_now = float(line.split(':')[0]) - start_time
                    GPU_Utilization = int(line.split("'GPU Utilization (%)': ")[1].split(",")[0].replace("'", ""))
                    Used_Memory = int(line.split("'Used Memory (MiB)': ")[1].split(",")[0].replace("'", "")) * 1024 * 1024               # MiB -> Bytes
                    Memory_Utilization = 100 * (Used_Memory / Total_memory)
                    Power_Usage = int(float(line.split("'Power Usage (W)': ")[1].split(",")[0].replace("'", "")) * 1000)   # W -> mW
                    Temperature = int(float(line.split("'Temperature (C)': ")[1].split("}")[0].replace("'", "")))
                    temp_data.append([time_now, GPU_Utilization, Memory_Utilization, Used_Memory, Power_Usage, Temperature])
                except:
                    break
        
        # csv格式(cndev monitor)
        if(self.device in ['MLU370-M8']):
            # Timestamp, GPU Utilization, Memory Utilization, Power Usage, Temperature
            # 1723023945.564, 0, 1600, 56, 36
            with open(self.input_path, 'r') as f:
                data = list(csv.reader(f))
            
            baseline = data[1]
            start_time = float(data[2][0])
            
            for line in data[2:]:
                try:
                    time_now = float(line[0]) - start_time
                    GPU_Utilization = int(line[1])
                    Used_Memory = int(line[2]) * 1024 * 1024        # MiB -> Bytes
                    Memory_Utilization = 100 * (Used_Memory / (42396 * 1024 * 1024))  # 42396 MiB
                    Power_Usage = int(line[3]) * 1000   # W -> mW
                    Temperature = int(line[4])
                    temp_data.append([time_now, GPU_Utilization, Memory_Utilization, Used_Memory, Power_Usage, Temperature])
                except:
                    break
                
        
        # 转换为标准格式
        self.temp_data = temp_data       
    
    # 保存
    def save(self):
        # 判断输出路径是否存在
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        # 全部保存为csv格式
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time(s)', 'GPU_Utilization(%)', 'Memory_Utilization(%)', 'Used_Memory(Bytes)', 'Power_Usage(mW)', 'Temperature(C)'])
            writer.writerows(self.temp_data)
        
        # 保存record(使用json格式)
        with open(self.output_path.replace('.csv', '.json'), 'w') as f:
            f.write(str(self.record).replace("'", '"'))

if __name__ == '__main__':
    # 初始化Format2Standard类
    # input_path = input("Please input the path of the file you want to convert:")
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input', required=True, help='path to the input file')
    
    # 获取输入文件路径
    args = vars(arg.parse_args())
    input_path = args['input']
    output_path = os.path.join(PATH, './../Temp_File')
    format_tool = Format2Standard(input_path, output_path)

    # 进行转换
    format_tool.convert()
    
    # 保存为中间文件(中间文件路径里面带一个json描述文件，后续可以使用统一的代码像load模型一下直接对该路径下内容进行处理)
    format_tool.save()