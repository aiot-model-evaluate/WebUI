# -*- coding: utf-8 -*-
import paramiko
import os
import json
import time

ssh_config_path = os.path.abspath("E:\\Project\\model_test\\stage2\\Streamlit\\code\\config\\SSH")
code_cache_path = os.path.abspath("E:\\Project\\model_test\\stage2\\Streamlit\\code\\backend\\storage\\user0\\code\\temp")
result_cache_path = os.path.abspath("E:\\Project\\model_test\\stage2\\Streamlit\\code\\backend\\storage\\user0\\original_result\\temp")
src_code_path = os.path.abspath("E:\\Project\\model_test\\stage2\\Streamlit\\code\\backend\\standard_src")

class ssh_connect():
    def __init__(self, host, username, password):
        pass

## 该函数需要实现的功能如下：
# 1. 通过paramiko模块连接到远程服务器
# 2. 上传测试文件与monitor.py到远程服务器工作目录下
# 3. 激活虚拟环境
# 4. 执行测试代码
# 5. 等待执行结果
# 6. 从远程服务器下载结果文件
def exec(choice):
    device = choice['device']
    model = choice['model']
    task = choice['task']
    dataset = choice['dataset']
    hyperparameters = choice['hyperparameters']
    
    #-------------step1.连接到服务器-------------
    ssh_config = json.load(open(os.path.join(ssh_config_path, "device.json"), "r"))
    hostname = ssh_config[device]["hostname"]
    port = ssh_config[device]["port"]
    username = ssh_config[device]["username"]
    password = ssh_config[device]["password"]
    private_key_type = ssh_config[device]["private_key_type"]
    
    # 创建SSH客户端
    client = paramiko.SSHClient()
    
    if private_key_type != "":
        private_key_path = os.path.join(ssh_config_path, "private_key", device, f"{device}.key")

        if private_key_type == "rsa":
            private_key = paramiko.RSAKey(filename=private_key_path)
        elif private_key_type == "ED25519":
            private_key = paramiko.Ed25519Key(filename=private_key_path)
            
        # 自动添加主机密钥
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 连接远程主机，使用SSH密钥进行身份验证
        client.connect(hostname, port, username, pkey=private_key)
    
    if  password != "":
        client.connect(hostname, port, username, password)
    
    # 创建一个交互式的shell
    shell = client.invoke_shell()
        
    #----step2.上传测试文件与monitor.py到远程服务器工作目录下----
    # 上传测试文件
    sftp = client.open_sftp()
    sftp.put(os.path.abspath("E:\\Project\\model_test\\stage2\\Streamlit\\code\\backend\\storage\\user0\\code\\temp\\cache.py"), f"/home/{username}/model_test/{model}/{task}.py") 
    # 上传monitor.py
    monitor_file_path = os.path.join(src_code_path, device, f"{device}_Monitor.py")
    sftp.put(monitor_file_path, f"/home/{username}/model_test/{model}/monitor.py")
    
    #------------step4.执行测试代码 + step5.等待执行结果------------
    
    shell.send(f"cd /home/{username}/model_test/{model}\n")
    time.sleep(0.1)
    shell.send(f"conda activate llm\n")
    time.sleep(1)
    shell.send(f"rm -f {task}_Result.log\n")
    time.sleep(0.1)
    shell.send(f"python {task}.py\n")
    time.sleep(0.1)
    while(True):
        time.sleep(1)
        output = shell.recv(2048).decode("utf-8")
        if "Program finished." in output:
            break
        
    # TODO:进行模型测试的时间是不确定的，该如何处理这里的时延？
    
    #------------step6.从远程服务器下载结果文件------------
    result_file_path = os.path.join(result_cache_path, f"{task}_Result.log")
    sftp.get(f"/home/{username}/model_test/{model}/{task}_Result.log", result_file_path)
    # 关闭sftp连接
    sftp.close()
    # 关闭shell连接
    shell.close()
    # # 关闭SSH连接
    client.close()
    