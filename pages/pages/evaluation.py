# -*- coding: utf-8 -*-

import os
import json
from mypack.run.execute import exec

# 获取当前绝对路径并回退一级目录，将其加入环境变量
import os
root_dir_inner = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 获取选项列表，返回对应文件路径(绝对路径)
def get_src_file(choice):
    # 从后端获取源代码文件
    if choice == {}:
        return None
    else:
        device = choice['device']
        model = choice['model']
        task = choice['task']
        src_file = os.path.join(root_dir_inner, f"./backend/standard_src/{device}/{model}/{task}.py")
    return src_file

# 确认按钮的回调函数
def confirm():
    import streamlit as st
    st.session_state.confirm = True

# 取消按钮的回调函数
def cancel():
    import streamlit as st
    st.session_state.cancel = True

# 本文件负责实现交互式的模型测评任务，具体页面结构参考./scripts/evaluation.png
import streamlit as st
from streamlit_modal import Modal
from streamlit_ace import st_ace

choice = {}

if 'confirm' not in st.session_state:
    st.session_state.confirm = False
    
if 'cancel' not in st.session_state:
    st.session_state.cancel = False

if 'code' not in st.session_state:
    st.session_state.code = ""

#获取配置文件
device_model_json = json.load(open(os.path.join(root_dir_inner, "./config/devices/device_model.json"), "r"))
device_list = list(device_model_json.keys())
# 定义页面元素
# 侧边栏
# 标题
st.title("交互式模型测评平台")
# 一系列选择栏与其名称
# 设备
device = st.selectbox("选择设备", device_list)

model_list = device_model_json[device]

# 模型
model = st.selectbox("选择模型", model_list)

# 任务
task = st.selectbox("选择任务", ["infer", "train"])

modelx_json = json.load(open(os.path.join(root_dir_inner, f"./config/models/{model}/{model}.json"), "r"))
dataset_list = modelx_json[task]["datasets"]

# 数据集
dataset = st.selectbox("选择数据集", dataset_list)

# ----------- 超参选择 ------------
st.write("超参选择")

# 定义一个列表(这个列表由后端模型的类进行返回，以确保超参与模型对应)
# 读取超参配置文件
hyperparameters = modelx_json[task]["hyperparameters"]
hyper_choice = {}

for name in hyperparameters.keys():
    # 滑动条
    hyper_choice[name] = st.slider(name, hyperparameters[name][0], hyperparameters[name][1])
    
choice = {"device": device, "model": model, "dataset": dataset, "task": task}
choice["hyperparameters"] = hyper_choice
        
# 代码编辑框

# 未来排版时加入上传功能
# uploaded_file = st.file_uploader("上传源代码文件", type=["py"])

# 读取一个本地py文件
src_file = get_src_file(choice)
if src_file == None:
    try:
        code = open(os.path.join(root_dir_inner, f"./backend/storage/user0/code/temp/cache.py"), "r", encoding="utf-8").read()
    except:
        code = ""
else:
    code = open(src_file, "r", encoding="utf-8").read()
    st.session_state.code = code

# 创建一个编辑器
content = st_ace(st.session_state.code, language="python")

# 将用户修改后代码放到cache中
if content:
    st.session_state.code = content
    open(os.path.join(root_dir_inner, f"./backend/storage/user0/code/temp/cache.py"), "w", encoding="UTF-8").write(content)

# 提交运行
# 确认弹窗
modal = Modal(key="confirm",title="提交并运行")
# 确认按钮
open_modal = st.button(label='提交运行')
msg = f"您选择的选项为：\n设备：{device}\n模型：{model}\n数据集：{dataset}\n任务：{task}\n"
for name in hyperparameters.keys():
    msg += f"{name}：{hyper_choice[name]}\n"
if open_modal:
    with modal.container():
        st.markdown(msg)
        # 使用回调函数拉起后端处理
        st.button(label='确认', on_click=confirm)
        st.button(label='取消',on_click=cancel)

# 如果按了确认按钮
if st.session_state.confirm:
    # TODO:处理动作
    exec(choice)
    st.session_state.confirm = False

# 如果按了取消按钮
if st.session_state.cancel:
    # TODO:处理动作
    st.session_state.cancel = False
    
# 跳转分析页面按钮
if st.button("结果分析"):   
    st.switch_page("pages/analyze.py")
    
    