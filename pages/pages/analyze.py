# -*- coding: utf-8 -*-

import os

root_dir_inner = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_images(choice_list):
    import os
    image_list = []
    # 查看文件夹下所有文件
    for root, dirs, files in os.walk("./../cache/analyzed_result/"):
        for file in files:
            image_list.append(os.path.join(root, file))
    return image_list

# 这个页面负责显示出后端分析处理之后的结果
import streamlit as st
from streamlit_modal import Modal
from streamlit_ace import st_ace
import json


# 定义页面元素
# 标题
st.title("测评结果展示")

# 获取选项列表
choice_list = json.load(open("./../../config/analyze/analyze_choice.json", "r"))["choices"]

# 选择想要展示的数据/图表
for choice in choice_list:
    st.checkbox(choice)

# 获取图像绝对路径
image_path_list = get_images(choice_list)    
    
# 动态显示结果
for image_path in image_path_list:
    # 显示文本信息
    # st.write()
    st.image(image_path)