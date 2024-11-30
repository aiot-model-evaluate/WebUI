# -*- coding: utf-8 -*-

# 获取当前绝对路径并回退一级目录，将其加入环境变量
import os
import json
root_dir_inner = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
user_config = os.path.join(root_dir_inner, "config", "user", "user.json")


import streamlit as st
from streamlit_modal import Modal

st.title("交互式模型测评平台")

# 账户登录
username = st.text_input(label="账户", help="请输入账户名")
password = st.text_input(label="密码", type="password", help="请输入密码")

# 登录按钮
if st.button("登录"):
    config = json.load(open(user_config, "r"))
    if username in config.keys() and password == config[username]["password"]:
        st.write("登录成功")
        st.switch_page("pages/evaluation.py")
    else:
        st.write("登录失败")