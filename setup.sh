#!/bin/bash

# 安装 virtualenv
pip install virtualenv

# 创建虚拟环境
virtualenv venv

# 激活虚拟环境
source ./venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 迁移数据库
python3 manage.py migrate

# 运行服务器
python3 manage.py runserver

# 睡眠1s
sleep 1

# 打开浏览器并访问localhost:8000
xdg-open http://localhost:8000
