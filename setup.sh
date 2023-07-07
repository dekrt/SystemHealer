#!/bin/bash

# 打印欢迎信息
echo "欢迎使用系统治愈师系统！"

# 安装virtualenv
pip install --break-system-packages virtualenv

# 创建虚拟环境
virtualenv venv

# 激活虚拟环境
source ./venv/bin/activate

# 安装requirements.txt中的依赖
pip install -r requirements.txt

# 迁移数据库
python3 manage.py migrate

# 运行服务器
python3 manage.py runserver 

# 睡眠1s
sleep 1

# 打开浏览器并访问localhost:8000
xdg-open http://localhost:8000
