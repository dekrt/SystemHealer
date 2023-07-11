# 使用官方 Python 运行时作为父镜像
FROM python:3.11-slim-buster

# 设置工作目录
WORKDIR /app

# 将当前目录内容复制到容器的 /app 目录中
COPY . /app

# 安装项目需要的包
RUN pip install --no-cache-dir virtualenv && \
    virtualenv venv && \
    . ./venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# 运行迁移
RUN python3 manage.py migrate

# 使端口 8000 可供此容器外的环境使用
EXPOSE 8000

# 定义环境变量
ENV NAME SystemHealer

# 运行服务器
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]

