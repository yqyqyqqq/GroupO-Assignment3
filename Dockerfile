# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码和模型文件
COPY app_v0_2.py .
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "app_v0_2:app", "--host", "0.0.0.0", "--port", "8000"]
