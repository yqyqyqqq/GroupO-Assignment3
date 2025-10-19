# Dockerfile

# 使用轻量级基础镜像
FROM python:3.10-slim

WORKDIR /app

# 环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. 接收一个构建参数 (build-arg)
ARG MODEL_VERSION=0.1

# 2. 将这个参数设置为环境变量，以便 src/main.py 可以读取
ENV MODEL_VERSION=${MODEL_VERSION}

# 3. 复制源代码 (现在在 src/ 目录中)
COPY src/ ./src

# 4. 复制对应版本的产物
# (我们假设这些文件已由 CI 流程在 'models/' 目录中生成)
COPY models/model_v${MODEL_VERSION}.joblib ./models/model_v${MODEL_VERSION}.joblib
COPY models/feature_list.json ./models/feature_list.json

# 暴露端口
EXPOSE 8000

# 启动 API，注意路径现在是 src.main
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
