# 消安智答 FireSage · 演示 / 试点一键镜像
# 默认轻量模式（FIRESAGE_LITE）：不拉 bge 大模型，启动约数十秒。
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIRESAGE_LITE=1 \
    USE_HF_MIRROR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY backend/ backend/
COPY frontend/ frontend/

# 意图/重排 joblib 已随 backend/data/models 拷入；审计目录运行时可写
RUN mkdir -p backend/data/audit \
    && chmod -R a+rX backend frontend

EXPOSE 8319

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8319/api/system >/dev/null || exit 1

WORKDIR /app/backend
CMD ["python3", "main.py"]
