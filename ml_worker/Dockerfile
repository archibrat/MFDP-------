# ml_worker/Dockerfile.ml
FROM python:3.12-slim as base

WORKDIR /app

# Системные зависимости для ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Промежуточный слой для разработки
FROM base as development
ENV PYTHONPATH=/app
ENV ML_ENV=development
COPY . /app
CMD ["python", "ml_worker_main.py"]

# Продакшн слой
FROM base as production
ENV PYTHONPATH=/app
ENV ML_ENV=production

# Копирование кода
COPY --chown=1000:1000 . /app

# Создание пользователя для безопасности
RUN useradd -m -u 1000 mlworker && chown -R mlworker:mlworker /app
USER mlworker

# Создание директорий
RUN mkdir -p /app/models /app/logs /app/cache

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

CMD ["python", "ml_worker_main.py"]
