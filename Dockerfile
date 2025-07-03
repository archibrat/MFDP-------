# Многоэтапный Dockerfile для MFDP Medical ML Platform
FROM python:3.12-slim as base

# Установка системных зависимостей для компиляции
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    build-essential \
    libpq-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app

# =============================================================================
# Этап для основного приложения
# =============================================================================
FROM base as app-builder

WORKDIR /app

# Копирование и установка зависимостей приложения
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Этап для ML worker
# =============================================================================
FROM base as ml-worker-builder

WORKDIR /app

# Копирование и установка зависимостей ML worker
COPY ml_worker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Финальный этап для основного приложения
# =============================================================================
FROM python:3.12-slim as app

# Установка только необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    libpq5 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Копирование Python пакетов из builder
COPY --from=app-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=app-builder /usr/local/bin /usr/local/bin

# Копирование кода приложения
COPY app/ .

# Создание необходимых директорий
RUN mkdir -p models logs cache && \
    chown -R app:app /app

USER app

EXPOSE 8080

CMD ["python", "api.py"]

# =============================================================================
# Финальный этап для ML worker
# =============================================================================
FROM python:3.12-slim as ml-worker

# Установка только необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    libpq5 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Копирование Python пакетов из builder
COPY --from=ml-worker-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=ml-worker-builder /usr/local/bin /usr/local/bin

# Копирование кода ML worker
COPY ml_worker/ .

# Создание необходимых директорий
RUN mkdir -p models logs cache && \
    chown -R app:app /app

USER app

EXPOSE 8080

CMD ["python", "main.py"] 