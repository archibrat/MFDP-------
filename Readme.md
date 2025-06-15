# ML-платформа для оптимизации потоков пациентов в МИС

Универсальная интеллектуальная платформа на базе машинного обучения для оптимизации потоков пациентов и ресурсов медицинских учреждений. Система интегрируется с различными МИС, включая Медиалог, и обеспечивает эффективное прогнозирование посещаемости и оптимизацию расписания работы клиники.

## Содержание

- [Возможности]
    
- [Архитектура]
    
- [Технологии]
    
- [Установка и запуск]
    
- [Структура проекта]
    
- [API]
    
- [ML-модель]
    

##  Возможности

- **Прогнозирование неявок пациентов** с высокой точностью (AUC-ROC >= 0.85)
    
- **Оптимизация расписания врачей** с учетом прогнозируемой нагрузки
    
- **Интеграция с МИС Медиалог** и другими системами через API(Будет добравлено позднее)
    
- **Микросервисная архитектура** для гибкого масштабирования
    
- **Аналитические дашборды** для визуализации потоков пациентов
    
- **REST API** для взаимодействия с фронтенд-приложениями
    
- **Асинхронная обработка задач** через RabbitMQ
    
- **Хранение данных** в PostgreSQL
    

## Архитектура

Система построена на микросервисной архитектуре с использованием Docker контейнеров. Ключевые компоненты:

1. **API сервис** - FastAPI приложение, обрабатывающее запросы
    
2. **ML Worker** - Сервис для асинхронной обработки ML задач
    
3. **База данных** - PostgreSQL для хранения данных
    
4. **Сервис очередей** - RabbitMQ для асинхронного обмена сообщениями
    
5. **Веб-интерфейс** - Frontend приложение для администраторов и врачей
    
6. **Прокси-сервер** - Nginx для балансировки нагрузки
    

## Схема взаимодействия компонентов

``` mermaid 
graph TB
    subgraph "Frontend Layer"
        A[Streamlit UI] --> B[FastAPI Gateway]
        A --> C[Web Interface]
    end
    subgraph "API Layer"
        B --> D[Authentication Service]
        B --> E[ML Prediction Service]
        B --> F[Analytics Service]
    end
    subgraph "ML Layer"
        E --> G[LightGBM Model]
        E --> H[XGBoost Model]
        E --> I[Stacking Ensemble]
    end
    subgraph "Data Layer"
        J[PostgreSQL] --> K[Redis Cache]
        L[RabbitMQ] --> M[ML Workers]
        N[Airflow ETL] --> J
    end

    subgraph "Monitoring"
        O[Prometheus] --> P[Grafana]
        Q[Health Checks] --> O
    end
```




## Технологии

- **Backend**: Python 3.12, FastAPI, SQLModel
    
- **ML**: scikit-learn, LightGBM, XGBoost, pandas, numpy
    
- **База данных**: PostgreSQL
    
- **Брокер сообщений: RabbitMQ
    
- **Контейнеризация**: Docker
    
- **Веб-сервер**: Nginx
    
- **Авторизация**: JWT
    
- **Фронтенд**: HTML, CSS, JavaScript, HTMX
    
- **LLM**: Ollama (Gemma)(Будет убрано в следующей версии)
    

## Установка и запуск

## Предварительные требования

- Docker и Docker Compose
    
- Git
    
- Доступ к исходным данным МИС
    

## Установка

1. Клонировать репозиторий:
    

2. Создать файл окружения `.env` на основе шаблона .env.example:
    

3. Настроить переменные окружения в `.env` файле:
    
## База данных 
    POSTGRES_USER=db_user 
    POSTGRES_PASSWORD=your_password 
    POSTGRES_DB=medical_db 
## RabbitMQ 
    RABBITMQ_USER=rmuser 
    RABBITMQ_PASS=rmpassword 
## Настройки приложения 
    SECRET_KEY=your_secret_key

4. Запустить контейнеры с помощью Docker Compose:
    

bash

`docker compose -f docker-compose.yml up -d --build `

5. Запуск Streamlit интерфейса

`docker exec -d event-planner-api \
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0`

6. Проверить работу системы:
    
bash

`curl http://localhost/health`

Доступ к мониторингу
http://localhost:9090 # Prometheus
http://localhost:3000 # Grafana (admin/admin123)


## Структура проекта

├── app/                  # Основной API сервис
│   ├── api.py            # Точка входа FastAPI приложения
│   ├── auth/             # Модули авторизации 
│   ├── dags/             # DAGs для Airflow(Будет внедрено в следующей версии)
│   ├── database/         # Настройки базы данных
│   ├── models/           # SQLModel модели
│   ├── routes/           # API эндпоинты
│   ├── schemas/          # Схемы БД (Будет перенесено в models)
│   ├── services/         # Бизнес-логика
│   │   ├── crud/         # CRUD операции
│   │   ├── rm/           # Интеграция с RabbitMQ
│   │   ├── auth/         # Авторизации 
│   │   ├── logging/      # Логирование 
│   │   ├── monitoring/   # Мониторинг работы приложения 
│   │   └── ml_service.py # Интеграция с ML моделью
│   └── view/             # HTML шаблоны
├── ml_worker/            # Сервис обработки ML задач
│   ├── rmq/              # Клиенты RabbitMQ
│   ├── llm.py            # Интеграция с LLM
│   └── main.py           # Основной код worker'а
├── models/               # Предобученные ML модели
├── nginx/                # Конфигурация Nginx
├── data/                 # Данные для ML моделей
├── tests/                # Тесты
│   ├── test_api.py       # Тесты API
│   ├── test_db.py        # Тесты базы данных
│   ├── test_ml_integration.py        # Тесты интеграции
│   ├── test_predication_api.py        # Тесты базы данных
│   └── test_events.py    # Тесты обработки событий
├── docker-compose.yaml   # Конфигурация Docker Compose
├── Dockerfile            # Сборка основного приложения
└── README.md             # Документация проекта

medical-ml-platform/
├── app/                    # Основное приложение FastAPI
│   ├── api/               # API routes
│   ├── core/              # Конфигурация и настройки
│   ├── models/            # SQLModel модели
│   ├── services/          # Бизнес-логика
│   └── main.py           # Точка входа FastAPI
├── ml_worker/             # ML воркеры
│   ├── models/           # ML модели
│   ├── preprocessing/    # Обработка данных
│   └── training/         # Обучение моделей
├── dags/                  # Airflow DAGs
├── monitoring/            # Конфигурация мониторинга
├── scripts/              # Утилиты и скрипты
├── tests/                # Тесты
├── docker-compose.yml    # Docker конфигурация
└── requirements.txt      # Python зависимости

## API

## Основные эндпоинты

## Авторизация

- `POST /auth/token` - Получение JWT токена
    
- `GET /auth/login` - Страница авторизации
    
- `POST /auth/login` - Обработка авторизации
    

## События

- `GET /events/` - Получение списка событий
    
- `GET /events/{id}` - Получение события по ID
    
- `POST /events/new` - Создание нового события
    
- `DELETE /events/{id}` - Удаление события
    

## ML операции

- `POST /ml/send_task` - Отправка задачи на обработку ML моделью
    
- `POST /ml/send_task_result` - Сохранение результата обработки
    
- `POST /ml/send_task_rpc` - Синхронный вызов ML модели
    
- `GET /ml/tasks` - Получение списка ML задач
    
- `GET /ml/tasks/{task_id}` - Получение задачи по ID
    

## Пользователи

- `POST /users/signup` - Регистрация нового пользователя
    
- `POST /users/signin` - Вход пользователя
    
- `GET /users/get_all_users` - Получение списка пользователей
    

## Примеры запросов

## Отправка задачи на ML обработку

bash

`curl -X POST "http://localhost/ml/send_task" \   -H "Content-Type: application/json" \  -d '{"message": "Предсказать вероятность неявки пациента", "user_id": 1}'`

## Получение результата обработки

bash

`curl -X GET "http://localhost/ml/tasks/1" \   -H "Authorization: Bearer YOUR_TOKEN"`

## ML-модель

## Описание

ML-модель использует алгоритм xgboost_opt с улучшением через щptuna для прогнозирования вероятности неявки пациента на прием. Модель обучена на наборе данных, содержащем информацию о 100,000+ приемах, и достигает точности 0.8923 (AUC-ROC 0.9729).

## Основные особенности модели

- **Класс DataLoader** - загрузка и интеграция данных из различных источников
    
- **Класс AdvancedFeatureEngineering** - расширенное создание признаков
    
    - Временные признаки (день недели, время дня, цикличность)
        
    - Признаки пациента (история посещений, показатель надежности)
        
    - Признаки записи (время предварительной записи, условия)
        
    - Интеграция внешних данных (погода, транспорт, сезонность)
        
- **Класс IntelligentFeatureProcessor** - интеллектуальная обработка и отбор признаков
    
- **Класс AdvancedModelArchitecture** - продвинутые ансамблевые модели
    
- **Класс PredictionPostprocessor** - постобработка предсказаний
    
- **Класс ModelQualityAnalyzer** - комплексный анализ качества модели
    
- **Класс ModelInterpreter** - интерпретация результатов модели
    

## Используемые алгоритмы

|Алгоритм|Точность|Качество предсказания|
|---|---|---|
|Логистическая регрессия|79.2%|68.6%|
|Дерево решений|79.3%|72.5%|
|Случайный лес|79.3%|68.2%|
|XGBoost|79.6%|74.9%|
|**LightGBM**|**79.6%**|**75.2%**|
|XGBoost|0.8923|0.9729|

