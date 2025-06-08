#В текущей итерации не используется, требуется согласование доступа к БД МИС
#При согласовании переписать, настроить подключения, переписать запросы и добравить airflow в docker 





from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd

default_args = {
    'owner': 'kaa',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False
}

def extract_external_data(**kwargs):
    # Загрузка данных из внешних источников
    from services.data_loader import DataLoader
    data_loader = DataLoader()
    
    # Основной датасет
    dataset = data_loader.load_primary_dataset()
    
    # Синтетические данные
    weather, transport, seasonal = data_loader.generate_synthetic_external_data()
    
    # Сохранение в временное хранилище
    dataset.to_parquet("/tmp/primary_dataset.parquet")
    weather.to_parquet("/tmp/weather.parquet")
    transport.to_parquet("/tmp/transport.parquet")
    seasonal.to_parquet("/tmp/seasonal.parquet")

def transform_data(**kwargs):
    # Обработка данных
    import pandas as pd
    from services.feature_engineering import AdvancedFeatureEngineering
    
    # Загрузка данных
    dataset = pd.read_parquet("/tmp/primary_dataset.parquet")
    weather = pd.read_parquet("/tmp/weather.parquet")
    transport = pd.read_parquet("/tmp/transport.parquet")
    seasonal = pd.read_parquet("/tmp/seasonal.parquet")
    
    # Feature Engineering
    feature_engineer = AdvancedFeatureEngineering()
    processed_data = feature_engineer.process_data(dataset, weather, transport, seasonal)
    
    # Сохранение обработанных данных
    processed_data.to_parquet("/tmp/processed_data.parquet")

def load_to_warehouse(**kwargs):
    # Загрузка в целевую базу
    from sqlmodel import create_engine, SQLModel
    
    engine = create_engine("postgresql+psycopg2://user:pass@db:5432/medical_db")
    SQLModel.metadata.create_all(engine)
    
    df = pd.read_parquet("/tmp/processed_data.parquet")
    df.to_sql("medical_records", engine, if_exists="append", index=False)

def load_redis_cache(**kwargs):
    # Кэширование предсказаний
    redis_hook = RedisHook(redis_conn_id="redis_default")
    r = redis_hook.get_conn()
    
    # Загрузка последних предсказаний
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    df = pg_hook.get_pandas_df("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1000")
    
    for _, row in df.iterrows():
        key = f"prediction:{row['patient_id']}:{row['created_at']}"
        r.hset(key, mapping=row.to_dict())

with DAG(
    'medical_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
) as dag:
    
    extract_task = PythonOperator(
        task_id='extract_external_data',
        python_callable=extract_external_data
    )
    
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )
    
    load_warehouse_task = PythonOperator(
        task_id='load_to_data_warehouse',
        python_callable=load_to_warehouse
    )
    
    load_cache_task = PythonOperator(
        task_id='load_redis_cache',
        python_callable=load_redis_cache
    )
    
    model_monitoring_task = DockerOperator(
        task_id='model_monitoring',
        image='ml_monitor:latest',
        api_version='auto',
        auto_remove=True,
        command="python monitor.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="host"
    )

    extract_task >> transform_task >> [load_warehouse_task, load_cache_task] >> model_monitoring_task
