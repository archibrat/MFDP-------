import pandas as pd
import numpy as np
import streamlit as st
from sqlmodel import Session, select
from app.models.prediction import ModelMetrics
from database.database import create_db_and_tables, engine
from models.prediction import PredictionResult
from services.ml_service import get_ml_service
import matplotlib.pyplot as plt
import os
import requests

# Инициализация базы данных
def init_db():
    create_db_and_tables()
    
# Загрузка тестовых данных
@st.cache_data
def load_test_data():
    test_data = {
        "age": 45,
        "gender": "f",
        "district": "central",
        "scholarship": False,
        "condition_a": True,
        "condition_b": False,
        "condition_c": False,
        "accessibility_level": 0,
        "notification_sent": True,
        "planned_date": "2025-06-10T09:00:00",
        "session_date": "2025-06-11T10:00:00"
    }
    return test_data

# Визуализация метрик моделей
def plot_model_metrics(metrics):
    fig, ax = plt.subplots(figsize=(10,6))
    models = list(metrics.keys())
    auc_scores = [m['roc_auc'] for m in metrics.values()]
    
    ax.barh(models, auc_scores, color='skyblue')
    ax.set_xlabel('AUC-ROC Score')
    ax.set_title('Сравнение производительности моделей')
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def main():
    st.set_page_config(
        page_title="Медицинская аналитическая платформа",
        page_icon="🏥",
        layout="wide"
    )
    
    init_db()
    ml_service = get_ml_service()
    
    with st.sidebar:
        st.header("Навигация")
        page = st.selectbox(
            "Выберите раздел",
            ["Главная", "Прогнозирование неявок", "Аналитика", "Iris Dataset"]
        )
        
        if page == "Прогнозирование неявок":
            st.info("""
                **Инструкция:**
                1. Заполните данные пациента
                2. Нажмите 'Сделать прогноз'
                3. Просмотрите рекомендации
            """)
            
        if page == "Аналитика":
            st.info("""
                **Доступные данные:**
                - История предсказаний
                - Метрики моделей
                - Распределение рисков
            """)

    if page == "Главная":
        st.title("Медицинская аналитическая платформа")
        st.image("https://example.com/medical-analytics.jpg", use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                ### Возможности платформы:
                - Прогнозирование неявок пациентов
                - Анализ факторов риска
                - Оптимизация расписания
                - Интеграция с МИС
            """)
            
        with col2:
            st.markdown("""
                ### Преимущества:
                - Точность прогнозов до 85%
                - Снижение простоев на 30%
                - Персонализированные рекомендации
                - Реальное время обработки
            """)
            
    elif page == "Прогнозирование неявок":
        st.title("Прогнозирование риска неявки пациента")
        
        test_data = load_test_data()
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Возраст", min_value=0, max_value=120, value=test_data["age"])
                gender = st.selectbox("Пол", ["m", "f"], index=1 if test_data["gender"] == "f" else 0)
                district = st.text_input("Район", value=test_data["district"])
                scholarship = st.checkbox("Льготник", value=test_data["scholarship"])
                
            with col2:
                condition_a = st.checkbox("Гипертония", value=test_data["condition_a"])
                condition_b = st.checkbox("Диабет", value=test_data["condition_b"])
                condition_c = st.checkbox("Алкоголизм", value=test_data["condition_c"])
                accessibility = st.selectbox("Группа инвалидности", 
                    options=[0,1,2,3,4], 
                    index=test_data["accessibility_level"]
                )
                notification = st.checkbox("SMS отправлено", value=test_data["notification_sent"])
                
            submitted = st.form_submit_button("Сделать прогноз")
            
        if submitted:
            with st.spinner("Анализ данных..."):
                try:
                    # Формирование запроса
                    patient_data = {
                        "age": age,
                        "gender": gender,
                        "district": district,
                        "scholarship": scholarship,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "condition_c": condition_c,
                        "accessibility_level": accessibility,
                        "notification_sent": notification,
                        "planned_date": test_data["planned_date"],
                        "session_date": test_data["session_date"]
                    }
                    
                    # Получение предсказания
                    prediction = ml_service.predict(patient_data)
                    
                    # Отображение результатов
                    st.subheader("Результат прогнозирования")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Вероятность неявки", f"{prediction.prediction_value*100:.1f}%")
                    with col2:
                        st.metric("Уровень риска", prediction.risk_level)
                    with col3: 
                        st.metric("Модель", prediction.model_version)
                        
                    # Рекомендации
                    st.subheader("Рекомендуемые действия")
                    for i, rec in enumerate(prediction.recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                        
                    # Сохранение результата
                    with Session(engine) as session:
                        db_prediction = PredictionResult(**prediction.dict())
                        session.add(db_prediction)
                        session.commit()
                        
                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {str(e)}")
                    
    elif page == "Аналитика":
        st.title("Аналитическая панель")
        
        with Session(engine) as session:
            # Загрузка метрик моделей
            model_metrics = session.exec(select(ModelMetrics)).all()
            
            # Визуализация
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Распределение рисков")
                risks = session.exec(
                    select(PredictionResult.risk_level)
                ).all()
                risk_counts = pd.Series(risks).value_counts()
                st.bar_chart(risk_counts)
                
            with col2:
                st.subheader("Точность моделей")
                metrics_data = {m.model_version: {"roc_auc": m.metric_value} 
                               for m in model_metrics if m.metric_name == "roc_auc"}
                fig = plot_model_metrics(metrics_data)
                st.pyplot(fig)
                
    elif page == "Iris Dataset":
        # Оригинальный код для Iris Dataset
        st.header("Демонстрация Fisher's Iris датасета")
        df = load_data()

        # Визуализация
        fig, ax = plt.subplots()
        plt.scatter(df['sepal_length'], df['sepal_width'], c='blue', label='Iris-setosa')
        plt.scatter(df['sepal_length'], df['petal_width'], c='red', label='Iris-versicolor')
        plt.scatter(df['sepal_length'], df['petal_length'], c='green', label='Iris-virginica')

        plt.xlabel('sepal_length')
        plt.ylabel('sepal_width')
        plt.title('Iris Fisher Dataset')
        plt.legend()
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
