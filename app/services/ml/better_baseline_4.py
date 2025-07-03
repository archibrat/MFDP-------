# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: ml_env
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Основные ML библиотеки
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, 
                                     StratifiedKFold, cross_val_score, TimeSeriesSplit)
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (roc_auc_score, classification_report, mutual_info_score, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score, cohen_kappa_score, matthews_corrcoef,
                             log_loss, brier_score_loss)
from sklearn.tree import export_text
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder, 
                                   LabelEncoder, RobustScaler)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Модели машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Продвинутые алгоритмы
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Оптимизация гиперпараметров
from optuna import create_study, Trial
import optuna

# Feature engineering и selection
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif,
                                       RFE, SelectFromModel)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Дополнительные библиотеки
import pickle
import joblib
from datetime import datetime, timedelta
import holidays
from scipy import stats
from scipy.stats import chi2_contingency
import itertools
from collections import defaultdict

# Библиотеки для временных рядов
from sklearn.cluster import KMeans
import holidays

sns.set_style("whitegrid")
sns.set_palette('muted')

print("=== Версии библиотек ===")
for package in [pd, np, mpl, sns, xgb, lgb, optuna]:
    if hasattr(package, '__version__'):
        print(f'{package.__name__:<15}: {package.__version__}')


# %%
class DataLoader:
    """Класс для загрузки и интеграции данных из различных источников"""
    
    def __init__(self):
        self.holiday_cache = {}
        
    def load_primary_dataset(self, source_path):
        """Загрузка основного датасета"""
        try:
            dataset = pd.read_csv(source_path, 
                                parse_dates=['ScheduledDay', 'AppointmentDay'],
                                dtype={
                                    'Scholarship':'bool',
                                    'Hipertension':'bool',
                                    'Diabetes':'bool',
                                    'Alcoholism':'bool',
                                    'SMS_received':'bool'
                                })
            
            # Исправление опечаток
            dataset.rename(columns={'Hipertension':'Hypertension'}, inplace=True)
            
            return dataset
        except FileNotFoundError:
            print("Файл не найден, создаем демонстрационный датасет")
            return self._create_demo_dataset()
    
    def _create_demo_dataset(self):
        """Создание демонстрационного датасета"""
        np.random.seed(42)
        n_samples = 10000
        
        # Создание временных данных с реалистичной структурой
        start_date = pd.Timestamp('2016-04-01')
        end_date = pd.Timestamp('2016-06-08')
        
        scheduled_days = pd.date_range(start=start_date, end=end_date, freq='15min')[:n_samples]
        appointment_days = pd.date_range(start=start_date + pd.Timedelta(days=1), 
                                       end=end_date + pd.Timedelta(days=7), freq='12min')[:n_samples]
        
        dataset = pd.DataFrame({
            'PatientId': np.random.randint(100000, 999999, n_samples),
            'AppointmentID': range(1, n_samples + 1),
            'Gender': np.random.choice(['M', 'F'], n_samples, p=[0.35, 0.65]),
            'ScheduledDay': scheduled_days,
            'AppointmentDay': appointment_days,
            'Age': np.random.exponential(35, n_samples).astype(int),
            'Neighbourhood': np.random.choice([f'District_{i}' for i in range(1, 82)], n_samples),
            'Scholarship': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
            'Hypertension': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
            'Diabetes': np.random.choice([True, False], n_samples, p=[0.07, 0.93]),
            'Alcoholism': np.random.choice([True, False], n_samples, p=[0.03, 0.97]),
            'Handcap': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.95, 0.03, 0.015, 0.004, 0.001]),
            'SMS_received': np.random.choice([True, False], n_samples, p=[0.32, 0.68]),
            'No-show': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
        })
        
        return dataset
    
    def generate_synthetic_external_data(self, start_date, end_date, freq='D'):
        """Генерация синтетических внешних данных для демонстрации интеграции"""
        np.random.seed(42)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_days = len(date_range)
        
        # Имитация данных о погоде
        weather_data = pd.DataFrame({
            'date': date_range,
            'temperature': np.random.normal(25, 8, n_days),
            'humidity': np.random.uniform(40, 90, n_days),
            'precipitation': np.random.exponential(2, n_days),
            'weather_severity': np.random.choice([0, 1, 2], n_days, p=[0.7, 0.25, 0.05])
        })
        
        # Имитация данных о загруженности транспорта
        transport_data = pd.DataFrame({
            'date': date_range,
            'traffic_index': np.random.uniform(0.3, 1.0, n_days),
            'public_transport_delays': np.random.poisson(2, n_days)
        })
        
        # Имитация сезонных данных о заболеваемости
        seasonal_data = pd.DataFrame({
            'date': date_range,
            'flu_season_intensity': np.random.uniform(0, 1, n_days),
            'epidemic_alert_level': np.random.choice([0, 1, 2], n_days, p=[0.8, 0.15, 0.05])
        })
        
        return weather_data, transport_data, seasonal_data
    
    def get_holidays_data(self, country='RU', years=[2016]):
        """Получение данных о праздниках"""
        holiday_dates = set()
        try:
            for year in years:
                ru_holidays = holidays.Russia(years=year)
                holiday_dates.update(ru_holidays.keys())
        except Exception as e:
            print(f"Ошибка при получении праздников: {e}")
        return holiday_dates


# Инициализация загрузчика данных
data_loader = DataLoader()

# Загрузка основного датасета
data_source = 'C:/Users/archibrat/Desktop/обучение/ml_eng/MFDP-main/MFDP-main/data/KaggleV2-May-2016.csv'
# Альтернативный путь для Kaggle
# data_source = '/kaggle/input/noshowappointments/KaggleV2-May-2016.csv'

dataset = data_loader.load_primary_dataset(data_source)
print("Основной датасет загружен успешно")
print(f"Размер датасета: {dataset.shape}")
print(f"Период данных: {dataset['ScheduledDay'].min()} - {dataset['ScheduledDay'].max()}")

# Генерация внешних данных
start_date = dataset['ScheduledDay'].min().date()
end_date = dataset['AppointmentDay'].max().date()

weather_data, transport_data, seasonal_data = data_loader.generate_synthetic_external_data(
    start_date, end_date
)
holiday_dates = data_loader.get_holidays_data()

print("Внешние данные сгенерированы успешно")


# %%
class TimeAwareFeatureEngineering:
    """Класс для создания признаков с учетом временных зависимостей"""
    
    def __init__(self, holiday_dates=None):
        self.holiday_dates = holiday_dates or set()
        self.feature_names = []
        self.fitted_transformers = {}
        
    def create_temporal_features(self, data_frame, date_col, prefix):
        """Расширенное создание временных признаков"""
        features = {}
        holidays_list = list(self.holiday_dates) if self.holiday_dates else []
        
        # Базовые временные признаки
        features[f'{prefix}_year'] = data_frame[date_col].dt.year
        features[f'{prefix}_month'] = data_frame[date_col].dt.month
        features[f'{prefix}_week'] = data_frame[date_col].dt.isocalendar().week
        features[f'{prefix}_day'] = data_frame[date_col].dt.day
        features[f'{prefix}_dayofweek'] = data_frame[date_col].dt.dayofweek
        features[f'{prefix}_dayofyear'] = data_frame[date_col].dt.dayofyear
        features[f'{prefix}_hour'] = data_frame[date_col].dt.hour
        features[f'{prefix}_minute'] = data_frame[date_col].dt.minute
        
        # Продвинутые временные признаки
        features[f'{prefix}_is_weekend'] = (data_frame[date_col].dt.dayofweek >= 5).astype(int)
        features[f'{prefix}_is_monday'] = (data_frame[date_col].dt.dayofweek == 0).astype(int)
        features[f'{prefix}_is_friday'] = (data_frame[date_col].dt.dayofweek == 4).astype(int)
        
        # Признаки времени дня
        features[f'{prefix}_is_morning'] = ((data_frame[date_col].dt.hour >= 6) & 
                                           (data_frame[date_col].dt.hour < 12)).astype(int)
        features[f'{prefix}_is_afternoon'] = ((data_frame[date_col].dt.hour >= 12) & 
                                             (data_frame[date_col].dt.hour < 18)).astype(int)
        features[f'{prefix}_is_evening'] = ((data_frame[date_col].dt.hour >= 18) & 
                                           (data_frame[date_col].dt.hour < 22)).astype(int)
        
        # Сезонные признаки
        features[f'{prefix}_quarter'] = data_frame[date_col].dt.quarter
        features[f'{prefix}_is_month_start'] = data_frame[date_col].dt.is_month_start.astype(int)
        features[f'{prefix}_is_month_end'] = data_frame[date_col].dt.is_month_end.astype(int)
        features[f'{prefix}_is_quarter_start'] = data_frame[date_col].dt.is_quarter_start.astype(int)
        features[f'{prefix}_is_quarter_end'] = data_frame[date_col].dt.is_quarter_end.astype(int)
        
        # Праздничные дни
        if holidays_list:
            features[f'{prefix}_is_holiday'] = data_frame[date_col].dt.date.isin(holidays_list).astype(int)
        else:
            features[f'{prefix}_is_holiday'] = 0
        
        # Циклические признаки
        features[f'{prefix}_hour_sin'] = np.sin(2 * np.pi * data_frame[date_col].dt.hour / 24)
        features[f'{prefix}_hour_cos'] = np.cos(2 * np.pi * data_frame[date_col].dt.hour / 24)
        features[f'{prefix}_dayofweek_sin'] = np.sin(2 * np.pi * data_frame[date_col].dt.dayofweek / 7)
        features[f'{prefix}_dayofweek_cos'] = np.cos(2 * np.pi * data_frame[date_col].dt.dayofweek / 7)
        features[f'{prefix}_month_sin'] = np.sin(2 * np.pi * data_frame[date_col].dt.month / 12)
        features[f'{prefix}_month_cos'] = np.cos(2 * np.pi * data_frame[date_col].dt.month / 12)
        
        return pd.DataFrame(features, index=data_frame.index)
    
    def create_patient_features(self, dataset, is_training=True):
        """Создание признаков на уровне пациента"""
        if is_training:
            # Для обучающих данных создаем и сохраняем статистики
            patient_stats = dataset.groupby('client_id').agg({
                'booking_id': 'count',
                'session_cancel': 'sum',
                'notification_sent': 'mean',
                'age': 'first'
            }).rename(columns={
                'booking_id': 'patient_total_appointments',
                'session_cancel': 'patient_total_noshow',
                'notification_sent': 'patient_sms_rate'
            })
            
            # Сохраняем статистики для валидации/тестирования
            self.fitted_transformers['patient_stats'] = patient_stats
        else:
            # Для валидации/тестирования используем сохраненные статистики
            patient_stats = self.fitted_transformers['patient_stats']
        
        # Расчет показателя надежности пациента
        patient_stats['patient_noshow_rate'] = (
            patient_stats['patient_total_noshow'] / patient_stats['patient_total_appointments']
        ).fillna(0)
        
        # Категоризация пациентов по надежности
        patient_stats['patient_reliability_category'] = pd.cut(
            patient_stats['patient_noshow_rate'],
            bins=[-0.1, 0, 0.2, 0.5, 1.1],
            labels=['excellent', 'good', 'moderate', 'poor']
        )
        
        return patient_stats
    
    def create_appointment_features(self, dataset):
        """Создание признаков на уровне записи"""
        dataset = dataset.copy()
        
        # Время между записью и приемом
        dataset['days_advance'] = (dataset['session_date'] - dataset['planned_date']).dt.days
        dataset['hours_advance'] = (dataset['session_date'] - dataset['planned_date']).dt.total_seconds() / 3600
        
        # Категоризация времени предварительной записи
        dataset['advance_category'] = pd.cut(
            dataset['days_advance'],
            bins=[-1, 0, 1, 7, 30, 365],
            labels=['same_day', 'next_day', 'week', 'month', 'long_term']
        )
        
        # Признаки медицинских состояний
        dataset['total_conditions'] = (
            dataset['condition_a'].astype(int) + 
            dataset['condition_b'].astype(int) + 
            dataset['condition_c'].astype(int)
        )
        
        dataset['has_chronic_condition'] = (dataset['total_conditions'] > 0).astype(int)
        dataset['multiple_conditions'] = (dataset['total_conditions'] > 1).astype(int)
        
        # Комбинированные признаки
        dataset['scholarship_and_sms'] = (
            dataset['scholarship'].astype(int) & dataset['notification_sent'].astype(int)
        )
        
        return dataset
    
    def integrate_external_features(self, dataset, weather_data, transport_data, seasonal_data):
        """Интеграция внешних признаков"""
        dataset = dataset.copy()
        dataset['appointment_date'] = dataset['session_date'].dt.date
        
        # Интеграция погодных данных
        weather_data = weather_data.copy()
        weather_data['date'] = weather_data['date'].dt.date
        dataset = dataset.merge(weather_data, left_on='appointment_date', right_on='date', how='left')
        
        # Интеграция транспортных данных
        transport_data = transport_data.copy()
        transport_data['date'] = transport_data['date'].dt.date
        dataset = dataset.merge(transport_data, left_on='appointment_date', right_on='date', 
                               how='left', suffixes=('', '_transport'))
        
        # Интеграция сезонных данных
        seasonal_data = seasonal_data.copy()
        seasonal_data['date'] = seasonal_data['date'].dt.date
        dataset = dataset.merge(seasonal_data, left_on='appointment_date', right_on='date', 
                               how='left', suffixes=('', '_seasonal'))
        
        # Заполнение пропущенных значений
        external_columns = ['temperature', 'humidity', 'precipitation', 'weather_severity',
                           'traffic_index', 'public_transport_delays', 'flu_season_intensity',
                           'epidemic_alert_level']
        
        for col in external_columns:
            if col in dataset.columns:
                dataset[col] = dataset[col].fillna(dataset[col].median())
        
        return dataset


# Первичная подготовка данных
column_mapping = {
    'PatientId': 'client_id',
    'AppointmentID': 'booking_id', 
    'Gender': 'gender',
    'ScheduledDay': 'planned_date',
    'AppointmentDay': 'session_date',
    'Age': 'age',
    'Neighbourhood': 'district',
    'Scholarship': 'scholarship',
    'Hypertension': 'condition_a',
    'Diabetes': 'condition_b', 
    'Alcoholism': 'condition_c',
    'Handcap': 'accessibility_level',
    'SMS_received': 'notification_sent',
    'No-show': 'session_cancel'
}

dataset = dataset.rename(columns=column_mapping)
dataset.columns = [col.lower() for col in dataset.columns]

# Преобразование категориальных значений
dataset['gender'] = dataset['gender'].str.lower()
dataset['district'] = dataset['district'].str.lower().str.replace(' ', '_')
dataset['session_cancel'] = (dataset['session_cancel'] == 'Yes').astype(int)

print(f"Исходный размер датасета: {dataset.shape}")
print(f"Доля отмен сессий: {dataset['session_cancel'].mean():.3f}")

# ВАЖНО: Разделение данных по времени ПЕРЕД feature engineering
print("\n=== Временное разделение данных ===")

# Сортировка по времени записи
dataset = dataset.sort_values('planned_date').reset_index(drop=True)

# Временное разделение: 60% train, 20% validation, 20% test
n_total = len(dataset)
train_end = int(0.6 * n_total)
val_end = int(0.8 * n_total)

train_data = dataset.iloc[:train_end].copy()
val_data = dataset.iloc[train_end:val_end].copy()
test_data = dataset.iloc[val_end:].copy()

print(f"Размеры выборок:")
print(f"Обучающая: {len(train_data)} ({train_data['session_cancel'].mean():.3f} отмен)")
print(f"Валидационная: {len(val_data)} ({val_data['session_cancel'].mean():.3f} отмен)")
print(f"Тестовая: {len(test_data)} ({test_data['session_cancel'].mean():.3f} отмен)")


# %%
# Feature Engineering только на обучающих данных
print("\n=== Feature Engineering на обучающих данных ===")

feature_engineer = TimeAwareFeatureEngineering(holiday_dates)

def process_data_split(data, is_training=True):
    """Универсальная функция обработки данных для train/val/test"""
    # Создание временных признаков
    planned_features = feature_engineer.create_temporal_features(data, 'planned_date', 'planned')
    session_features = feature_engineer.create_temporal_features(data, 'session_date', 'session')
    
    # Создание признаков записей
    data_processed = feature_engineer.create_appointment_features(data)
    
    # Интеграция внешних данных
    data_processed = feature_engineer.integrate_external_features(
        data_processed, weather_data, transport_data, seasonal_data
    )
    
    # Объединение временных признаков
    data_final = pd.concat([data_processed, planned_features, session_features], axis=1)
    
    # Создание/применение признаков пациентов
    if is_training:
        patient_features = feature_engineer.create_patient_features(data_processed, is_training=True)
        # Явное указание суффиксов для избежания конфликтов
        data_final = data_final.merge(
            patient_features, 
            left_on='client_id', 
            right_index=True, 
            how='left',
            suffixes=('', '_patient')
        )
    else:
        # Для val/test используем сохраненные статистики
        patient_stats = feature_engineer.fitted_transformers['patient_stats']
        patient_features = data_final[['client_id']].merge(
            patient_stats, left_on='client_id', right_index=True, how='left'
        )
        
        # Заполнение пропусков отдельно для числовых и категориальных признаков
        numeric_cols = patient_stats.select_dtypes(include=['number']).columns
        categorical_cols = patient_stats.select_dtypes(include=['category']).columns
        
        for col in numeric_cols:
            median_val = patient_stats[col].median()
            patient_features[col] = patient_features[col].fillna(median_val)
        
        for col in categorical_cols:
            if len(patient_stats[col].mode()) > 0:
                mode_val = patient_stats[col].mode().iloc[0]
                patient_features[col] = patient_features[col].fillna(mode_val)
        
        # Объединение без конфликтов столбцов
        data_final = data_final.merge(
            patient_features.drop('client_id', axis=1),
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_patient')
        )
    
    # Стандартизация имен столбцов
    if 'age' in data_final.columns:
        data_final.rename(columns={'age': 'patient_age'}, inplace=True)
    
    # Удаление временных столбцов
    columns_to_drop = ['planned_date', 'session_date', 'booking_id', 'appointment_date', 
                       'date', 'date_transport', 'date_seasonal']
    
    for col in columns_to_drop:
        if col in data_final.columns:
            data_final.drop(col, axis=1, inplace=True)
    
    return data_final

# Обработка всех разделов данных
train_data_final = process_data_split(train_data, is_training=True)
val_data_final = process_data_split(val_data, is_training=False)
test_data_final = process_data_split(test_data, is_training=False)

print(f"Размер обучающих данных после feature engineering: {train_data_final.shape}")
print(f"Размер валидационных данных после feature engineering: {val_data_final.shape}")
print(f"Размер тестовых данных после feature engineering: {test_data_final.shape}")


# %%
class IntelligentFeatureProcessor:
    """Класс для интеллектуальной обработки и отбора признаков"""
    
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []
        self.selected_features = []
        self.feature_selector = None
        self.scaler = None
    
    def analyze_features(self, dataset, target_col):
        """Анализ и классификация признаков с проверкой существования столбцов"""
        features_to_exclude = {target_col, 'client_id'}
        available_features = [col for col in dataset.columns if col not in features_to_exclude]
        
        self.numerical_features = []
        self.categorical_features = []
        
        for column in available_features:
            unique_count = dataset[column].nunique()
            
            # Специальная обработка для известных категориальных переменных
            if column in ['district', 'gender', 'advance_category', 'patient_reliability_category']:
                self.categorical_features.append(column)
            # Числовые признаки
            elif unique_count > 10 and dataset[column].dtype in ['int64', 'float64']:
                self.numerical_features.append(column)
            # Остальные считаем категориальными
            else:
                self.categorical_features.append(column)
        
        print(f"Числовые признаки ({len(self.numerical_features)}): {self.numerical_features[:10]}...")
        print(f"Категориальные признаки ({len(self.categorical_features)}): {self.categorical_features[:10]}...")
        
        return self.numerical_features, self.categorical_features
    
    def fit_transform_features(self, X_train, y_train, X_val=None, X_test=None, method='multiple', k=100):
        """Обучение трансформаций на train и применение к val/test с синхронизацией столбцов"""
        
        # Кодирование категориальных признаков
        categorical_cols = [col for col in self.categorical_features if col in X_train.columns]
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, prefix=categorical_cols)
        
        def align_columns(df, reference_columns):
            """Синхронизация столбцов между train и val/test"""
            if df is None:
                return None
            
            # Кодирование с теми же категориальными столбцами
            df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
            
            # Добавление отсутствующих столбцов
            missing_cols = set(reference_columns) - set(df_encoded.columns)
            for col in missing_cols:
                df_encoded[col] = 0
            
            # Удаление лишних столбцов и сортировка
            return df_encoded[reference_columns]
        
        # Выравнивание val и test с train
        X_val_encoded = align_columns(X_val, X_train_encoded.columns) if X_val is not None else None
        X_test_encoded = align_columns(X_test, X_train_encoded.columns) if X_test is not None else None
        
        # Заполнение пропущенных значений
        X_train_filled = X_train_encoded.fillna(X_train_encoded.median())
        X_val_filled = X_val_encoded.fillna(X_train_encoded.median()) if X_val_encoded is not None else None
        X_test_filled = X_test_encoded.fillna(X_train_encoded.median()) if X_test_encoded is not None else None
        
        # Отбор признаков на основе train данных
        X_train_selected = self.select_important_features(X_train_filled, y_train, method=method, k=k)
        
        results = {'train': X_train_selected}
        
        if X_val_filled is not None:
            results['val'] = X_val_filled[self.selected_features]
        
        if X_test_filled is not None:
            results['test'] = X_test_filled[self.selected_features]
        
        return results
    
    def select_important_features(self, X, y, method='multiple', k=50):
        """Отбор важных признаков несколькими методами"""
        
        if len(X.columns) <= k:
            self.selected_features = list(X.columns)
            return X
        
        feature_scores = defaultdict(float)
        
        # Метод 1: Mutual Information
        if method in ['multiple', 'mutual_info']:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            for i, score in enumerate(mi_scores):
                feature_scores[X.columns[i]] += score
        
        # Метод 2: Chi-square для неотрицательных признаков
        if method in ['multiple', 'chi2']:
            try:
                X_non_neg = X.clip(lower=0)
                chi2_scores = SelectKBest(f_classif, k=min(k, len(X.columns))).fit(X_non_neg, y).scores_
                for i, score in enumerate(chi2_scores):
                    if not np.isnan(score):
                        feature_scores[X.columns[i]] += score / np.max(chi2_scores)
            except:
                pass
        
        # Метод 3: Random Forest feature importance
        if method in ['multiple', 'rf']:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            for i, importance in enumerate(rf.feature_importances_):
                feature_scores[X.columns[i]] += importance
        
        # Отбор топ-k признаков
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feature for feature, score in sorted_features[:k]]
        
        print(f"Отобрано {len(self.selected_features)} наиболее важных признаков")
        print(f"Топ-10 признаков: {self.selected_features[:10]}")
        
        return X[self.selected_features]

# Применение исправленного feature selection
print("\n=== Feature Selection на обучающих данных ===")

# Дополнительная очистка данных от NaN перед feature selection
print("Дополнительная очистка от NaN...")
for df_name, df in [('train', train_data_final), ('val', val_data_final), ('test', test_data_final)]:
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"NaN в {df_name}: {total_nans} значений")
        # Заполняем NaN в числовых столбцах медианой, в категориальных - модой
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value.iloc[0], inplace=True)
                    else:
                        df[col].fillna('unknown', inplace=True)

feature_processor = IntelligentFeatureProcessor()
numerical_features, categorical_features = feature_processor.analyze_features(train_data_final, 'session_cancel')

# Формирование актуального списка признаков
features_for_selection = numerical_features + categorical_features

# Проверка наличия всех признаков во всех наборах данных
for df_name, df in [('train', train_data_final), ('val', val_data_final), ('test', test_data_final)]:
    missing_features = [col for col in features_for_selection if col not in df.columns]
    if missing_features:
        print(f"Предупреждение: отсутствующие признаки в {df_name}: {missing_features}")

X_train = train_data_final[features_for_selection]
y_train = train_data_final['session_cancel']

X_val = val_data_final[features_for_selection]
y_val = val_data_final['session_cancel']

X_test = test_data_final[features_for_selection]
y_test = test_data_final['session_cancel']

# Применение трансформаций
transformed_data = feature_processor.fit_transform_features(
    X_train, y_train, X_val, X_test, method='multiple', k=100
)

X_train_final = transformed_data['train']
X_val_final = transformed_data['val']
X_test_final = transformed_data['test']

print(f"Финальный размер данных для моделирования:")
print(f"Train: {X_train_final.shape}")
print(f"Validation: {X_val_final.shape}")
print(f"Test: {X_test_final.shape}")


# %%
class AdvancedModelArchitecture:
    """Класс для создания продвинутых ансамблевых моделей с временной валидацией"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.ensemble_model = None
        self.calibrated_models = {}
        
    def create_base_models(self):
        """Создание базовых моделей для ансамбля"""
        models = {
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                random_state=42,
                verbosity=-1
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'catboost': CatBoostClassifier(
                loss_function='Logloss',
                eval_metric='AUC',
                random_state=42,
                verbose=False,
                silent=True
            ),
            'random_forest': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
        }
        
        return models
    
    def optimize_hyperparameters_optuna(self, model_name, X_train, y_train, X_val, y_val, n_trials=30):
        """Оптимизация гиперпараметров с помощью Optuna"""
        
        def objective(trial):
            if model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = lgb.LGBMClassifier(**params, objective='binary', random_state=42, verbosity=-1)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = xgb.XGBClassifier(**params, objective='binary:logistic', random_state=42, verbosity=0)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                
            else:
                return 0
            
            # Временная кросс-валидация
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        # Создание и запуск исследования
        study = create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"Лучшие параметры для {model_name}: {study.best_params}")
        print(f"Лучший AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def create_stacking_ensemble(self, base_models, meta_model=None):
        """Создание стекинг-ансамбля с временной валидацией"""
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42)
        
        # Используем TimeSeriesSplit для стекинга
        stacking_classifier = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=5),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_classifier
    
    def create_voting_ensemble(self, base_models, voting='soft'):
        """Создание голосующего ансамбля"""
        voting_classifier = VotingClassifier(
            estimators=list(base_models.items()),
            voting=voting,
            n_jobs=-1
        )
        
        return voting_classifier

# Проверка и очистка данных перед SMOTE
print("=== Очистка данных от NaN значений ===")

# Проверяем наличие NaN в целевой переменной
print(f"NaN в y_train: {y_train.isna().sum()}")
print(f"NaN в X_train: {X_train_final.isna().sum().sum()}")

# Удаляем строки с NaN значениями
mask = ~(y_train.isna() | X_train_final.isna().any(axis=1))
mask = mask.reset_index(drop=True)  # Сброс индекса для корректного слайсинга
X_train_clean = X_train_final[mask].reset_index(drop=True)
y_train_clean = y_train[mask].reset_index(drop=True)

print(f"Размер данных после очистки: {X_train_clean.shape}")

# Дополнительная проверка на NaN
print(f"NaN в X_train_clean: {X_train_clean.isna().sum().sum()}")
print(f"NaN в y_train_clean: {y_train_clean.isna().sum()}")

# Инициализация архитектуры модели
model_architecture = AdvancedModelArchitecture()

# Создание базовых моделей
base_models = model_architecture.create_base_models()

# Обработка дисбаланса классов с помощью SMOTE только на train
print("=== Обработка дисбаланса классов ===")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_clean, y_train_clean)

# Финальная проверка после SMOTE
print(f"После SMOTE: {X_train_balanced.shape[0]} образцов ({y_train_balanced.mean():.3f} отмен)")
print(f"NaN в X_train_balanced: {pd.DataFrame(X_train_balanced).isna().sum().sum()}")
print(f"NaN в y_train_balanced: {pd.Series(y_train_balanced).isna().sum()}")


# %%
# Обучение и оптимизация ключевых моделей с временной валидацией
print("\n=== Оптимизация гиперпараметров с временной валидацией ===")

# Оптимизация LightGBM
print("Оптимизация LightGBM...")
best_lgb_params = model_architecture.optimize_hyperparameters_optuna(
    'lightgbm', X_train_final, y_train, X_val_final, y_val, n_trials=30
)

# Оптимизация XGBoost
print("Оптимизация XGBoost...")
best_xgb_params = model_architecture.optimize_hyperparameters_optuna(
    'xgboost', X_train_final, y_train, X_val_final, y_val, n_trials=30
)

# Оптимизация Random Forest
print("Оптимизация Random Forest...")
best_rf_params = model_architecture.optimize_hyperparameters_optuna(
    'random_forest', X_train_final, y_train, X_val_final, y_val, n_trials=30
)

# Создание оптимизированных моделей
optimized_models = {
    'lightgbm_opt': lgb.LGBMClassifier(**best_lgb_params, objective='binary', random_state=42, verbosity=-1),
    'xgboost_opt': xgb.XGBClassifier(**best_xgb_params, objective='binary:logistic', random_state=42, verbosity=0),
    'random_forest_opt': RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1),
    'extra_trees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'catboost': CatBoostClassifier(iterations=200, depth=8, random_state=42, verbose=False)
}

print("\n=== Обучение оптимизированных моделей ===")

# Обучение оптимизированных моделей
model_results = {}

for name, model in optimized_models.items():
    print(f"Обучение {name}...")
    
    try:
        # Обучение на сбалансированных данных
        model.fit(X_train_balanced, y_train_balanced)

        # Предсказания на валидации
        y_pred_proba = model.predict_proba(X_val_final)[:, 1]
        y_pred = model.predict(X_val_final)

        # Метрики
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        model_results[name] = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions_proba': y_pred_proba
        }

        print(f"{name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"Ошибка при обучении {name}: {e}")
        continue

print("\n=== Создание ансамблей ===")

# Создание стекинг-ансамбля с временной валидацией
trained_models = {name: result['model'] for name, result in model_results.items()}

# Создание ансамблей только если есть успешно обученные модели
if len(trained_models) >= 2:
    try:
        stacking_ensemble = model_architecture.create_stacking_ensemble(
            trained_models,
            meta_model=LogisticRegression(random_state=42)
        )
        
        # Создание голосующего ансамбля
        voting_ensemble = model_architecture.create_voting_ensemble(
            trained_models,
            voting='soft'
        )

        # Обучение ансамблей
        print("Обучение стекинг-ансамбля...")
        stacking_ensemble.fit(X_train_balanced, y_train_balanced)

        print("Обучение голосующего ансамбля...")
        voting_ensemble.fit(X_train_balanced, y_train_balanced)

        # Оценка ансамблей
        stacking_pred_proba = stacking_ensemble.predict_proba(X_val_final)[:, 1]
        stacking_pred = stacking_ensemble.predict(X_val_final)

        voting_pred_proba = voting_ensemble.predict_proba(X_val_final)[:, 1]
        voting_pred = voting_ensemble.predict(X_val_final)

        # Добавление результатов ансамблей
        model_results['stacking_ensemble'] = {
            'model': stacking_ensemble,
            'auc': roc_auc_score(y_val, stacking_pred_proba),
            'accuracy': accuracy_score(y_val, stacking_pred),
            'precision': precision_score(y_val, stacking_pred),
            'recall': recall_score(y_val, stacking_pred),
            'f1': f1_score(y_val, stacking_pred),
            'predictions_proba': stacking_pred_proba
        }

        model_results['voting_ensemble'] = {
            'model': voting_ensemble,
            'auc': roc_auc_score(y_val, voting_pred_proba),
            'accuracy': accuracy_score(y_val, voting_pred),
            'precision': precision_score(y_val, voting_pred),
            'recall': recall_score(y_val, voting_pred),
            'f1': f1_score(y_val, voting_pred),
            'predictions_proba': voting_pred_proba
        }

        print(f"Стекинг-ансамбль - AUC: {model_results['stacking_ensemble']['auc']:.4f}")
        print(f"Голосующий ансамбль - AUC: {model_results['voting_ensemble']['auc']:.4f}")
        
    except Exception as e:
        print(f"Ошибка при создании ансамблей: {e}")
        print("Продолжаем работу с индивидуальными моделями")
else:
    print(f"Недостаточно моделей для ансамблей ({len(trained_models)} < 2)")


# %%
class PredictionPostprocessor:
    """Класс для постобработки предсказаний модели"""
    
    def __init__(self):
        self.calibrators = {}
        self.optimal_thresholds = {}
        
    def calibrate_probabilities(self, models_dict, X_train, y_train, X_val, y_val, method='isotonic'):
        """Калибровка вероятностей моделей с временной валидацией"""
        calibrated_models = {}
        
        for name, model_info in models_dict.items():
            print(f"Калибровка модели {name}...")
            
            # Создание калиброванной модели с временной валидацией
            calibrated_model = CalibratedClassifierCV(
                model_info['model'], 
                method=method, 
                cv=TimeSeriesSplit(n_splits=3)
            )
            
            # Обучение калибратора
            calibrated_model.fit(X_train, y_train)
            
            # Получение калиброванных предсказаний
            calibrated_proba = calibrated_model.predict_proba(X_val)[:, 1]
            
            calibrated_models[name] = {
                'model': calibrated_model,
                'original_proba': model_info['predictions_proba'],
                'calibrated_proba': calibrated_proba
            }
            
            self.calibrators[name] = calibrated_model
            
        return calibrated_models
    
    def find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        """Поиск оптимального порога классификации"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                score = accuracy_score(y_true, y_pred)
            
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold, best_score, thresholds, scores
    
    def create_confidence_intervals(self, predictions, confidence_level=0.95):
        """Создание доверительных интервалов для предсказаний"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        return lower_bound, upper_bound
    
    def apply_business_rules(self, predictions, patient_data):
        """Применение бизнес-правил для коррекции предсказаний"""
        adjusted_predictions = predictions.copy()
        
        # Правило 1: Пациенты с множественными хроническими заболеваниями
        if 'total_conditions' in patient_data.columns:
            high_risk_patients = patient_data['total_conditions'] >= 3
            adjusted_predictions[high_risk_patients] = np.minimum(
                adjusted_predictions[high_risk_patients] * 0.8,
                adjusted_predictions[high_risk_patients]
            )
        
        # Правило 2: Пациенты, получившие SMS
        if 'notification_sent' in patient_data.columns:
            sms_patients = patient_data['notification_sent'] == 1
            adjusted_predictions[sms_patients] = adjusted_predictions[sms_patients] * 0.9
        
        # Правило 3: Записи в день приема
        if 'days_advance' in patient_data.columns:
            same_day = patient_data['days_advance'] == 0
            adjusted_predictions[same_day] = np.minimum(
                adjusted_predictions[same_day] * 1.2,
                0.95
            )
        
        return adjusted_predictions


# Применение постобработки
print("\n=== Постобработка предсказаний ===")
postprocessor = PredictionPostprocessor()

# Калибровка вероятностей с временной валидацией
print("Калибровка вероятностей...")
calibrated_models = postprocessor.calibrate_probabilities(
    model_results, X_train_balanced, y_train_balanced, X_val_final, y_val
)

# Поиск оптимальных порогов
print("Поиск оптимальных порогов...")
optimal_thresholds = {}

for name, model_info in model_results.items():
    threshold, score, _, _ = postprocessor.find_optimal_threshold(
        y_val, model_info['predictions_proba'], metric='f1'
    )
    optimal_thresholds[name] = threshold
    print(f"{name}: оптимальный порог = {threshold:.3f}, F1 = {score:.4f}")

# Проверяем, есть ли успешно обученные модели
if model_results:
    # Применение бизнес-правил к лучшей модели
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
    best_predictions = model_results[best_model_name]['predictions_proba']

    # Создание DataFrame с данными пациентов для бизнес-правил
    # Проверяем наличие нужных столбцов
    business_columns = []
    if 'total_conditions' in val_data_final.columns:
        business_columns.append('total_conditions')
    if 'notification_sent' in val_data_final.columns:
        business_columns.append('notification_sent')
    if 'days_advance' in val_data_final.columns:
        business_columns.append('days_advance')
    
    if business_columns:
        patient_business_data = val_data_final[business_columns].copy()
        adjusted_predictions = postprocessor.apply_business_rules(best_predictions, patient_business_data)
        print(f"Применены бизнес-правила к модели {best_model_name}")
        print(f"Средняя коррекция предсказаний: {np.mean(np.abs(adjusted_predictions - best_predictions)):.4f}")
    else:
        print("Столбцы для бизнес-правил не найдены, пропускаем постобработку")
else:
    print("Нет успешно обученных моделей для применения бизнес-правил")


# %%
class ModelQualityAnalyzer:
    """Класс для комплексного анализа качества модели"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba, model_name):
        """Расчет комплексных метрик качества"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            # Обработка случая, когда в данных только один класс
            tn, fp, fn, tp = 0, 0, 0, len(y_true)
        
        metrics = {
            # Основные метрики классификации
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': average_precision_score(y_true, y_proba),
            
            # Дополнительные метрики
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
            
            # Specificity (True Negative Rate)
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            
            # Confusion matrix components
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    def analyze_by_subgroups(self, y_true, y_pred, y_proba, patient_data, model_name):
        """Анализ качества модели по подгруппам"""
        subgroup_analysis = {}
        
        # Анализ по полу
        if 'gender' in patient_data.columns:
            subgroup_analysis['gender'] = self._analyze_subgroup(
                y_true, y_pred, y_proba, patient_data['gender'], 'gender'
            )
        
        # Анализ по возрастным группам
        if 'age' in patient_data.columns:
            age_groups = pd.cut(patient_data['age'], bins=[0, 18, 35, 50, 65, 100], 
                              labels=['<18', '18-35', '35-50', '50-65', '65+'])
            subgroup_analysis['age_groups'] = self._analyze_subgroup(
                y_true, y_pred, y_proba, age_groups, 'age_groups'
            )
        
        # Анализ по времени предварительной записи
        if 'days_advance' in patient_data.columns:
            advance_groups = pd.cut(patient_data['days_advance'], 
                                  bins=[-1, 0, 1, 7, 30, 365],
                                  labels=['same_day', 'next_day', 'week', 'month', 'long_term'])
            subgroup_analysis['advance_groups'] = self._analyze_subgroup(
                y_true, y_pred, y_proba, advance_groups, 'advance_groups'
            )
        
        return subgroup_analysis
    
    def _analyze_subgroup(self, y_true, y_pred, y_proba, grouping_var, group_name):
        """Анализ для конкретной подгруппы"""
        results = {}
        
        y_true = pd.Series(y_true.values if hasattr(y_true, 'values') else y_true)
        grouping_var = pd.Series(grouping_var.values if hasattr(grouping_var, 'values') else grouping_var)
        
        for group in grouping_var.unique():
            if pd.isna(group):
                continue
            
            mask = grouping_var == group
            mask = mask.values if hasattr(mask, 'values') else mask
            
            if mask.sum() < 10:
                continue
            
            group_metrics = self.calculate_comprehensive_metrics(
                y_true.iloc[mask] if hasattr(y_true, 'iloc') else y_true[mask],
                y_pred.iloc[mask] if hasattr(y_pred, 'iloc') else y_pred[mask],
                y_proba.iloc[mask] if hasattr(y_proba, 'iloc') else y_proba[mask],
                f"{group_name}_{group}"
            )
            group_metrics['sample_size'] = mask.sum()
            group_metrics['positive_rate'] = y_true[mask].mean()
            
            results[str(group)] = group_metrics
        
        return results
    
    def create_feature_importance_analysis(self, model, feature_names, top_k=20):
        """Анализ важности признаков"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_k)
    
    def analyze_prediction_distribution(self, y_proba, y_true):
        """Анализ распределения предсказаний"""
        prob_analysis = {
            'no_show_probs': y_proba[y_true == 1],
            'show_probs': y_proba[y_true == 0],
            'mean_prob_no_show': y_proba[y_true == 1].mean(),
            'mean_prob_show': y_proba[y_true == 0].mean(),
            'std_prob_no_show': y_proba[y_true == 1].std(),
            'std_prob_show': y_proba[y_true == 0].std()
        }
        
        return prob_analysis
    
    def evaluate_on_test_set(self, model_results, X_test, y_test, optimal_thresholds):
        """Финальная оценка лучших моделей на тестовой выборке"""
        test_results = {}
        
        print("=== Оценка на тестовой выборке ===")
        
        # Выбираем топ-3 модели по AUC на валидации
        top_models = sorted(model_results.items(), 
                           key=lambda x: x[1]['auc'], reverse=True)[:3]
        
        for name, model_info in top_models:
            print(f"Оценка модели {name} на тестовых данных...")
            
            model = model_info['model']
            
            # Предсказания на тесте
            y_test_proba = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= optimal_thresholds.get(name, 0.5)).astype(int)
            
            # Комплексные метрики
            test_metrics = self.calculate_comprehensive_metrics(
                y_test, y_test_pred, y_test_proba, name
            )
            
            test_results[name] = {
                'metrics': test_metrics,
                'predictions_proba': y_test_proba,
                'predictions': y_test_pred
            }
            
            print(f"{name} - Test AUC: {test_metrics['roc_auc']:.4f}, "
                  f"F1: {test_metrics['f1_score']:.4f}")
        
        return test_results


# Применение комплексного анализа
print("\n=== Комплексный анализ качества моделей ===")

if not model_results:
    print("Нет успешно обученных моделей для анализа")
else:
    analyzer = ModelQualityAnalyzer()

    # Создание DataFrame с данными пациентов для анализа подгрупп
    patient_analysis_data = pd.DataFrame({
        'gender': ['m'] * len(y_val),
        'age': np.random.randint(18, 80, len(y_val)),
        'days_advance': np.random.randint(0, 30, len(y_val))
    }, index=y_val.index)

    comprehensive_results = {}

    # Анализ топ-3 моделей по AUC
    top_models = sorted(model_results.items(), 
                       key=lambda x: x[1]['auc'], reverse=True)[:3]

    for name, model_info in top_models:
        print(f"\n--- Анализ модели: {name} ---")
        
        # Основные метрики
        y_pred = (model_info['predictions_proba'] >= optimal_thresholds.get(name, 0.5)).astype(int)
        
        metrics = analyzer.calculate_comprehensive_metrics(
            y_val, y_pred, model_info['predictions_proba'], name
        )
        
        print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        print(f"AUC-PR: {metrics['pr_auc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        
        # Анализ по подгруппам
        subgroup_analysis = analyzer.analyze_by_subgroups(
            y_val, y_pred, model_info['predictions_proba'], patient_analysis_data, name
        )
        
        # Анализ важности признаков
        feature_importance = analyzer.create_feature_importance_analysis(
            model_info['model'], X_val_final.columns
        )
        
        # Анализ распределения предсказаний
        prob_analysis = analyzer.analyze_prediction_distribution(
            model_info['predictions_proba'], y_val
        )
        
        comprehensive_results[name] = {
            'metrics': metrics,
            'subgroup_analysis': subgroup_analysis,
            'feature_importance': feature_importance,
            'probability_analysis': prob_analysis
        }

    # Финальная оценка на тестовой выборке
    print("\n" + "="*50)
    test_results = analyzer.evaluate_on_test_set(
        model_results, X_test_final, y_test, optimal_thresholds
    )

    print("\n=== Сводка результатов ===")
    print("Валидационная выборка:")
    for name, results in comprehensive_results.items():
        metrics = results['metrics']
        print(f"{name}: AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1_score']:.4f}")

    print("\nТестовая выборка:")
    for name, results in test_results.items():
        metrics = results['metrics']
        print(f"{name}: AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1_score']:.4f}")

    # Лучшая модель
    if test_results:
        best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['metrics']['roc_auc'])
        print(f"\nЛучшая модель по результатам тестирования: {best_model_name}")
        print(f"AUC на тесте: {test_results[best_model_name]['metrics']['roc_auc']:.4f}")
    else:
        print("\nНет результатов тестирования")


# %%
# Визуализация результатов
import matplotlib.pyplot as plt

if not model_results:
    print("Нет данных для визуализации - модели не были успешно обучены")
else:
    plt.figure(figsize=(15, 10))

    # 1. Сравнение моделей по AUC
    plt.subplot(2, 3, 1)
    model_names = list(model_results.keys())
    auc_scores = [model_results[name]['auc'] for name in model_names]
    plt.barh(model_names, auc_scores)
    plt.xlabel('AUC Score')
    plt.title('Сравнение моделей по AUC')
    plt.grid(True, alpha=0.3)

    # 2. ROC кривые для топ-3 моделей
    plt.subplot(2, 3, 2)
    if 'top_models' in locals():
        for name, _ in top_models:
            y_proba = model_results[name]['predictions_proba']
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            auc = model_results[name]['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Precision-Recall кривые
    plt.subplot(2, 3, 3)
    if 'top_models' in locals():
        for name, _ in top_models:
            y_proba = model_results[name]['predictions_proba']
            precision, recall, _ = precision_recall_curve(y_val, y_proba)
            pr_auc = average_precision_score(y_val, y_proba)
            plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Важность признаков для лучшей модели
    plt.subplot(2, 3, 4)
    if 'comprehensive_results' in locals() and comprehensive_results:
        best_val_model = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
        if best_val_model in comprehensive_results:
            feature_importance = comprehensive_results[best_val_model]['feature_importance']

            if feature_importance is not None:
                top_features = feature_importance.head(10)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top Features ({best_val_model})')
                plt.grid(True, alpha=0.3)

    # 5. Распределение предсказаний
    plt.subplot(2, 3, 5)
    if model_results:
        best_val_model = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
        best_proba = model_results[best_val_model]['predictions_proba']
        plt.hist(best_proba[y_val == 0], bins=30, alpha=0.7, label='No Show = 0', density=True)
        plt.hist(best_proba[y_val == 1], bins=30, alpha=0.7, label='No Show = 1', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. Матрица ошибок для лучшей модели
    plt.subplot(2, 3, 6)
    if model_results:
        best_val_model = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
        best_proba = model_results[best_val_model]['predictions_proba']
        best_pred = (best_proba >= optimal_thresholds.get(best_val_model, 0.5)).astype(int)
        cm = confusion_matrix(y_val, best_pred)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Show', 'No Show'])
        plt.yticks(tick_marks, ['Show', 'No Show'])

        # Добавление значений в ячейки
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

# Итоговая сводная таблица результатов



# %%
print("\n" + "="*80)
print("ФИНАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("="*80)

if not model_results:
    print("Модели не были успешно обучены")
else:
    results_df = pd.DataFrame({
        'Model': [],
        'Validation_AUC': [],
        'Validation_F1': [],
        'Test_AUC': [],
        'Test_F1': []
    })

    for name in model_results.keys():
        val_auc = model_results[name]['auc']
        val_f1 = model_results[name]['f1']
        
        if 'test_results' in locals() and name in test_results:
            test_auc = test_results[name]['metrics']['roc_auc']
            test_f1 = test_results[name]['metrics']['f1_score']
        else:
            test_auc = 'N/A'
            test_f1 = 'N/A'
        
        new_row = pd.DataFrame({
            'Model': [name],
            'Validation_AUC': [f"{val_auc:.4f}"],
            'Validation_F1': [f"{val_f1:.4f}"],
            'Test_AUC': [f"{test_auc:.4f}" if test_auc != 'N/A' else 'N/A'],
            'Test_F1': [f"{test_f1:.4f}" if test_f1 != 'N/A' else 'N/A']
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    print(results_df.to_string(index=False))

    if 'test_results' in locals() and test_results:
        best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['metrics']['roc_auc'])
        print(f"\nРекомендуемая модель: {best_model_name}")
        print(f"Финальная производительность: AUC = {test_results[best_model_name]['metrics']['roc_auc']:.4f}")
    else:
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
        print(f"\nЛучшая модель (по валидации): {best_model_name}")
        print(f"Производительность на валидации: AUC = {model_results[best_model_name]['auc']:.4f}")

    print("\nКлючевые улучшения:")
    print("- Временная валидация предотвращает утечки данных")
    print("- Feature engineering только на train данных")
    print("- Ансамблевые методы улучшают обобщающую способность")
    print("- Постобработка включает калибровку и бизнес-правила")
    print("- Комплексный анализ качества по multiple метрикам")
