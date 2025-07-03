from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from app.models.base import BaseDBModel, TimestampMixin


class MedialogPatient(BaseDBModel, table=True):
    """Модель пациента МИС Медиалог (таблица PATIENTS)"""
    __tablename__ = "medialog_patients"
    
    patients_id: int = Field(unique=True, index=True, description="ID пациента в МИС")
    surname: str = Field(max_length=100, description="Фамилия")
    name: str = Field(max_length=100, description="Имя")
    middle_name: Optional[str] = Field(max_length=100, description="Отчество")
    birth_date: datetime = Field(description="Дата рождения")
    gender: str = Field(max_length=1, description="Пол (M/F)")
    phone: Optional[str] = Field(max_length=20, description="Телефон")
    email: Optional[str] = Field(max_length=100, description="Email")
    policy_number: Optional[str] = Field(max_length=50, description="Номер полиса")
    creation_date: datetime = Field(default_factory=datetime.utcnow, description="Дата создания записи")
    
    # Связи
    consultations: List["Consultation"] = Relationship(back_populates="patient")
    appointments: List["Appointment"] = Relationship(back_populates="patient")
    directions: List["Direction"] = Relationship(back_populates="patient")
    transfers: List["DataTransfer"] = Relationship(back_populates="patient")


class Medecin(BaseDBModel, table=True):
    """Модель врача МИС Медиалог (таблица MEDECINS)"""
    __tablename__ = "medecins"
    
    medecins_id: int = Field(unique=True, index=True, description="ID врача в МИС")
    surname: str = Field(max_length=100, description="Фамилия")
    name: str = Field(max_length=100, description="Имя")
    speciality: str = Field(max_length=100, description="Специальность")
    department: str = Field(max_length=100, description="Отделение")
    position: str = Field(max_length=100, description="Должность")
    work_schedule: Optional[str] = Field(max_length=200, description="Расписание работы")
    active_flag: bool = Field(default=True, description="Активность")
    
    # Связи
    consultations: List["Consultation"] = Relationship(back_populates="medecin")
    schedules: List["Schedule"] = Relationship(back_populates="medecin")
    appointments: List["Appointment"] = Relationship(back_populates="medecin")
    directions: List["Direction"] = Relationship(back_populates="medecin")


class Consultation(BaseDBModel, table=True):
    """Модель консультации МИС Медиалог (таблица MOTCONSU)"""
    __tablename__ = "motconsu"
    
    motconsu_id: int = Field(unique=True, index=True, description="ID консультации")
    patients_id: int = Field(foreign_key="medialog_patients.id", description="ID пациента")
    medecins_id: int = Field(foreign_key="medecins.id", description="ID врача")
    visit_date: datetime = Field(description="Дата посещения")
    diagnosis: Optional[str] = Field(description="Диагноз")
    treatment: Optional[str] = Field(description="Лечение")
    visit_type: str = Field(max_length=50, description="Тип посещения")
    duration: Optional[int] = Field(description="Длительность в минутах")
    
    # Связи
    patient: MedialogPatient = Relationship(back_populates="consultations")
    medecin: Medecin = Relationship(back_populates="consultations")


class Schedule(BaseDBModel, table=True):
    """Модель расписания врачей (таблица PL_SCHEDULE)"""
    __tablename__ = "pl_schedule"
    
    schedule_id: int = Field(unique=True, index=True, description="ID расписания")
    medecins_id: int = Field(foreign_key="medecins.id", description="ID врача")
    date: datetime = Field(description="Дата")
    time_start: datetime = Field(description="Время начала")
    time_end: datetime = Field(description="Время окончания")
    cabinet: str = Field(max_length=20, description="Кабинет")
    slots_total: int = Field(description="Общее количество слотов")
    slots_booked: int = Field(default=0, description="Забронированные слоты")
    status: str = Field(max_length=20, default="active", description="Статус")
    
    # Связи
    medecin: Medecin = Relationship(back_populates="schedules")
    appointments: List["Appointment"] = Relationship(back_populates="schedule")


class Appointment(BaseDBModel, table=True):
    """Модель записи на прием (таблица PL_APPOINTMENTS)"""
    __tablename__ = "pl_appointments"
    
    appointment_id: int = Field(unique=True, index=True, description="ID записи")
    patients_id: int = Field(foreign_key="medialog_patients.id", description="ID пациента")
    schedule_id: int = Field(foreign_key="pl_schedule.schedule_id", description="ID расписания")
    appointment_time: datetime = Field(description="Время записи")
    visit_type: str = Field(max_length=50, description="Тип посещения")
    status: str = Field(max_length=20, default="confirmed", description="Статус")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Дата создания")
    no_show_flag: bool = Field(default=False, description="Флаг неявки")
    cancellation_reason: Optional[str] = Field(description="Причина отмены")
    
    # Связи
    patient: MedialogPatient = Relationship(back_populates="appointments")
    schedule: Schedule = Relationship(back_populates="appointments")
    medecin: Medecin = Relationship(back_populates="appointments")
    predictions: List["NoShowPrediction"] = Relationship(back_populates="appointment")


class ScheduleModel(BaseDBModel, table=True):
    """Модель шаблона расписания (таблица PL_MODELS)"""
    __tablename__ = "pl_models"
    
    model_id: int = Field(unique=True, index=True, description="ID модели")
    model_name: str = Field(max_length=100, description="Название модели")
    template_data: str = Field(description="Данные шаблона (JSON)")
    department: str = Field(max_length=100, description="Отделение")
    active_flag: bool = Field(default=True, description="Активность")


class Direction(BaseDBModel, table=True):
    """Модель направления (таблица DIR_APPOINTMENTS)"""
    __tablename__ = "dir_appointments"
    
    direction_id: int = Field(unique=True, index=True, description="ID направления")
    patients_id: int = Field(foreign_key="medialog_patients.id", description="ID пациента")
    medecins_id: int = Field(foreign_key="medecins.id", description="ID врача")
    service_type: str = Field(max_length=100, description="Тип услуги")
    urgency: str = Field(max_length=20, description="Срочность")
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Дата создания")
    execution_date: Optional[datetime] = Field(description="Дата выполнения")
    
    # Связи
    patient: MedialogPatient = Relationship(back_populates="directions")
    medecin: Medecin = Relationship(back_populates="directions")


class DataTransfer(BaseDBModel, table=True):
    """Модель движения пациентов (таблица DATA_TRANSFERS)"""
    __tablename__ = "data_transfers"
    
    transfer_id: int = Field(unique=True, index=True, description="ID перевода")
    patients_id: int = Field(foreign_key="medialog_patients.id", description="ID пациента")
    from_department: str = Field(max_length=100, description="Отделение отправления")
    to_department: str = Field(max_length=100, description="Отделение назначения")
    transfer_date: datetime = Field(description="Дата перевода")
    reason: Optional[str] = Field(description="Причина перевода")
    
    # Связи
    patient: MedialogPatient = Relationship(back_populates="transfers")


class NoShowPrediction(BaseDBModel, table=True):
    """Модель прогноза неявки пациента"""
    __tablename__ = "no_show_predictions"
    
    appointment_id: int = Field(foreign_key="pl_appointments.appointment_id", description="ID записи")
    prediction_probability: float = Field(ge=0.0, le=1.0, description="Вероятность неявки")
    risk_level: str = Field(max_length=20, description="Уровень риска")
    recommendations: str = Field(description="Рекомендации")
    prediction_date: datetime = Field(default_factory=datetime.utcnow, description="Дата прогноза")
    model_version: str = Field(max_length=50, description="Версия модели")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Уровень уверенности")
    features_used: str = Field(description="Использованные признаки (JSON)")
    
    # Связи
    appointment: Appointment = Relationship(back_populates="predictions") 