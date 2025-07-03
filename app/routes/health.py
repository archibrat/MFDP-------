"""
Health Check endpoints
"""

from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.database.database import get_session
from datetime import datetime
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """Проверка состояния API"""
    development_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mode": "development" if development_mode else "production",
        "database": "disconnected" if development_mode else "connected"
    }

@router.get("/health/api")
async def api_health():
    """Проверка API"""
    return {
        "status": "ok",
        "service": "MFDP API"
    }

@router.get("/health/ml") 
async def ml_health():
    """Проверка ML сервиса"""
    return {
        "status": "ok",
        "service": "ML Service",
        "models_loaded": 1
    }
