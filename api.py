"""
MFDP API - Medical Flow Data Platform
Корневой файл для запуска приложения
"""

import os
import sys
import uvicorn
from pathlib import Path

# Добавляем папку app в путь Python
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.api import app

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["app/"],
        log_level="info"
    ) 