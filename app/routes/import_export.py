"""
API endpoints для импорта и экспорта данных
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional
from datetime import datetime

from services.integration.import_export_service import (
    get_import_export_service,
    ImportExportService,
    DataFormat,
    ImportTask,
    ExportTask
)

import_export_router = APIRouter()


@import_export_router.post("/import/")
async def import_data_file(
    file: UploadFile = File(...),
    format: str = Form(...),
    data_type: str = Form("patient"),
    service: ImportExportService = Depends(get_import_export_service)
):
    """Импорт данных из файла"""
    
    try:
        # Проверка формата
        data_format = DataFormat(format.lower())
        
        # Чтение содержимого файла
        content = await file.read()
        
        # Запуск импорта
        task = await service.import_data(
            file_content=content,
            filename=file.filename,
            format=data_format,
            data_type=data_type
        )
        
        return {
            "status": "success",
            "task_id": task.task_id,
            "message": f"Импорт файла {file.filename} запущен"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Некорректный формат: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка импорта: {str(e)}")


@import_export_router.post("/export/")
async def export_data(
    format: str,
    data_type: str = "patient",
    query_params: Optional[Dict] = None,
    service: ImportExportService = Depends(get_import_export_service)
):
    """Экспорт данных в указанном формате"""
    
    try:
        data_format = DataFormat(format.lower())
        
        if query_params is None:
            query_params = {}
        
        task = await service.export_data(
            query_params=query_params,
            format=data_format,
            data_type=data_type
        )
        
        return {
            "status": "success",
            "task_id": task.task_id,
            "message": f"Экспорт данных в формате {format} запущен"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Некорректный формат: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@import_export_router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    service: ImportExportService = Depends(get_import_export_service)
):
    """Получение статуса задачи импорта/экспорта"""
    
    task = service.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "filename": getattr(task, 'filename', None),
        "format": task.format.value if hasattr(task, 'format') else None,
        "processed_records": getattr(task, 'processed_records', 0),
        "total_records": getattr(task, 'total_records', 0),
        "error_messages": getattr(task, 'error_messages', [])
    }


@import_export_router.get("/tasks/")
async def list_tasks(
    service: ImportExportService = Depends(get_import_export_service)
):
    """Получение списка всех задач"""
    
    tasks = service.get_all_tasks()
    
    task_list = []
    for task_id, task in tasks.items():
        task_info = {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "type": "import" if isinstance(task, ImportTask) else "export"
        }
        
        if isinstance(task, ImportTask):
            task_info.update({
                "filename": task.filename,
                "processed_records": task.processed_records,
                "total_records": task.total_records
            })
        else:  # ExportTask
            task_info.update({
                "filename": task.filename,
                "record_count": task.record_count
            })
        
        task_list.append(task_info)
    
    return {
        "tasks": task_list,
        "total_count": len(task_list)
    }


@import_export_router.get("/formats/")
async def get_supported_formats():
    """Получение списка поддерживаемых форматов"""
    
    formats = [
        {
            "code": format.value,
            "name": format.value.upper(),
            "description": f"Формат {format.value.upper()}"
        }
        for format in DataFormat
    ]
    
    return {
        "supported_formats": formats
    }


@import_export_router.post("/validate/")
async def validate_data(
    data: List[Dict],
    data_type: str = "patient",
    service: ImportExportService = Depends(get_import_export_service)
):
    """Валидация данных перед импортом"""
    
    validation_results = []
    
    for i, record in enumerate(data):
        if data_type == "patient":
            is_valid, errors = service.validator.validate_patient_data(record)
        elif data_type == "appointment":
            is_valid, errors = service.validator.validate_appointment_data(record)
        else:
            is_valid, errors = True, []
        
        validation_results.append({
            "record_index": i,
            "is_valid": is_valid,
            "errors": errors
        })
    
    total_records = len(data)
    valid_records = sum(1 for r in validation_results if r["is_valid"])
    
    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": total_records - valid_records,
        "validation_results": validation_results
    } 