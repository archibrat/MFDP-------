"""
Сервис импорта и экспорта данных для интеграции с внешними системами
"""

import pandas as pd
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass
import asyncio
from io import StringIO, BytesIO
import csv

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    HL7 = "hl7"
    FHIR = "fhir"


class ImportStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ImportTask:
    task_id: str
    filename: str
    format: DataFormat
    status: ImportStatus
    created_at: datetime
    processed_records: int = 0
    total_records: int = 0
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


@dataclass
class ExportTask:
    task_id: str
    format: DataFormat
    query_params: Dict[str, Any]
    status: ImportStatus
    created_at: datetime
    filename: Optional[str] = None
    record_count: int = 0


class DataValidator:
    """Валидатор данных для проверки корректности импортируемых данных"""
    
    def __init__(self):
        self.required_fields = {
            "patient": ["patient_id", "gender", "age"],
            "appointment": ["appointment_id", "patient_id", "date", "time"],
            "event": ["event_id", "type", "date", "patient_id"]
        }
    
    def validate_patient_data(self, data: Dict) -> tuple[bool, List[str]]:
        """Валидация данных пациента"""
        errors = []
        required = self.required_fields["patient"]
        
        for field in required:
            if field not in data or data[field] is None:
                errors.append(f"Отсутствует обязательное поле: {field}")
        
        # Проверка типов данных
        if "age" in data and not isinstance(data["age"], (int, float)):
            errors.append("Возраст должен быть числом")
        
        if "gender" in data and data["gender"] not in ["M", "F", "male", "female"]:
            errors.append("Некорректное значение пола")
        
        return len(errors) == 0, errors
    
    def validate_appointment_data(self, data: Dict) -> tuple[bool, List[str]]:
        """Валидация данных записи на прием"""
        errors = []
        required = self.required_fields["appointment"]
        
        for field in required:
            if field not in data or data[field] is None:
                errors.append(f"Отсутствует обязательное поле: {field}")
        
        # Проверка формата даты
        if "date" in data:
            try:
                datetime.fromisoformat(str(data["date"]))
            except ValueError:
                errors.append("Некорректный формат даты")
        
        return len(errors) == 0, errors


class ImportExportService:
    """Основной сервис импорта и экспорта данных"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.active_tasks: Dict[str, Union[ImportTask, ExportTask]] = {}
        self.supported_formats = list(DataFormat)
        self.storage_path = Path("./data/imports")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def import_data(self, 
                         file_content: Union[str, bytes], 
                         filename: str,
                         format: DataFormat,
                         data_type: str = "patient") -> ImportTask:
        """Импорт данных из файла"""
        
        task_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = ImportTask(
            task_id=task_id,
            filename=filename,
            format=format,
            status=ImportStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        
        try:
            # Запуск обработки в фоновом режиме
            asyncio.create_task(self._process_import(task, file_content, data_type))
            
        except Exception as e:
            task.status = ImportStatus.FAILED
            task.error_messages.append(f"Ошибка запуска импорта: {str(e)}")
            logger.error(f"Import failed: {str(e)}")
        
        return task
    
    async def _process_import(self, task: ImportTask, file_content: Union[str, bytes], data_type: str):
        """Фоновая обработка импорта данных"""
        try:
            task.status = ImportStatus.PROCESSING
            
            # Парсинг данных в зависимости от формата
            if task.format == DataFormat.CSV:
                data = self._parse_csv(file_content)
            elif task.format == DataFormat.JSON:
                data = self._parse_json(file_content)
            elif task.format == DataFormat.XML:
                data = self._parse_xml(file_content)
            elif task.format == DataFormat.EXCEL:
                data = self._parse_excel(file_content)
            else:
                raise ValueError(f"Неподдерживаемый формат: {task.format}")
            
            task.total_records = len(data)
            
            # Валидация и обработка записей
            valid_records = []
            for i, record in enumerate(data):
                if data_type == "patient":
                    is_valid, errors = self.validator.validate_patient_data(record)
                elif data_type == "appointment":
                    is_valid, errors = self.validator.validate_appointment_data(record)
                else:
                    is_valid, errors = True, []
                
                if is_valid:
                    valid_records.append(record)
                    task.processed_records += 1
                else:
                    task.error_messages.extend([f"Запись {i+1}: {error}" for error in errors])
            
            # Сохранение валидных записей
            if valid_records:
                await self._save_imported_data(valid_records, data_type, task.task_id)
            
            task.status = ImportStatus.COMPLETED
            logger.info(f"Import completed: {task.processed_records}/{task.total_records} records")
            
        except Exception as e:
            task.status = ImportStatus.FAILED
            task.error_messages.append(f"Ошибка обработки: {str(e)}")
            logger.error(f"Import processing failed: {str(e)}")
    
    def _parse_csv(self, content: Union[str, bytes]) -> List[Dict]:
        """Парсинг CSV данных"""
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        csv_data = []
        csv_reader = csv.DictReader(StringIO(content))
        for row in csv_reader:
            csv_data.append(dict(row))
        
        return csv_data
    
    def _parse_json(self, content: Union[str, bytes]) -> List[Dict]:
        """Парсинг JSON данных"""
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        data = json.loads(content)
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            return [data]
    
    def _parse_xml(self, content: Union[str, bytes]) -> List[Dict]:
        """Парсинг XML данных"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        root = ET.fromstring(content)
        data = []
        
        for item in root:
            record = {}
            for child in item:
                record[child.tag] = child.text
            data.append(record)
        
        return data
    
    def _parse_excel(self, content: bytes) -> List[Dict]:
        """Парсинг Excel данных"""
        df = pd.read_excel(BytesIO(content))
        return df.to_dict('records')
    
    async def _save_imported_data(self, data: List[Dict], data_type: str, task_id: str):
        """Сохранение импортированных данных"""
        filename = self.storage_path / f"{data_type}_{task_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Saved {len(data)} records to {filename}")
    
    async def export_data(self, 
                         query_params: Dict[str, Any],
                         format: DataFormat,
                         data_type: str = "patient") -> ExportTask:
        """Экспорт данных в указанном формате"""
        
        task_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = ExportTask(
            task_id=task_id,
            format=format,
            query_params=query_params,
            status=ImportStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        
        try:
            # Запуск экспорта в фоновом режиме
            asyncio.create_task(self._process_export(task, data_type))
            
        except Exception as e:
            task.status = ImportStatus.FAILED
            logger.error(f"Export failed: {str(e)}")
        
        return task
    
    async def _process_export(self, task: ExportTask, data_type: str):
        """Фоновая обработка экспорта данных"""
        try:
            task.status = ImportStatus.PROCESSING
            
            # Получение данных для экспорта (заглушка)
            data = await self._fetch_export_data(task.query_params, data_type)
            task.record_count = len(data)
            
            # Генерация файла в нужном формате
            if task.format == DataFormat.CSV:
                filename = f"{data_type}_{task.task_id}.csv"
                content = self._generate_csv(data)
            elif task.format == DataFormat.JSON:
                filename = f"{data_type}_{task.task_id}.json"
                content = self._generate_json(data)
            elif task.format == DataFormat.XML:
                filename = f"{data_type}_{task.task_id}.xml"
                content = self._generate_xml(data)
            elif task.format == DataFormat.EXCEL:
                filename = f"{data_type}_{task.task_id}.xlsx"
                content = self._generate_excel(data)
            else:
                raise ValueError(f"Неподдерживаемый формат экспорта: {task.format}")
            
            # Сохранение файла
            file_path = self.storage_path / filename
            if task.format == DataFormat.EXCEL:
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            task.filename = filename
            task.status = ImportStatus.COMPLETED
            
            logger.info(f"Export completed: {task.record_count} records to {filename}")
            
        except Exception as e:
            task.status = ImportStatus.FAILED
            logger.error(f"Export processing failed: {str(e)}")
    
    async def _fetch_export_data(self, query_params: Dict[str, Any], data_type: str) -> List[Dict]:
        """Получение данных для экспорта"""
        # Заглушка для демонстрации
        demo_data = []
        
        if data_type == "patient":
            demo_data = [
                {"patient_id": f"P{i:04d}", "gender": "M" if i % 2 == 0 else "F", 
                 "age": 25 + (i % 50), "neighbourhood": f"District_{i % 10}"}
                for i in range(1, 101)
            ]
        elif data_type == "appointment":
            demo_data = [
                {"appointment_id": f"A{i:04d}", "patient_id": f"P{i:04d}", 
                 "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}", "time": f"{(i % 12) + 8:02d}:00"}
                for i in range(1, 101)
            ]
        
        # Применение фильтров из query_params
        if "limit" in query_params:
            demo_data = demo_data[:query_params["limit"]]
        
        return demo_data
    
    def _generate_csv(self, data: List[Dict]) -> str:
        """Генерация CSV контента"""
        if not data:
            return ""
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _generate_json(self, data: List[Dict]) -> str:
        """Генерация JSON контента"""
        return json.dumps({
            "data": data,
            "count": len(data),
            "exported_at": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2, default=str)
    
    def _generate_xml(self, data: List[Dict]) -> str:
        """Генерация XML контента"""
        root = ET.Element("export")
        root.set("count", str(len(data)))
        root.set("exported_at", datetime.now().isoformat())
        
        for item in data:
            record = ET.SubElement(root, "record")
            for key, value in item.items():
                field = ET.SubElement(record, key)
                field.text = str(value) if value is not None else ""
        
        return ET.tostring(root, encoding='unicode')
    
    def _generate_excel(self, data: List[Dict]) -> bytes:
        """Генерация Excel контента"""
        df = pd.DataFrame(data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Exported Data', index=False)
        
        return output.getvalue()
    
    def get_task_status(self, task_id: str) -> Optional[Union[ImportTask, ExportTask]]:
        """Получение статуса задачи"""
        return self.active_tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Union[ImportTask, ExportTask]]:
        """Получение всех задач"""
        return self.active_tasks.copy()
    
    def cleanup_completed_tasks(self, days_old: int = 7):
        """Очистка завершенных задач старше указанного количества дней"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        to_remove = []
        for task_id, task in self.active_tasks.items():
            if (task.status in [ImportStatus.COMPLETED, ImportStatus.FAILED] 
                and task.created_at < cutoff_date):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.active_tasks[task_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old tasks")


# Глобальный экземпляр сервиса
import_export_service = ImportExportService()


def get_import_export_service() -> ImportExportService:
    """Получение экземпляра сервиса импорта/экспорта"""
    return import_export_service 