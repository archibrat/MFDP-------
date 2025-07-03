"""
Сервис конструктора отчетов для создания настраиваемых отчетов
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import asyncio
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

logger = logging.getLogger(__name__)


class ReportType(Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    ANALYTICS = "analytics"
    DASHBOARD = "dashboard"
    COMPARISON = "comparison"


class ReportFormat(Enum):
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


class AggregationType(Enum):
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class ReportField:
    name: str
    display_name: str
    data_type: str  # "string", "number", "date", "boolean"
    aggregation: Optional[AggregationType] = None
    format: Optional[str] = None  # Формат отображения


@dataclass
class ReportFilter:
    field: str
    operator: str  # "equals", "contains", "greater_than", "less_than", "between"
    value: Union[str, int, float, List[Any]]


@dataclass
class ReportChart:
    type: ChartType
    title: str
    x_field: str
    y_field: str
    group_by: Optional[str] = None
    aggregation: Optional[AggregationType] = None


@dataclass
class ReportTemplate:
    id: str
    name: str
    description: str
    type: ReportType
    data_source: str
    fields: List[ReportField]
    filters: List[ReportFilter]
    charts: List[ReportChart]
    grouping: Optional[List[str]] = None
    sorting: Optional[List[Dict[str, str]]] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class ReportRequest:
    template_id: str
    parameters: Dict[str, Any]
    format: ReportFormat
    email_to: Optional[str] = None


@dataclass
class ReportResult:
    id: str
    template_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    file_path: Optional[str]
    error_message: Optional[str] = None


class ReportBuilderService:
    """Сервис конструктора отчетов"""
    
    def __init__(self):
        self.templates: Dict[str, ReportTemplate] = {}
        self.reports: Dict[str, ReportResult] = {}
        self.storage_path = Path("./data/reports")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализация стандартных шаблонов
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Инициализация стандартных шаблонов отчетов"""
        
        # Шаблон ежедневного отчета
        daily_report = ReportTemplate(
            id="daily_summary",
            name="Ежедневная сводка",
            description="Сводка показателей за день",
            type=ReportType.SUMMARY,
            data_source="appointments",
            fields=[
                ReportField("date", "Дата", "date"),
                ReportField("total_appointments", "Всего записей", "number", AggregationType.COUNT),
                ReportField("no_shows", "Неявки", "number", AggregationType.COUNT),
                ReportField("no_show_rate", "Процент неявок", "number", format="%.2f%%"),
                ReportField("accuracy", "Точность прогноза", "number", format="%.2f%%")
            ],
            filters=[
                ReportFilter("date", "between", ["2024-01-01", "2024-12-31"])
            ],
            charts=[
                ReportChart(ChartType.LINE, "Тренд неявок", "date", "no_show_rate"),
                ReportChart(ChartType.BAR, "Количество записей", "date", "total_appointments")
            ]
        )
        
        # Шаблон аналитики пациентов
        patient_analytics = ReportTemplate(
            id="patient_analytics",
            name="Аналитика пациентов",
            description="Детальная аналитика по пациентам",
            type=ReportType.ANALYTICS,
            data_source="patients",
            fields=[
                ReportField("age_group", "Возрастная группа", "string"),
                ReportField("gender", "Пол", "string"),
                ReportField("neighbourhood", "Район", "string"),
                ReportField("appointment_count", "Количество записей", "number", AggregationType.COUNT),
                ReportField("no_show_count", "Количество неявок", "number", AggregationType.COUNT),
                ReportField("risk_score", "Средний риск", "number", AggregationType.AVERAGE, "%.2f")
            ],
            filters=[],
            charts=[
                ReportChart(ChartType.PIE, "Распределение по полу", "gender", "appointment_count", aggregation=AggregationType.COUNT),
                ReportChart(ChartType.BAR, "Неявки по районам", "neighbourhood", "no_show_count", aggregation=AggregationType.COUNT)
            ],
            grouping=["age_group", "gender"]
        )
        
        # Шаблон производительности ML
        ml_performance = ReportTemplate(
            id="ml_performance",
            name="Производительность ML моделей",
            description="Отчет о производительности машинного обучения",
            type=ReportType.ANALYTICS,
            data_source="ml_predictions",
            fields=[
                ReportField("model_name", "Название модели", "string"),
                ReportField("accuracy", "Точность", "number", format="%.2f%%"),
                ReportField("precision", "Точность (Precision)", "number", format="%.2f%%"),
                ReportField("recall", "Полнота (Recall)", "number", format="%.2f%%"),
                ReportField("f1_score", "F1-мера", "number", format="%.2f%%"),
                ReportField("prediction_count", "Количество предсказаний", "number", AggregationType.COUNT)
            ],
            filters=[],
            charts=[
                ReportChart(ChartType.BAR, "Точность моделей", "model_name", "accuracy"),
                ReportChart(ChartType.LINE, "Тренд производительности", "date", "accuracy", "model_name")
            ]
        )
        
        self.templates["daily_summary"] = daily_report
        self.templates["patient_analytics"] = patient_analytics
        self.templates["ml_performance"] = ml_performance
    
    def create_template(self, template: ReportTemplate) -> str:
        """Создание нового шаблона отчета"""
        
        template.created_at = datetime.now()
        self.templates[template.id] = template
        
        # Сохранение в файл
        self._save_template_to_file(template)
        
        logger.info(f"Created report template: {template.name}")
        return template.id
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Получение шаблона по ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[ReportTemplate]:
        """Получение списка всех шаблонов"""
        return list(self.templates.values())
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Обновление шаблона отчета"""
        
        template = self.templates.get(template_id)
        if not template:
            return False
        
        # Обновление полей
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        # Сохранение
        self._save_template_to_file(template)
        
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """Удаление шаблона отчета"""
        
        if template_id in self.templates:
            del self.templates[template_id]
            
            # Удаление файла
            template_file = self.storage_path / f"template_{template_id}.json"
            if template_file.exists():
                template_file.unlink()
            
            return True
        
        return False
    
    async def generate_report(self, request: ReportRequest) -> str:
        """Генерация отчета"""
        
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        result = ReportResult(
            id=report_id,
            template_id=request.template_id,
            status="processing",
            created_at=datetime.now()
        )
        
        self.reports[report_id] = result
        
        try:
            # Запуск генерации в фоновом режиме
            asyncio.create_task(self._generate_report_async(result, request))
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"Report generation failed: {str(e)}")
        
        return report_id
    
    async def _generate_report_async(self, result: ReportResult, request: ReportRequest):
        """Асинхронная генерация отчета"""
        
        try:
            template = self.templates.get(request.template_id)
            if not template:
                raise ValueError(f"Template {request.template_id} not found")
            
            # Получение данных
            data = await self._fetch_report_data(template, request.parameters)
            
            # Обработка данных
            processed_data = self._process_report_data(data, template)
            
            # Создание графиков
            charts = await self._generate_charts(processed_data, template.charts)
            
            # Генерация отчета в нужном формате
            file_path = await self._generate_report_file(
                processed_data, 
                charts, 
                template, 
                request.format,
                result.id
            )
            
            result.file_path = str(file_path)
            result.status = "completed"
            result.completed_at = datetime.now()
            
            # Отправка по email, если указано
            if request.email_to:
                await self._send_report_by_email(request.email_to, file_path, template.name)
            
            logger.info(f"Report {result.id} generated successfully")
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"Report generation failed: {str(e)}")
    
    async def _fetch_report_data(self, template: ReportTemplate, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Получение данных для отчета"""
        
        # Симуляция получения данных из различных источников
        if template.data_source == "appointments":
            return await self._fetch_appointments_data(parameters)
        elif template.data_source == "patients":
            return await self._fetch_patients_data(parameters)
        elif template.data_source == "ml_predictions":
            return await self._fetch_ml_predictions_data(parameters)
        else:
            raise ValueError(f"Unknown data source: {template.data_source}")
    
    async def _fetch_appointments_data(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Получение данных о записях"""
        
        # Демонстрационные данные
        dates = pd.date_range(
            start=parameters.get("start_date", "2024-01-01"),
            end=parameters.get("end_date", "2024-12-31"),
            freq="D"
        )
        
        data = []
        for date in dates:
            appointments = np.random.poisson(50)  # Среднее количество записей
            no_shows = np.random.poisson(appointments * 0.15)  # 15% неявок
            accuracy = np.random.normal(89, 3)  # Точность прогноза
            
            data.append({
                "date": date,
                "total_appointments": appointments,
                "no_shows": no_shows,
                "no_show_rate": (no_shows / appointments) * 100 if appointments > 0 else 0,
                "accuracy": max(0, min(100, accuracy))
            })
        
        return pd.DataFrame(data)
    
    async def _fetch_patients_data(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Получение данных о пациентах"""
        
        import numpy as np
        
        # Демонстрационные данные
        n_patients = 1000
        
        ages = np.random.randint(18, 80, n_patients)
        age_groups = pd.cut(ages, bins=[0, 30, 50, 70, 100], labels=["18-30", "31-50", "51-70", "70+"])
        
        data = {
            "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
            "age": ages,
            "age_group": age_groups,
            "gender": np.random.choice(["M", "F"], n_patients),
            "neighbourhood": np.random.choice([f"District_{i}" for i in range(1, 11)], n_patients),
            "appointment_count": np.random.poisson(3, n_patients),
            "no_show_count": np.random.poisson(0.5, n_patients),
            "risk_score": np.random.beta(2, 5, n_patients) * 100
        }
        
        return pd.DataFrame(data)
    
    async def _fetch_ml_predictions_data(self, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Получение данных о ML предсказаниях"""
        
        import numpy as np
        
        models = ["baseline_model", "advanced_model", "ensemble_model"]
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        
        data = []
        for model in models:
            for date in dates:
                data.append({
                    "model_name": model,
                    "date": date,
                    "accuracy": np.random.normal(85 + models.index(model) * 3, 2),
                    "precision": np.random.normal(88 + models.index(model) * 2, 2),
                    "recall": np.random.normal(82 + models.index(model) * 2, 2),
                    "f1_score": np.random.normal(85 + models.index(model) * 2, 2),
                    "prediction_count": np.random.poisson(100)
                })
        
        return pd.DataFrame(data)
    
    def _process_report_data(self, data: pd.DataFrame, template: ReportTemplate) -> pd.DataFrame:
        """Обработка данных отчета"""
        
        # Применение фильтров
        filtered_data = self._apply_filters(data, template.filters)
        
        # Группировка данных
        if template.grouping:
            grouped_data = self._apply_grouping(filtered_data, template.grouping, template.fields)
        else:
            grouped_data = filtered_data
        
        # Сортировка
        if template.sorting:
            grouped_data = self._apply_sorting(grouped_data, template.sorting)
        
        return grouped_data
    
    def _apply_filters(self, data: pd.DataFrame, filters: List[ReportFilter]) -> pd.DataFrame:
        """Применение фильтров к данным"""
        
        filtered_data = data.copy()
        
        for filter_obj in filters:
            if filter_obj.field not in filtered_data.columns:
                continue
            
            if filter_obj.operator == "equals":
                filtered_data = filtered_data[filtered_data[filter_obj.field] == filter_obj.value]
            elif filter_obj.operator == "contains":
                filtered_data = filtered_data[filtered_data[filter_obj.field].str.contains(filter_obj.value, na=False)]
            elif filter_obj.operator == "greater_than":
                filtered_data = filtered_data[filtered_data[filter_obj.field] > filter_obj.value]
            elif filter_obj.operator == "less_than":
                filtered_data = filtered_data[filtered_data[filter_obj.field] < filter_obj.value]
            elif filter_obj.operator == "between" and isinstance(filter_obj.value, list) and len(filter_obj.value) == 2:
                filtered_data = filtered_data[
                    (filtered_data[filter_obj.field] >= filter_obj.value[0]) &
                    (filtered_data[filter_obj.field] <= filter_obj.value[1])
                ]
        
        return filtered_data
    
    def _apply_grouping(self, data: pd.DataFrame, grouping: List[str], fields: List[ReportField]) -> pd.DataFrame:
        """Применение группировки данных"""
        
        # Определение агрегаций
        agg_dict = {}
        for field in fields:
            if field.aggregation and field.name in data.columns:
                if field.aggregation == AggregationType.SUM:
                    agg_dict[field.name] = 'sum'
                elif field.aggregation == AggregationType.COUNT:
                    agg_dict[field.name] = 'count'
                elif field.aggregation == AggregationType.AVERAGE:
                    agg_dict[field.name] = 'mean'
                elif field.aggregation == AggregationType.MIN:
                    agg_dict[field.name] = 'min'
                elif field.aggregation == AggregationType.MAX:
                    agg_dict[field.name] = 'max'
                elif field.aggregation == AggregationType.MEDIAN:
                    agg_dict[field.name] = 'median'
        
        if agg_dict:
            grouped_data = data.groupby(grouping).agg(agg_dict).reset_index()
        else:
            grouped_data = data.groupby(grouping).size().reset_index(name='count')
        
        return grouped_data
    
    def _apply_sorting(self, data: pd.DataFrame, sorting: List[Dict[str, str]]) -> pd.DataFrame:
        """Применение сортировки данных"""
        
        for sort_config in sorting:
            field = sort_config.get("field")
            direction = sort_config.get("direction", "asc")
            
            if field in data.columns:
                ascending = direction.lower() == "asc"
                data = data.sort_values(by=field, ascending=ascending)
        
        return data
    
    async def _generate_charts(self, data: pd.DataFrame, charts: List[ReportChart]) -> List[str]:
        """Генерация графиков для отчета"""
        
        chart_files = []
        
        for i, chart in enumerate(charts):
            try:
                plt.figure(figsize=(10, 6))
                
                if chart.type == ChartType.BAR:
                    if chart.group_by:
                        grouped = data.groupby(chart.group_by)[chart.y_field].sum()
                        grouped.plot(kind='bar')
                    else:
                        data.plot(x=chart.x_field, y=chart.y_field, kind='bar')
                
                elif chart.type == ChartType.LINE:
                    if chart.group_by:
                        for group in data[chart.group_by].unique():
                            group_data = data[data[chart.group_by] == group]
                            plt.plot(group_data[chart.x_field], group_data[chart.y_field], label=group)
                        plt.legend()
                    else:
                        plt.plot(data[chart.x_field], data[chart.y_field])
                
                elif chart.type == ChartType.PIE:
                    values = data[chart.y_field]
                    labels = data[chart.x_field]
                    plt.pie(values, labels=labels, autopct='%1.1f%%')
                
                plt.title(chart.title)
                plt.tight_layout()
                
                # Сохранение графика
                chart_file = self.storage_path / f"chart_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                chart_files.append(str(chart_file))
                
            except Exception as e:
                logger.error(f"Failed to generate chart {chart.title}: {str(e)}")
                plt.close()
        
        return chart_files
    
    async def _generate_report_file(self, 
                                  data: pd.DataFrame, 
                                  charts: List[str], 
                                  template: ReportTemplate,
                                  format: ReportFormat,
                                  report_id: str) -> Path:
        """Генерация файла отчета"""
        
        if format == ReportFormat.CSV:
            return await self._generate_csv_report(data, template, report_id)
        elif format == ReportFormat.EXCEL:
            return await self._generate_excel_report(data, charts, template, report_id)
        elif format == ReportFormat.JSON:
            return await self._generate_json_report(data, template, report_id)
        elif format == ReportFormat.HTML:
            return await self._generate_html_report(data, charts, template, report_id)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    async def _generate_csv_report(self, data: pd.DataFrame, template: ReportTemplate, report_id: str) -> Path:
        """Генерация CSV отчета"""
        
        file_path = self.storage_path / f"{report_id}.csv"
        data.to_csv(file_path, index=False, encoding='utf-8')
        
        return file_path
    
    async def _generate_json_report(self, data: pd.DataFrame, template: ReportTemplate, report_id: str) -> Path:
        """Генерация JSON отчета"""
        
        file_path = self.storage_path / f"{report_id}.json"
        
        report_data = {
            "report_id": report_id,
            "template": template.name,
            "generated_at": datetime.now().isoformat(),
            "data": data.to_dict('records')
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        return file_path
    
    async def _generate_excel_report(self, 
                                   data: pd.DataFrame, 
                                   charts: List[str], 
                                   template: ReportTemplate, 
                                   report_id: str) -> Path:
        """Генерация Excel отчета"""
        
        file_path = self.storage_path / f"{report_id}.xlsx"
        
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            # Основные данные
            data.to_excel(writer, sheet_name='Data', index=False)
            
            # Информация о шаблоне
            template_info = pd.DataFrame([{
                "Report Name": template.name,
                "Description": template.description,
                "Generated At": datetime.now().isoformat(),
                "Template ID": template.id
            }])
            template_info.to_excel(writer, sheet_name='Info', index=False)
        
        return file_path
    
    async def _generate_html_report(self, 
                                  data: pd.DataFrame, 
                                  charts: List[str], 
                                  template: ReportTemplate, 
                                  report_id: str) -> Path:
        """Генерация HTML отчета"""
        
        file_path = self.storage_path / f"{report_id}.html"
        
        # HTML шаблон
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ template_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }
                .chart { margin: 20px 0; text-align: center; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ template_name }}</h1>
                <p>{{ description }}</p>
                <p>Создано: {{ generated_at }}</p>
            </div>
            
            {% for chart in charts %}
            <div class="chart">
                <img src="{{ chart }}" alt="Chart">
            </div>
            {% endfor %}
            
            <h2>Данные</h2>
            {{ data_table }}
        </body>
        </html>
        """)
        
        # Генерация HTML таблицы
        data_table = data.to_html(index=False, classes='data-table')
        
        html_content = html_template.render(
            template_name=template.name,
            description=template.description,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            charts=charts,
            data_table=data_table
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    def _save_template_to_file(self, template: ReportTemplate):
        """Сохранение шаблона в файл"""
        
        try:
            file_path = self.storage_path / f"template_{template.id}.json"
            
            template_dict = asdict(template)
            if template.created_at:
                template_dict['created_at'] = template.created_at.isoformat()
            
            # Преобразование Enum в строки
            template_dict['type'] = template.type.value
            
            for field in template_dict['fields']:
                if field['aggregation']:
                    field['aggregation'] = field['aggregation'].value
            
            for chart in template_dict['charts']:
                chart['type'] = chart['type'].value
                if chart['aggregation']:
                    chart['aggregation'] = chart['aggregation'].value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save template: {str(e)}")
    
    async def _send_report_by_email(self, email_to: str, file_path: Path, report_name: str):
        """Отправка отчета по email"""
        
        try:
            # Здесь должна быть интеграция с email сервисом
            logger.info(f"Report {report_name} would be sent to {email_to}")
            
        except Exception as e:
            logger.error(f"Failed to send report by email: {str(e)}")
    
    def get_report_status(self, report_id: str) -> Optional[ReportResult]:
        """Получение статуса генерации отчета"""
        return self.reports.get(report_id)
    
    def list_reports(self, limit: int = 100) -> List[ReportResult]:
        """Получение списка отчетов"""
        reports = list(self.reports.values())
        reports.sort(key=lambda x: x.created_at, reverse=True)
        return reports[:limit]
    
    def download_report(self, report_id: str) -> Optional[Path]:
        """Получение пути к файлу отчета для скачивания"""
        
        result = self.reports.get(report_id)
        if result and result.status == "completed" and result.file_path:
            file_path = Path(result.file_path)
            if file_path.exists():
                return file_path
        
        return None


# Глобальный экземпляр сервиса конструктора отчетов
report_builder_service = ReportBuilderService()


def get_report_builder_service() -> ReportBuilderService:
    """Получение экземпляра сервиса конструктора отчетов"""
    return report_builder_service 