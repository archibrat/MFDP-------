"""
API endpoints для конструктора отчетов
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from typing import Dict, List, Optional
from pydantic import BaseModel

from services.report_builder_service import (
    get_report_builder_service,
    ReportBuilderService,
    ReportTemplate,
    ReportField,
    ReportFilter,
    ReportChart,
    ReportRequest,
    ReportType,
    ReportFormat,
    ChartType,
    AggregationType
)

reports_router = APIRouter()


class CreateReportTemplateRequest(BaseModel):
    name: str
    description: str
    type: str
    data_source: str
    fields: List[Dict]
    filters: List[Dict] = []
    charts: List[Dict] = []
    grouping: Optional[List[str]] = None
    sorting: Optional[List[Dict]] = None


class GenerateReportRequest(BaseModel):
    template_id: str
    parameters: Dict = {}
    format: str = "json"
    email_to: Optional[str] = None


@reports_router.get("/templates/")
async def list_report_templates(
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Получение списка шаблонов отчетов"""
    
    templates = service.list_templates()
    
    templates_data = []
    for template in templates:
        templates_data.append({
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "type": template.type.value,
            "data_source": template.data_source,
            "field_count": len(template.fields),
            "chart_count": len(template.charts),
            "created_by": template.created_by,
            "created_at": template.created_at.isoformat() if template.created_at else None
        })
    
    return {
        "templates": templates_data,
        "total_count": len(templates_data)
    }


@reports_router.get("/templates/{template_id}")
async def get_report_template(
    template_id: str,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Получение шаблона отчета по ID"""
    
    template = service.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    
    # Сериализация полей
    fields_data = []
    for field in template.fields:
        field_data = {
            "name": field.name,
            "display_name": field.display_name,
            "data_type": field.data_type,
            "format": field.format
        }
        if field.aggregation:
            field_data["aggregation"] = field.aggregation.value
        fields_data.append(field_data)
    
    # Сериализация фильтров
    filters_data = []
    for filter_obj in template.filters:
        filters_data.append({
            "field": filter_obj.field,
            "operator": filter_obj.operator,
            "value": filter_obj.value
        })
    
    # Сериализация графиков
    charts_data = []
    for chart in template.charts:
        chart_data = {
            "type": chart.type.value,
            "title": chart.title,
            "x_field": chart.x_field,
            "y_field": chart.y_field,
            "group_by": chart.group_by
        }
        if chart.aggregation:
            chart_data["aggregation"] = chart.aggregation.value
        charts_data.append(chart_data)
    
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "type": template.type.value,
        "data_source": template.data_source,
        "fields": fields_data,
        "filters": filters_data,
        "charts": charts_data,
        "grouping": template.grouping,
        "sorting": template.sorting,
        "created_by": template.created_by,
        "created_at": template.created_at.isoformat() if template.created_at else None
    }


@reports_router.post("/templates/")
async def create_report_template(
    request: CreateReportTemplateRequest,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Создание нового шаблона отчета"""
    
    try:
        # Парсинг типа отчета
        report_type = ReportType(request.type.lower())
        
        # Создание полей
        fields = []
        for field_data in request.fields:
            aggregation = None
            if field_data.get("aggregation"):
                aggregation = AggregationType(field_data["aggregation"])
            
            field = ReportField(
                name=field_data["name"],
                display_name=field_data["display_name"],
                data_type=field_data["data_type"],
                aggregation=aggregation,
                format=field_data.get("format")
            )
            fields.append(field)
        
        # Создание фильтров
        filters = []
        for filter_data in request.filters:
            filter_obj = ReportFilter(
                field=filter_data["field"],
                operator=filter_data["operator"],
                value=filter_data["value"]
            )
            filters.append(filter_obj)
        
        # Создание графиков
        charts = []
        for chart_data in request.charts:
            chart_type = ChartType(chart_data["type"])
            aggregation = None
            if chart_data.get("aggregation"):
                aggregation = AggregationType(chart_data["aggregation"])
            
            chart = ReportChart(
                type=chart_type,
                title=chart_data["title"],
                x_field=chart_data["x_field"],
                y_field=chart_data["y_field"],
                group_by=chart_data.get("group_by"),
                aggregation=aggregation
            )
            charts.append(chart)
        
        # Создание шаблона
        template = ReportTemplate(
            id=f"template_{len(service.templates) + 1}",
            name=request.name,
            description=request.description,
            type=report_type,
            data_source=request.data_source,
            fields=fields,
            filters=filters,
            charts=charts,
            grouping=request.grouping,
            sorting=request.sorting
        )
        
        template_id = service.create_template(template)
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": f"Шаблон '{request.name}' создан успешно"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Некорректные параметры: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка создания шаблона: {str(e)}")


@reports_router.put("/templates/{template_id}")
async def update_report_template(
    template_id: str,
    updates: Dict,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Обновление шаблона отчета"""
    
    success = service.update_template(template_id, updates)
    
    if success:
        return {
            "status": "success",
            "message": "Шаблон обновлен успешно"
        }
    else:
        raise HTTPException(status_code=404, detail="Шаблон не найден")


@reports_router.delete("/templates/{template_id}")
async def delete_report_template(
    template_id: str,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Удаление шаблона отчета"""
    
    success = service.delete_template(template_id)
    
    if success:
        return {
            "status": "success",
            "message": "Шаблон удален успешно"
        }
    else:
        raise HTTPException(status_code=404, detail="Шаблон не найден")


@reports_router.post("/generate/")
async def generate_report(
    request: GenerateReportRequest,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Генерация отчета"""
    
    try:
        # Парсинг формата
        format = ReportFormat(request.format.lower())
        
        # Создание запроса на генерацию
        report_request = ReportRequest(
            template_id=request.template_id,
            parameters=request.parameters,
            format=format,
            email_to=request.email_to
        )
        
        # Запуск генерации
        report_id = await service.generate_report(report_request)
        
        return {
            "status": "success",
            "report_id": report_id,
            "message": "Генерация отчета запущена"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Некорректные параметры: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации отчета: {str(e)}")


@reports_router.get("/status/{report_id}")
async def get_report_status(
    report_id: str,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Получение статуса генерации отчета"""
    
    result = service.get_report_status(report_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Отчет не найден")
    
    return {
        "report_id": result.id,
        "template_id": result.template_id,
        "status": result.status,
        "created_at": result.created_at.isoformat(),
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "file_available": result.file_path is not None,
        "error_message": result.error_message
    }


@reports_router.get("/")
async def list_reports(
    limit: int = 50,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Получение списка отчетов"""
    
    reports = service.list_reports(limit=limit)
    
    reports_data = []
    for report in reports:
        reports_data.append({
            "id": report.id,
            "template_id": report.template_id,
            "status": report.status,
            "created_at": report.created_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None,
            "file_available": report.file_path is not None
        })
    
    return {
        "reports": reports_data,
        "total_count": len(reports_data)
    }


@reports_router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    service: ReportBuilderService = Depends(get_report_builder_service)
):
    """Скачивание готового отчета"""
    
    file_path = service.download_report(report_id)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="Файл отчета не найден")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type='application/octet-stream'
    )


@reports_router.get("/formats/")
async def get_report_formats():
    """Получение списка поддерживаемых форматов отчетов"""
    
    formats = [
        {
            "code": format.value,
            "name": format.value.upper(),
            "description": f"Отчет в формате {format.value.upper()}"
        }
        for format in ReportFormat
    ]
    
    return {
        "formats": formats
    }


@reports_router.get("/chart-types/")
async def get_chart_types():
    """Получение типов графиков"""
    
    chart_types = [
        {
            "code": chart_type.value,
            "name": chart_type.value.capitalize(),
            "description": f"График типа {chart_type.value}"
        }
        for chart_type in ChartType
    ]
    
    return {
        "chart_types": chart_types
    }


@reports_router.get("/aggregations/")
async def get_aggregation_types():
    """Получение типов агрегации"""
    
    aggregations = [
        {
            "code": agg.value,
            "name": agg.value.capitalize(),
            "description": f"Агрегация {agg.value}"
        }
        for agg in AggregationType
    ]
    
    return {
        "aggregations": aggregations
    } 