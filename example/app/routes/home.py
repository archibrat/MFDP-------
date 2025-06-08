from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from auth.authenticate import authenticate_cookie, authenticate
from auth.hash_password import HashPassword
from database.config import get_settings
from typing import Dict

settings = get_settings()
home_route = APIRouter()
hash_password = HashPassword()
templates = Jinja2Templates(directory="view")

@home_route.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Главная страница приложения.
    
    Args:
        request (Request): Объект запроса FastAPI

    Returns:
        HTMLResponse: HTML страница с контекстом пользователя
    """
    token = request.cookies.get(settings.COOKIE_NAME)
    if token:
        user = await authenticate_cookie(token)
    else:
        user = None

    context = {
        "user": user,
        "request": request
    }
    return templates.TemplateResponse("index.html", context)

@home_route.get("/private", response_class=HTMLResponse)
async def index_private(request: Request, user:str=Depends(authenticate_cookie)):
    """
    Приватная страница, доступная только авторизованным пользователям через cookie.
    
    Args:
        request (Request): Объект запроса FastAPI
        user (str): Информация о пользователе из cookie-аутентификации

    Returns:
        HTMLResponse: Приватная HTML страница
    """
    context = {
        "user": user,
        "request": request
    }
    return templates.TemplateResponse("private.html", context)

@home_route.get("/private2")
async def index_privat2(request: Request, user:str=Depends(authenticate)):
    """
    Приватный API эндпоинт, доступный только авторизованным пользователям.
    
    Args:
        request (Request): Объект запроса FastAPI
        user (str): Информация о пользователе из заголовка авторизации

    Returns:
        dict: Словарь с информацией о пользователе
    """
    return {"user": user}

@home_route.get(
    "/health",
    response_model=Dict[str, str],
    summary="Проверка работоспособности",
    description="Возвращает статус работоспособности сервиса"
)
async def health_check() -> Dict[str, str]:
    """
    Эндпоинт проверки работоспособности сервиса.

    Returns:
        Dict[str, str]: Сообщение о статусе работоспособности
    
    Raises:
        HTTPException: Если сервис недоступен
    """
    try:
        # Add actual health checks here
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable"
        )

