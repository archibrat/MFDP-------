from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database.config import get_settings

settings = get_settings()
security = HTTPBearer()

class JWTHandler:
    """Обработчик JWT токенов"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Создание access токена
        
        Аргументы:
            data: Данные для включения в токен
            
        Возвращает:
            str: JWT токен
        """
        to_encode = data.copy()
        expire = datetime.now() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Создание refresh токена
        
        Аргументы:
            data: Данные для включения в токен
            
        Возвращает:
            str: JWT токен
        """
        to_encode = data.copy()
        expire = datetime.now() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Проверка и декодирование токена
        
        Аргументы:
            token: JWT токен для проверки
            
        Возвращает:
            Dict[str, Any]: Декодированные данные токена
            
        Исключения:
            HTTPException: При невалидном токене
        """
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Невалидный токен",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user_id(self, token: str) -> int:
        """
        Получение ID текущего пользователя из токена
        
        Аргументы:
            token: JWT токен
            
        Возвращает:
            int: ID пользователя
        """
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Не удалось получить данные пользователя",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return int(user_id)

# Глобальный экземпляр обработчика JWT
jwt_handler = JWTHandler()

def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Dependency для получения текущего пользователя из Bearer токена
    
    Аргументы:
        credentials: HTTP авторизационные данные
        
    Возвращает:
        int: ID текущего пользователя
    """
    return jwt_handler.get_current_user_id(credentials.credentials)

def get_current_user_from_cookie(request: Request) -> Optional[int]:
    """
    Получение текущего пользователя из cookie
    
    Аргументы:
        request: HTTP запрос
        
    Возвращает:
        Optional[int]: ID пользователя или None
    """
    token = request.cookies.get(settings.COOKIE_NAME)
    if not token:
        return None
    
    try:
        return jwt_handler.get_current_user_id(token)
    except HTTPException:
        return None

# Функции для совместимости с существующим кодом
def create_access_token(data: Dict[str, Any]) -> str:
    """Создание access токена (для совместимости)"""
    return jwt_handler.create_access_token(data)

def verify_token(token: str) -> Dict[str, Any]:
    """Проверка токена (для совместимости)"""
    return jwt_handler.verify_token(token)