import re
from typing import Optional
from passlib.context import CryptContext
from fastapi import HTTPException, status

class PasswordManager:
    """Менеджер для работы с паролями"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Настройки валидации пароля
        self.min_length = 8
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digit = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    def hash_password(self, password: str) -> str:
        """
        Хеширование пароля
        
        Аргументы:
            password: Пароль для хеширования
            
        Возвращает:
            str: Хешированный пароль
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Проверка пароля
        
        Аргументы:
            plain_password: Пароль в открытом виде
            hashed_password: Хешированный пароль
            
        Возвращает:
            bool: True если пароль верный
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> tuple[bool, Optional[str]]:
        """
        Валидация силы пароля
        
        Аргументы:
            password: Пароль для проверки
            
        Возвращает:
            tuple[bool, Optional[str]]: (валиден ли пароль, сообщение об ошибке)
        """
        if len(password) < self.min_length:
            return False, f"Пароль должен содержать минимум {self.min_length} символов"
        
        if len(password) > self.max_length:
            return False, f"Пароль должен содержать максимум {self.max_length} символов"
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            return False, "Пароль должен содержать хотя бы одну заглавную букву"
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            return False, "Пароль должен содержать хотя бы одну строчную букву"
        
        if self.require_digit and not re.search(r'\d', password):
            return False, "Пароль должен содержать хотя бы одну цифру"
        
        if self.require_special and not re.search(f'[{re.escape(self.special_chars)}]', password):
            return False, f"Пароль должен содержать хотя бы один специальный символ: {self.special_chars}"
        
        return True, None
    
    def validate_and_hash_password(self, password: str) -> str:
        """
        Валидация и хеширование пароля
        
        Аргументы:
            password: Пароль для валидации и хеширования
            
        Возвращает:
            str: Хешированный пароль
            
        Исключения:
            HTTPException: При невалидном пароле
        """
        is_valid, error_message = self.validate_password_strength(password)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
        
        return self.hash_password(password)
    
    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Проверка необходимости перехеширования пароля
        
        Аргументы:
            hashed_password: Хешированный пароль
            
        Возвращает:
            bool: True если нужно перехешировать
        """
        return self.pwd_context.needs_update(hashed_password)

# Глобальный экземпляр менеджера паролей
password_manager = PasswordManager()

# Функции для совместимости с существующим кодом
def create_hash(password: str) -> str:
    """Создание хеша пароля (для совместимости)"""
    return password_manager.hash_password(password)

def verify_hash(password: str, hashed_password: str) -> bool:
    """Проверка хеша пароля (для совместимости)"""
    return password_manager.verify_password(password, hashed_password)

# Класс для совместимости с существующим кодом
class HashPassword:
    """Класс для совместимости с существующим кодом"""
    
    def create_hash(self, password: str) -> str:
        return password_manager.hash_password(password)
    
    def verify_hash(self, password: str, hashed_password: str) -> bool:
        return password_manager.verify_password(password, hashed_password)