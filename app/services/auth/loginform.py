from fastapi import Request
from typing import Optional, List

class LoginForm:
    """Класс для обработки формы авторизации.
    
    Отвечает за валидацию данных формы входа пользователя,
    включая проверку email и пароля.
    """
    
    def __init__(self, request: Request):
        """
        Инициализация формы логина.
        
        Args:
            request (Request): Объект запроса FastAPI
        """
        self.request: Request = request
        self.errors: List[str] = []  # Уточняем тип списка для ошибок
        self.username: Optional[str] = None
        self.password: Optional[str] = None

    async def load_data(self) -> None:
        """Загружает данные из формы запроса."""
        form = await self.request.form()
        self.username = form.get("username")
        self.password = form.get("password")

    async def is_valid(self) -> bool:
        """Проверяет валидность введенных данных.
        
        Returns:
            bool: True если данные валидны, False если есть ошибки
        """
        # Проверка email
        if not self.username or '@' not in self.username:
            self.errors.append("Требуется указать корректный email")
        
        # Проверка пароля
        if not self.password or len(self.password) < 4:
            self.errors.append("Требуется указать пароль")

        return len(self.errors) == 0