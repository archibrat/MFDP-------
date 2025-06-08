from sqlmodel import Session 
from models.user import User
import pytest

def test_create_user(session: Session):
    """
    Тест создания пользователя с валидными данными.
    
    Аргументы:
        session (Session): Сессия базы данных
    """
    user = User(id=1, email="test@mail.ru", password="1234")
    session.add(user)
    session.commit()
    
def test_fail_create_user(session: Session):
    """
    Тест создания пользователя с невалидными данными.
    
    Аргументы:
        session (Session): Сессия базы данных
        
    Ожидается исключение из-за слишком короткого пароля.
    """
    with pytest.raises(Exception) as ex:
        user = User(id=1, email="test_2@mail.ru", password="12")  # Слишком короткий пароль
        session.add(user)
        session.commit()
        
def test_delete_user(session: Session):
    """
    Тест удаления пользователя.
    
    Аргументы:
        session (Session): Сессия базы данных
    """
    test_create_user(session)
    
    user = session.get(User, 1)
    assert user is not None, "Пользователь с id=1 не найден"  # Fixed incorrect ID in error message

    session.delete(user)
    session.commit()

    deleted_user = session.get(User, 1)
    assert deleted_user is None, "Пользователь не был удален"
