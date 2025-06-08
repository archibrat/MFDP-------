from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from database.database import get_session
from auth.hash_password import HashPassword
from auth.jwt_handler import create_access_token
from models.user import User
from services.crud import user as UserService
from typing import List, Dict
from services.logging.logging import get_logger

logger = get_logger(logger_name=__name__)

user_route = APIRouter()
hash_password = HashPassword()

@user_route.post(
    '/signup',
    response_model=Dict[str, str],
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация пользователя",
    description="Регистрация нового пользователя с помощью email и пароля")
async def signup(user: User, session=Depends(get_session)) -> Dict[str, str]:
    """
    Создание новой учетной записи пользователя.

    Аргументы:
        user: Данные для регистрации пользователя
        session: Сессия базы данных

    Возвращает:
        dict: Сообщение об успешной регистрации

    Исключения:
        HTTPException: Если пользователь уже существует
    """
    try:
        user_exist = UserService.get_user_by_email(user.email, session)
        
        if user_exist:
            raise HTTPException( 
            status_code=status.HTTP_409_CONFLICT, 
            detail="User with email provided exists already.")
        
        hashed_password = hash_password.create_hash(user.password)
        user.password = hashed_password 
        UserService.create_user(user, session)
        
        return {"message": "User created successfully"}

    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@user_route.post('/signin')
async def signin(form_data: OAuth2PasswordRequestForm = Depends(), session=Depends(get_session)) -> Dict[str, str]:
    """
    Аутентификация существующего пользователя.

    Аргументы:
        form_data: Учетные данные пользователя
        session: Сессия базы данных

    Возвращает:
        dict: Токен доступа и его тип

    Исключения:
        HTTPException: Если аутентификация не удалась
    """
    user_exist = UserService.get_user_by_email(form_data.username, session)
    
    if user_exist is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist")
    
    if hash_password.verify_hash(form_data.password, user_exist.password):
        access_token = create_access_token(user_exist.email)
        return {"access_token": access_token, "token_type": "Bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid details passed."
    )

@user_route.get(
    "/get_all_users",
    response_model=List[User],
    summary="Получение списка пользователей",
    response_description="Список всех пользователей"
)
async def get_all_users(session=Depends(get_session)) -> List[User]:
    """
    Получение списка всех пользователей.

    Аргументы:
        session: Сессия базы данных

    Возвращает:
        List[User]: Список пользователей

    Исключения:
        HTTPException: Если возникла ошибка при получении списка пользователей
    """
    try:
        users = UserService.get_all_users(session)
        logger.info(f"Retrieved {len(users)} users")
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving users"
        )