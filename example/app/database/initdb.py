from sqlmodel import SQLModel, Session, select
from models.user import User
from models.event import Event
from auth.hash_password import HashPassword
from database.database import get_database_engine

def init_db(drop_all: bool = False) -> None:
    """
    Инициализация схемы базы данных.
    
    Аргументы:
        drop_all: Если True, удаляет все таблицы перед созданием
    
    Исключения:
        Exception: Любые исключения, связанные с базой данных
    """
    try:
        engine = get_database_engine()
        if drop_all:
            # Удаление всех таблиц, если указано
            SQLModel.metadata.drop_all(engine)
        
        # Создание всех таблиц
        SQLModel.metadata.create_all(engine)

        # Создание начальных данных
        with Session(engine) as session:
            # Проверяем, существуют ли уже пользователи
            if session.exec(select(User)).first() is None:
                hash_password = HashPassword()
                
                # Создание стандартных пользователей
                admin = User(
                    email="admin@example.com",
                    password=hash_password.create_hash("admin123")
                )
                user1 = User(
                    email="user1@example.com",
                    password=hash_password.create_hash("user123")
                )
                user2 = User(
                    email="user2@example.com",
                    password=hash_password.create_hash("user123")
                )

                # Создание стандартных событий
                event1 = Event(
                    title="Открытие IT конференции",
                    image="conference.jpg",
                    description="Ежегодная конференция по IT технологиям"
                )
                event2 = Event(
                    title="Хакатон по ML",
                    image="hackathon.jpg",
                    description="Двухдневный хакатон по машинному обучению"
                )
                event3 = Event(
                    title="Мастер-класс по Python",
                    image="python_workshop.jpg",
                    description="Практический мастер-класс по Python для начинающих"
                )

                # Связывание событий с пользователями
                admin.events.append(event1)
                user1.events.append(event2)
                user2.events.append(event3)

                # Сохранение в базу данных
                session.add(admin)
                session.add(user1)
                session.add(user2)
                session.commit()

    except Exception as e:
        raise
