import pytest
from fastapi.testclient import TestClient
from api import app
from sqlmodel import SQLModel, Session, create_engine 
from database.database import get_session 
from sqlalchemy.pool import StaticPool
from auth.authenticate import authenticate

@pytest.fixture(name="session")  
def session_fixture():  
    engine = create_engine("sqlite:///testing.db", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(name="client") 
def client_fixture(session: Session):  
    def get_session_override():  
        return session

    app.dependency_overrides[get_session] = get_session_override  
    app.dependency_overrides[authenticate] = lambda: "user@test.ru"  
    
    client = TestClient(app)  
    yield client  
    app.dependency_overrides.clear()