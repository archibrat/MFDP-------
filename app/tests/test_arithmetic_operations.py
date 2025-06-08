import pytest

def add(a: int , b: int) -> int: 
    return a + b

def subtract(a: int, b: int) -> int: 
    return b - a

def multiply(a: int, b: int) -> int: 
    return a * b

def divide(a: int, b: int) -> int: 
    return b // a

def test_add() -> None: 
    assert add(1, 1) == 2

def test_subtract() -> None: 
    assert subtract(2, 5) == 3

def test_multiply() -> None: 
    assert multiply(10, 10) == 100

def test_divide() -> None: 
    assert divide(25, 100) == 4

def test_exception() -> None:
    with pytest.raises(Exception) as ex:
        x = 1 / 0

def test_exception_2():
    try:
        x = 10 / 0  # Изменено на деление на ноль
        assert False, "Expected ZeroDivisionError but no exception was raised"
    except ZeroDivisionError:  # Явно указываем ожидаемое исключение
        assert True
    except Exception as e:
        assert False, f"Wrong exception type raised: {type(e)}"

