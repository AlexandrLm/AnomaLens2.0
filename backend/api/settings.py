from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from ..database import get_db
from .. import crud, schemas

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    responses={404: {"description": "Not found"}},
)

# Определяем ключи настроек и их типы для валидации и конвертации
# (Можно расширить по мере необходимости)
DEFAULT_SETTINGS = {
    "limit": {"type": int, "default": 10000},
    "z_threshold": {"type": float, "default": 3.0},
    "dbscan_eps": {"type": float, "default": 0.5},
    "dbscan_min_samples": {"type": int, "default": 5},
}

def _get_typed_settings(db: Session) -> Dict[str, Any]:
    """Вспомогательная функция для получения настроек с конвертацией типов."""
    settings_from_db = crud.get_all_settings(db)
    typed_settings = {}
    for key, info in DEFAULT_SETTINGS.items():
        value_str = settings_from_db.get(key, str(info["default"]))
        try:
            typed_settings[key] = info["type"](value_str)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert setting '{key}' value '{value_str}' to type {info['type']}. Using default.")
            typed_settings[key] = info["default"]
    return typed_settings

@router.get("/", response_model=Dict[str, Any])
def read_settings(db: Session = Depends(get_db)):
    """
    Получает все текущие настройки из базы данных с преобразованием типов.
    Если настройка отсутствует в БД, используется значение по умолчанию.
    """
    return _get_typed_settings(db)

@router.put("/", response_model=Dict[str, Any])
def update_settings(
    settings_update: Dict[str, Any], # Принимаем словарь ключ-значение
    db: Session = Depends(get_db)
):
    """
    Обновляет настройки в базе данных.
    Принимает словарь с ключами и новыми значениями.
    Неизвестные ключи игнорируются. Значения сохраняются как строки.
    """
    updated_settings = {}
    for key, value in settings_update.items():
        if key in DEFAULT_SETTINGS: # Обновляем только известные настройки
            # Перед сохранением преобразуем значение обратно в строку
            value_str = str(value)
            # Валидация типа перед сохранением (опционально, но полезно)
            try:
                DEFAULT_SETTINGS[key]["type"](value) # Пробуем конвертировать
                crud.update_setting(db=db, key=key, value=value_str)
                updated_settings[key] = value # Сохраняем типизированное значение для ответа
                print(f"Updated setting: {key} = {value_str}")
            except (ValueError, TypeError):
                print(f"Warning: Invalid type for setting '{key}'. Expected {DEFAULT_SETTINGS[key]['type']}, got '{value}'. Skipping update.")
        else:
            print(f"Warning: Unknown setting key '{key}' received. Ignoring.")
    
    # Возвращаем все текущие настройки (включая обновленные) с правильными типами
    return _get_typed_settings(db) 