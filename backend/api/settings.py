from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from ..database import get_db
from .. import crud, schemas
# --- Импортируем утилиты --- 
from ..utils import get_typed_settings, DEFAULT_SETTINGS

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    responses={404: {"description": "Not found"}},
)

# --- Удаляем локальные определения --- 
# # Определяем ключи настроек и их типы для валидации и конвертации
# DEFAULT_SETTINGS = {
#     "limit": {"type": int, "default": 10000},
#     "z_threshold": {"type": float, "default": 3.0},
#     "dbscan_eps": {"type": float, "default": 0.5},
#     "dbscan_min_samples": {"type": int, "default": 5},
# }
# 
# def _get_typed_settings(db: Session) -> Dict[str, Any]:
#    ...
#    return typed_settings
# ------------------------------------

@router.get("/", response_model=Dict[str, Any])
def read_settings(db: Session = Depends(get_db)):
    """
    Получает все текущие настройки из базы данных с преобразованием типов.
    Если настройка отсутствует в БД, используется значение по умолчанию.
    Использует общую функцию get_typed_settings из utils.
    """
    # Используем импортированную функцию
    return get_typed_settings(db)

@router.put("/", response_model=Dict[str, Any])
def update_settings(
    settings_update: Dict[str, Any], # Принимаем словарь ключ-значение
    db: Session = Depends(get_db)
):
    """
    Обновляет настройки в базе данных.
    Принимает словарь с ключами и новыми значениями.
    Неизвестные ключи игнорируются. Значения сохраняются как строки.
    Использует DEFAULT_SETTINGS из utils для валидации.
    """
    updated_settings = {}
    # Используем импортированный DEFAULT_SETTINGS
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
    # Используем импортированную функцию
    return get_typed_settings(db) 