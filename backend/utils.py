# backend/utils.py

from sqlalchemy.orm import Session
from typing import Dict, Any

# Импортируем crud напрямую, т.к. utils на том же уровне, что и crud
from . import crud

# Определяем ключи настроек и их типы для валидации и конвертации
DEFAULT_SETTINGS = {
    "limit": {"type": int, "default": 10000},
    "z_threshold": {"type": float, "default": 3.0},
    "dbscan_eps": {"type": float, "default": 0.5},
    "dbscan_min_samples": {"type": int, "default": 5},
    "autoencoder_threshold": {"type": float, "default": 0.5}, # Добавим позже, если нужно
}

def get_typed_settings(db: Session) -> Dict[str, Any]:
    """Вспомогательная функция для получения настроек с конвертацией типов."""
    settings_from_db = crud.get_all_settings(db)
    typed_settings = {}
    for key, info in DEFAULT_SETTINGS.items():
        value_str = settings_from_db.get(key, str(info["default"]))
        try:
            typed_settings[key] = info["type"](value_str)
        except (ValueError, TypeError):
            # В реальном приложении здесь может быть более строгая обработка или логирование
            print(f"Warning (utils.py): Could not convert setting '{key}' value '{value_str}' to type {info['type']}. Using default: {info['default']}.")
            typed_settings[key] = info["default"]
    return typed_settings 