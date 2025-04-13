from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
# Убираем локальные определения, будем импортировать из schemas
# from pydantic import BaseModel, Field, field_validator
# from enum import Enum

from ..database import SessionLocal, get_db # Добавляем SessionLocal
from .. import crud, models, schemas # Импортируем схемы отсюда
# Импортируем get_detector и нужные классы детекторов
from ..ml_service.detector import IsolationForestDetector, StatisticalDetector, DbscanDetector, get_detector
# Импортируем хелпер для получения типизированных настроек (или скопируем его)
# from .settings import _get_typed_settings # Предполагается, что settings.py в том же уровне
# --- Временно скопируем логику _get_typed_settings для простоты --- 
def _get_typed_settings(db: Session) -> Dict[str, Any]:
    """Вспомогательная функция для получения настроек с конвертацией типов."""
    # Определяем ключи настроек и их типы (дублируем из settings.py)
    DEFAULT_SETTINGS = {
        "limit": {"type": int, "default": 10000},
        "z_threshold": {"type": float, "default": 3.0},
        "dbscan_eps": {"type": float, "default": 0.5},
        "dbscan_min_samples": {"type": int, "default": 5},
    }
    settings_from_db = crud.get_all_settings(db)
    typed_settings = {}
    for key, info in DEFAULT_SETTINGS.items():
        value_str = settings_from_db.get(key, str(info["default"]))
        try:
            typed_settings[key] = info["type"](value_str)
        except (ValueError, TypeError):
            print(f"Warning (anomalies.py): Could not convert setting '{key}' value '{value_str}' to type {info['type']}. Using default.")
            typed_settings[key] = info["default"]
    return typed_settings
# --------------------------------------------------------------

router = APIRouter()

# Убираем Enum, будем использовать List[str] из схемы
# class AlgorithmType(str, Enum): ...

# Убираем локальные Pydantic модели
# class InternalAnomalyResult(BaseModel): ...
# class DetectionParams(BaseModel): ... # Будет schemas.DetectionParams
# class DetectResponse(BaseModel): ... # Будет schemas.DetectResponse

# --- Фабрика детекторов (Убираем, используем импортированную) ---
# def get_detector(algorithm: AlgorithmType, params: DetectionParams):
#     ...

# --- Функции для фоновых задач (возвращаем параметры) ---
def train_models_background(db_session_factory, limit: int, z_threshold: float, entity_type: str):
    # Эта функция вызывалась из /train, который снова будет принимать параметры
    db = db_session_factory()
    try:
        print(f"Background Training ({entity_type}): Starting with limit={limit}, z_threshold={z_threshold}")

        # Статистический детектор
        if entity_type in ['activity', 'order']:
             # Создаем mock params для вызова get_detector (он ожидает DetectionParams)
             mock_detect_params_stat = schemas.DetectionParams(
                 entity_type=entity_type, algorithms=['statistical_zscore'],
                 limit=limit, z_threshold=z_threshold,
                 dbscan_eps=0.5, dbscan_min_samples=5 # Дефолтные для DBSCAN
             )
             stat_detector: StatisticalDetector = get_detector('statistical_zscore', mock_detect_params_stat)
             print(f"Initiating training/stats calculation for StatisticalDetector (entity='{entity_type}')")
             stat_detector.train(db, entity_type=entity_type, limit=limit)
             print(f"StatisticalDetector training/stats calculation completed.")

        # Isolation Forest
        if entity_type == 'activity':
             mock_detect_params_if = schemas.DetectionParams(
                 entity_type=entity_type, algorithms=['isolation_forest'],
                 limit=limit, z_threshold=z_threshold,
                 dbscan_eps=0.5, dbscan_min_samples=5
             )
             if_detector: IsolationForestDetector = get_detector('isolation_forest', mock_detect_params_if)
             print(f"Initiating training for IsolationForestDetector (entity='{entity_type}')")
             if_detector.train(db=db, entity_type=entity_type, limit=limit)
             print(f"IsolationForestDetector training completed.")

        print(f"Background training finished successfully for entity_type='{entity_type}'.")

    except Exception as e:
        print(f"Error during background training for entity_type='{entity_type}': {e}")
    finally:
        db.close()

# --- Эндпоинты ---

# Эндпоинт для получения списка сохраненных аномалий
@router.get("/", response_model=List[schemas.Anomaly])
def list_detected_anomalies(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    anomalies = crud.get_detected_anomalies(db, skip=skip, limit=limit)
    return anomalies

# ВОЗВРАЩАЕМ schemas.TrainingParams
@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
def train_anomaly_detectors(
    params: schemas.TrainingParams, # Снова принимаем параметры
    background_tasks: BackgroundTasks,
    # db: Session = Depends(get_db) # db больше не нужен здесь
):
    """
    Запускает обучение/подготовку.
    Принимает limit, z_threshold. Если entity_type не указан, обучает все.
    """
    print(f"Received request to train models with params: {params}")

    supported_entity_types = ['activity', 'order']
    entity_types_to_train = []

    if params.entity_type:
        if params.entity_type in supported_entity_types:
            entity_types_to_train.append(params.entity_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported entity_type '{params.entity_type}'. Supported: {supported_entity_types}")
    else:
        entity_types_to_train = supported_entity_types
        print(f"No specific entity_type provided. Will train all supported types: {entity_types_to_train}")

    for entity_type in entity_types_to_train:
        print(f"Adding background task for training entity_type='{entity_type}'")
        # Передаем параметры из запроса в фоновую задачу
        background_tasks.add_task(train_models_background, SessionLocal, params.limit, params.z_threshold, entity_type)

    entity_type_str = params.entity_type if params.entity_type else "all supported types"
    return {"message": f"Model training initiated in the background for entity type(s): {entity_type_str}"}

# ВОЗВРАЩАЕМ использование params
@router.post("/detect", response_model=schemas.DetectResponse)
def run_detection_and_save(
    params: schemas.DetectionParams, # Снова принимаем полные параметры
    db: Session = Depends(get_db)
):
    """
    Запускает поиск аномалий, используя параметры из запроса.
    """
    algo_list = ', '.join(params.algorithms)
    # Используем model_dump() для вывода всех параметров
    print(f"Received request to detect anomalies with params: {params.model_dump()}")

    # Убираем загрузку настроек из БД
    # settings = _get_typed_settings(db)
    limit = params.limit # Берем limit из параметров запроса
    # print(f"Using settings from DB for detection: {settings}")

    # Валидация (остается)
    if params.entity_type == 'order' and any(algo not in ['statistical_zscore'] for algo in params.algorithms):
        raise HTTPException(status_code=400, detail=f"Only 'statistical_zscore' is supported for entity_type='order'. Received: {params.algorithms}")

    # Очистка старых аномалий
    print(f"Clearing ALL previous anomaly results... (Using limit={limit} from request for detection)")
    num_deleted = crud.delete_all_anomalies(db=db)
    print(f"Finished clearing. {num_deleted} anomalies deleted.")

    all_detected_anomalies: List[schemas.Anomaly] = []
    saved_counts: Dict[str, int] = {algo: 0 for algo in params.algorithms}

    # Цикл по алгоритмам
    for algorithm_name in params.algorithms:
        print(f"--- Detecting with {algorithm_name} for entity '{params.entity_type}' (limit={limit}) ---")
        try:
            # Передаем params в get_detector
            detector_instance = get_detector(algorithm_name, params)
            # Передаем limit в detect (он теперь тоже берется из params)
            anomalies_found: List[Dict] = detector_instance.detect(db=db, entity_type=params.entity_type, limit=limit)

            print(f"--- {algorithm_name} detector found {len(anomalies_found)} potential anomalies ---")

            # ... (логика сохранения остается прежней) ...
            count_saved_for_algo = 0
            for anomaly_data in anomalies_found:
                try:
                    entity_type_found = anomaly_data.get('entity_type', params.entity_type)
                    entity_id = None
                    if entity_type_found == 'activity' and 'activity_id' in anomaly_data:
                        entity_id = anomaly_data['activity_id']
                    elif entity_type_found == 'order' and 'order_id' in anomaly_data:
                        entity_id = anomaly_data['order_id']
                    else:
                        entity_id = anomaly_data.get('id', anomaly_data.get('entity_id'))

                    if entity_id is None:
                        print(f"Warning: Could not determine entity_id for anomaly from {algorithm_name}. Skipping save.")
                        continue

                    anomaly_to_create = schemas.AnomalyCreate(
                        entity_type=entity_type_found,
                        entity_id=int(entity_id),
                        description=anomaly_data.get('reason', f"Anomaly detected by {algorithm_name}"),
                        severity=None,
                        detector_name=algorithm_name,
                        details=anomaly_data.get('details', anomaly_data)
                    )
                    crud.create_anomaly(db=db, anomaly=anomaly_to_create)
                    count_saved_for_algo += 1
                except Exception as db_exc:
                    print(f"Error adding anomaly from {algorithm_name} to session: {db_exc}")
                    pass

            saved_counts[algorithm_name] = count_saved_for_algo
            print(f"--- Added {count_saved_for_algo} anomalies from {algorithm_name} detector to session ---")

        except ValueError as e:
            print(f"Skipping algorithm {algorithm_name} due to configuration error: {e}")
        except Exception as e:
            print(f"Error during detection with {algorithm_name}: {e}")

    # Финальный commit
    try:
        db.commit()
        print("Final commit successful.")
    except Exception as commit_exc:
        print(f"Error during final commit: {commit_exc}. Rolling back.")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save detection results.")

    return schemas.DetectResponse(
        message=f"Detection for '{params.entity_type}' finished. Cleared {num_deleted} previous results. New results saved.",
        anomalies_saved_by_algorithm=saved_counts
    ) 