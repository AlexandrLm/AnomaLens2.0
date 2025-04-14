from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import itertools
from operator import attrgetter
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

# --- Вспомогательная функция для определения общей серьезности ---
def get_overall_severity(detectors: List[schemas.AnomalyDetectorInfo]) -> Optional[str]:
    """Определяет общую серьезность аномалии по списку детекторов."""
    severities = [d.severity for d in detectors if d.severity] # Собираем все не-None серьезности
    if not severities: return 'Unknown' # Если серьезность не определена ни одним детектором
    if 'High' in severities: return 'High'
    if 'Medium' in severities: return 'Medium'
    return 'Low' # Если есть только Low

# --- Новый эндпоинт для получения деталей аномалии ---
@router.get("/{anomaly_id}", response_model=schemas.AnomalyDetailResponse, tags=["Anomaly Detection"])
def read_anomaly_details(
    anomaly_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve details for a specific anomaly, including the related entity (activity or order).
    """
    db_anomaly = crud.get_anomaly(db, anomaly_id=anomaly_id)
    if db_anomaly is None:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    related_entity_data = None
    try:
        if db_anomaly.entity_type == 'activity':
            related_entity_data = crud.get_user_activity(db, activity_id=db_anomaly.entity_id)
        elif db_anomaly.entity_type == 'order':
            related_entity_data = crud.get_order(db, order_id=db_anomaly.entity_id)
    except Exception as e:
        # Log the error, but don't fail the request if the related entity fetch fails
        print(f"Warning: Could not fetch related {db_anomaly.entity_type} (ID: {db_anomaly.entity_id}) for anomaly {anomaly_id}: {e}")

    return schemas.AnomalyDetailResponse(anomaly=db_anomaly, related_entity=related_entity_data)

# --- Обновленный эндпоинт GET / для консолидированных аномалий --- 
@router.get("/", response_model=List[schemas.ConsolidatedAnomaly], tags=["Anomaly Detection"])
def read_anomalies_consolidated(
    skip: int = 0, 
    limit: int = Query(100, ge=1, le=500, description="Max number of consolidated anomalies to return"), 
    db: Session = Depends(get_db)
):
    """Retrieve a consolidated list of detected anomalies, grouped by entity."""
    print(f"GET /api/anomalies/ called with skip={skip}, limit={limit}")
    
    consolidated_list = []
    
    # 1. Выбираем ВСЕ сырые аномалии из БД (без skip/limit здесь, применим позже)
    # ВАЖНО: Сразу сортируем по entity_type, entity_id для группировки
    raw_anomalies = db.query(models.Anomaly).order_by(
        models.Anomaly.entity_type,
        models.Anomaly.entity_id,
        models.Anomaly.detection_timestamp.desc() # Можно добавить вторичную сортировку внутри группы
    ).all() 
    
    if not raw_anomalies:
        return []
    
    # 2. Группируем и создаем ConsolidatedAnomaly
    keyfunc = attrgetter('entity_type', 'entity_id')
    # Так как отсортировали в запросе, дополнительная сортировка не нужна
    # raw_anomalies.sort(key=keyfunc) 

    for key, group in itertools.groupby(raw_anomalies, key=keyfunc):
        entity_type, entity_id = key
        group_list = list(group)
        
        # Преобразуем сырые аномалии группы в AnomalyDetectorInfo
        triggered_detectors = []
        for anomaly in group_list:
            # --- ИСПРАВЛЕНО: Используем description из модели Anomaly как reason --- 
            detector_info = schemas.AnomalyDetectorInfo(
                detector_name=anomaly.detector_name,
                anomaly_score=anomaly.anomaly_score, 
                severity=anomaly.severity,
                details=anomaly.details, 
                detection_timestamp=anomaly.detection_timestamp,
                reason=anomaly.description # <-- Вот здесь берем значение из поля description
            )
            triggered_detectors.append(detector_info)
            # Старый способ: schemas.AnomalyDetectorInfo.from_orm(anomaly) - не сработает, т.к. поля reason нет в модели Anomaly
        # ---------------------------------------------------------------------
        
        # Определяем агрегированные поля
        last_detected = max(info.detection_timestamp for info in triggered_detectors)
        overall_severity = get_overall_severity(triggered_detectors)
        detector_count = len(triggered_detectors)
        
        consolidated_list.append(schemas.ConsolidatedAnomaly(
            entity_type=entity_type,
            entity_id=entity_id,
            last_detected_at=last_detected,
            overall_severity=overall_severity,
            detector_count=detector_count,
            triggered_detectors=triggered_detectors
        ))
        
    # 3. Сортируем результат по времени последнего обнаружения (сначала новые)
    consolidated_list.sort(key=attrgetter('last_detected_at'), reverse=True)
    
    # 4. Применяем лимит и skip к консолидированному списку
    final_list = consolidated_list[skip:skip+limit]

    print(f"Returning {len(final_list)} consolidated anomalies.")
    return final_list

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
    params: schemas.DetectionParams,
    db: Session = Depends(get_db)
):
    """
    Запускает поиск аномалий, используя параметры из запроса.
    Теперь также использует 'severity', возвращаемое детекторами.
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
    print(f"Clearing ALL previous anomaly results... (Using limit={params.limit} from request for detection)")
    num_deleted = crud.delete_all_anomalies(db=db)
    print(f"Finished clearing. {num_deleted} anomalies deleted.")

    all_detected_anomalies: List[schemas.Anomaly] = [] # Технически, это не используется
    saved_counts: Dict[str, int] = {algo: 0 for algo in params.algorithms}

    # Цикл по алгоритмам
    for algorithm_name in params.algorithms:
        print(f"--- Detecting with {algorithm_name} for entity '{params.entity_type}' (limit={params.limit}) ---")
        try:
            detector_instance = get_detector(algorithm_name, params)
            anomalies_found: List[Dict] = detector_instance.detect(db=db, entity_type=params.entity_type, limit=params.limit)

            print(f"--- {algorithm_name} detector found {len(anomalies_found)} potential anomalies ---")

            count_saved_for_algo = 0
            for anomaly_data in anomalies_found:
                try:
                    # --- Логика определения entity_type и entity_id (остается) --- 
                    entity_type_found = anomaly_data.get('entity_type', params.entity_type) # Берем из данных или из параметров
                    entity_id = None
                    if entity_type_found == 'activity':
                        entity_id = anomaly_data.get('activity_id', anomaly_data.get('id', anomaly_data.get('entity_id')))
                    elif entity_type_found == 'order':
                        entity_id = anomaly_data.get('order_id', anomaly_data.get('id', anomaly_data.get('entity_id')))
                    else: # Общий случай, если детектор вернул entity_id
                        entity_id = anomaly_data.get('entity_id', anomaly_data.get('id'))

                    if entity_id is None:
                        print(f"Warning: Could not determine entity_id for anomaly from {algorithm_name}. Skipping save.")
                        continue
                        
                    # --- Получаем severity и score из данных детектора --- 
                    # (Устанавливаем значения по умолчанию, если их нет)
                    determined_severity = anomaly_data.get('severity', 'Low') # По умолчанию Low
                    anomaly_score = anomaly_data.get('anomaly_score') # Может быть None
                    # ------------------------------------------------------

                    anomaly_to_create = schemas.AnomalyCreate(
                        entity_type=entity_type_found,
                        entity_id=int(entity_id),
                        detector_name=algorithm_name,
                        detection_timestamp=datetime.utcnow(), # Используем текущее время UTC
                        severity=determined_severity,       # <-- Используем полученную серьезность
                        anomaly_score=anomaly_score,        # <-- Используем полученный score
                        details=anomaly_data.get('details', {}), # Берем детали или пустой dict
                        description=anomaly_data.get('reason', f"Anomaly detected by {algorithm_name}") # Используем reason если есть
                    )
                    # --- Вывод создаваемого объекта для отладки ---
                    # print(f"  Creating anomaly record: {anomaly_to_create.model_dump()}")
                    # ---------------------------------------------
                    crud.create_anomaly(db=db, anomaly=anomaly_to_create)
                    count_saved_for_algo += 1
                except Exception as db_exc:
                    print(f"Error preparing/adding anomaly from {algorithm_name} to session: {db_exc}")
                    # Можно добавить traceback для детальной отладки
                    # import traceback
                    # traceback.print_exc()
                    pass # Пропускаем эту аномалию, но продолжаем цикл

            saved_counts[algorithm_name] = count_saved_for_algo
            print(f"--- Added {count_saved_for_algo} anomalies from {algorithm_name} detector to session ---")

        except ValueError as e:
            print(f"Skipping algorithm {algorithm_name} due to configuration error: {e}")
        except Exception as e:
            print(f"Error during detection with {algorithm_name}: {e}")
            # import traceback
            # traceback.print_exc()

    # Финальный commit
    try:
        db.commit()
        print("Final commit successful.")
    except Exception as commit_exc:
        print(f"Error during final commit: {commit_exc}. Rolling back.")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save detection results.")

    # Формируем ответ
    total_anomalies = sum(saved_counts.values())
    return schemas.DetectResponse(
        message=f"Detection completed for entity '{params.entity_type}' using [{', '.join(params.algorithms)}]. Found and saved {total_anomalies} anomaly records.",
        algorithms_used=params.algorithms,
        entity_type=params.entity_type,
        anomalies_saved_by_algorithm=saved_counts
    ) 