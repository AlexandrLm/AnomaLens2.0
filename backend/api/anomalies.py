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
from ..ml_service.detector import IsolationForestDetector, StatisticalDetector, DbscanDetector, get_detector, AutoencoderDetector
# --- Импортируем утилиту --- 
from ..utils import get_typed_settings

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
    db = db_session_factory()
    try:
        print(f"Background Training ({entity_type}): Starting with limit={limit}, z_threshold={z_threshold}")

        # --- Общий список алгоритмов для данного типа сущности ---
        algorithms_to_train = []
        if entity_type == 'activity':
            algorithms_to_train = ['statistical_zscore', 'isolation_forest', 'autoencoder']
        elif entity_type == 'order':
            algorithms_to_train = ['statistical_zscore']
        else:
            print(f"No specific algorithms defined for training entity_type='{entity_type}'")

        # --- Цикл по алгоритмам --- 
        for algo_name in algorithms_to_train:
            print(f"\n--- Initiating training for {algo_name} (entity='{entity_type}') ---")
            try:
                # Создаем mock DetectionParams, так как get_detector его ожидает.
                # Важно передать правильные параметры, которые могут понадобиться детектору.
                # Для Autoencoder конкретные параметры пока не важны, но для других могут быть.
                mock_detect_params = schemas.DetectionParams(
                    entity_type=entity_type,
                    algorithms=[algo_name],
                    limit=limit, # Limit может использоваться детекторами при загрузке данных
                    z_threshold=z_threshold, # Передаем, т.к. нужен для StatisticalDetector
                    # Передаем дефолты для DBSCAN, т.к. они есть в схеме
                    dbscan_eps=0.5, 
                    dbscan_min_samples=5
                )
                
                detector_instance = get_detector(algo_name, mock_detect_params)
                
                # Вызываем метод train детектора
                detector_instance.train(db=db, entity_type=entity_type, limit=limit)
                
                print(f"--- Completed training for {algo_name} (entity='{entity_type}') ---")

            except ImportError as ie:
                 print(f"Skipping training for {algo_name} due to missing dependency: {ie}")
            except Exception as train_exc:
                 print(f"ERROR during training for {algo_name} (entity='{entity_type}'): {train_exc}")
                 import traceback
                 traceback.print_exc() # Выводим полный traceback для отладки
        # ------------------------

        print(f"\nBackground training finished for entity_type='{entity_type}'.")

    except Exception as e:
        print(f"General error during background training setup for entity_type='{entity_type}': {e}")
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
@router.post("/detect", response_model=schemas.DetectResponse, tags=["Anomaly Detection"])
def run_detection_and_save(
    params: schemas.DetectionParams,
    db: Session = Depends(get_db)
):
    """
    Runs anomaly detection using the specified algorithms and saves results.
    Fetches current settings using the utility function.
    Clears previous anomaly results before saving new ones.
    """
    print(f"Received request to detect anomalies with params: {params}")
    anomalies_saved_count = {} 

    # --- ДОБАВЛЕНО: Очистка старых аномалий перед новым запуском --- 
    print("Clearing previous anomaly results...")
    try:
        num_deleted = crud.delete_all_anomalies(db=db)
        print(f"Finished clearing. {num_deleted} anomalies deleted.")
    except Exception as e:
        # Логируем ошибку, но не прерываем выполнение детекции
        print(f"Error clearing anomalies, proceeding anyway: {e}")
        db.rollback() # Откатываем транзакцию удаления, если она была начата
    # --------------------------------------------------------------

    # --- Получаем актуальные настройки с помощью утилиты ---
    current_settings = get_typed_settings(db)
    print(f"Using current settings for detection: {current_settings}")
    # ---------------------------------------------------

    for algo_name in params.algorithms:
        try:
            print(f"\n--- Initiating detection with {algo_name} ---")
            # Создаем параметры детекции, обогащая их текущими настройками
            # Важно: передаем параметры из запроса (params), если они заданы, иначе из current_settings
            detector_params = params.model_copy(update={
                'z_threshold': params.z_threshold if params.z_threshold is not None else current_settings.get('z_threshold', 3.0),
                'dbscan_eps': params.dbscan_eps if params.dbscan_eps is not None else current_settings.get('dbscan_eps', 0.5),
                'dbscan_min_samples': params.dbscan_min_samples if params.dbscan_min_samples is not None else current_settings.get('dbscan_min_samples', 5),
                # Добавляем порог для автоэнкодера из настроек
                'autoencoder_threshold': current_settings.get('autoencoder_threshold', 0.5)
            })

            detector_instance = get_detector(algo_name, detector_params)
            detected_anomalies = detector_instance.detect(db=db, entity_type=params.entity_type, limit=params.limit)
            print(f"--- {algo_name} detected {len(detected_anomalies)} potential anomalies.")
            
            anomalies_saved_for_algo = 0
            for anomaly_data in detected_anomalies:
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
                        print(f"Warning: Could not determine entity_id for anomaly from {algo_name}. Skipping save.")
                        continue
                        
                    # --- Получаем severity и score из данных детектора --- 
                    # (Устанавливаем значения по умолчанию, если их нет)
                    determined_severity = anomaly_data.get('severity', 'Low') # По умолчанию Low
                    anomaly_score = anomaly_data.get('anomaly_score') # Может быть None
                    # ------------------------------------------------------

                    anomaly_to_create = schemas.AnomalyCreate(
                        entity_type=entity_type_found,
                        entity_id=int(entity_id),
                        detector_name=algo_name,
                        detection_timestamp=datetime.utcnow(), # Используем текущее время UTC
                        severity=determined_severity,       # <-- Используем полученную серьезность
                        anomaly_score=anomaly_score,        # <-- Используем полученный score
                        details=anomaly_data.get('details', {}), # Берем детали или пустой dict
                        description=anomaly_data.get('reason', f"Anomaly detected by {algo_name}") # Используем reason если есть
                    )
                    # --- Вывод создаваемого объекта для отладки ---
                    # print(f"  Creating anomaly record: {anomaly_to_create.model_dump()}")
                    # ---------------------------------------------
                    crud.create_anomaly(db=db, anomaly=anomaly_to_create)
                    anomalies_saved_for_algo += 1
                except Exception as db_exc:
                    print(f"Error preparing/adding anomaly from {algo_name} to session: {db_exc}")
                    # Можно добавить traceback для детальной отладки
                    # import traceback
                    # traceback.print_exc()
                    pass # Пропускаем эту аномалию, но продолжаем цикл

            anomalies_saved_count[algo_name] = anomalies_saved_for_algo
            print(f"--- Added {anomalies_saved_for_algo} anomalies from {algo_name} detector to session ---")

        except ValueError as e:
            print(f"Skipping algorithm {algo_name} due to configuration error: {e}")
        except Exception as e:
            print(f"Error during detection with {algo_name}: {e}")
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
    total_anomalies = sum(anomalies_saved_count.values())
    response_message = f"Detection completed for entity '{params.entity_type}' using algorithms: [{', '.join(params.algorithms)}]. Found and saved {total_anomalies} anomaly records."
    
    # --- ИСПРАВЛЕНО: Возвращаем только поля, определенные в схеме DetectResponse --- 
    return schemas.DetectResponse(
        message=response_message, 
        anomalies_saved_by_algorithm=anomalies_saved_count
    ) 
    # -----------------------------------------------------------------------------

# ... (endpoint /{anomaly_id} - если есть) ... 