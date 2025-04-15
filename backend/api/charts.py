from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

from ..database import get_db
from .. import crud

router = APIRouter(
    tags=["charts"],
    responses={404: {"description": "Not found"}},
)

@router.get("/activity_timeline", response_model=List[Dict])
def get_activity_timeline_data(
    interval: str = Query('day', enum=['day', 'hour'], description="Group activities by 'day' or 'hour'"),
    db: Session = Depends(get_db)
):
    """
    Возвращает данные для графика активности пользователей по времени.
    """
    try:
        data = crud.get_activity_counts_by_interval(db=db, interval=interval)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Логируем непредвиденную ошибку
        print(f"Error in /activity_timeline endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching activity timeline.")

@router.get("/anomaly_summary", response_model=List[Dict])
def get_anomaly_summary_data(db: Session = Depends(get_db)):
    """
    Возвращает данные для графика распределения аномалий по детекторам.
    """
    try:
        data = crud.get_anomaly_counts_by_detector(db=db)
        return data
    except Exception as e:
        # Логируем непредвиденную ошибку
        print(f"Error in /anomaly_summary endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching anomaly summary.")

# --- НОВЫЙ Эндпоинт для скоров аномалий ---

# Определяем допустимые имена детекторов
class DetectorName(str, Enum):
    statistical_zscore = "statistical_zscore"
    isolation_forest = "isolation_forest"
    autoencoder = "autoencoder"
    # DBSCAN не имеет скора, поэтому его не добавляем

@router.get("/anomaly_scores", response_model=List[float])
def get_anomaly_scores_distribution(
    detector_name: DetectorName = Query(..., description="Name of the detector to get scores for"),
    start_time: Optional[datetime] = Query(None, description="Filter by start time (ISO 8601 format)"),
    end_time: Optional[datetime] = Query(None, description="Filter by end time (ISO 8601 format)"),
    db: Session = Depends(get_db)
):
    """
    Возвращает список значений anomaly_score для указанного детектора,
    опционально отфильтрованный по времени.
    Используется для построения гистограмм распределения.
    """
    try:
        scores = crud.get_anomaly_scores_by_detector(
            db=db,
            detector_name=detector_name.value, # Используем значение Enum
            start_time=start_time,
            end_time=end_time
        )
        return scores
    except Exception as e:
        # Логируем непредвиденную ошибку
        print(f"Error in /anomaly_scores endpoint for detector {detector_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching anomaly scores.")
# -------------------------------------------- 

# === НОВЫЙ ЭНДПОИНТ ДЛЯ SCATTER PLOT ===

# Определяем схему ответа для Scatter Plot
class ScatterPoint(BaseModel):
    id: int
    x: float
    y: float # item_count будет float? Хотя в crud он int. Сделаем float для гибкости.
    is_anomaly: bool

@router.get("/feature_scatter", response_model=List[ScatterPoint])
def get_feature_scatter_plot(
    entity_type: str = Query("order", description="Type of entity ('order' or 'activity_session')"),
    feature_x: str = Query("total_amount", description="Feature for X-axis (depends on entity_type)"),
    feature_y: str = Query("item_count", description="Feature for Y-axis (depends on entity_type)"),
    limit: int = Query(500, description="Maximum number of points to return", ge=10, le=5000),
    db: Session = Depends(get_db)
):
    """
    Возвращает данные для диаграммы рассеяния (scatter plot).
    Поддерживает entity_type='order' и 'activity_session'.
    """
    try:
        if entity_type == 'order':
            data = crud.get_feature_scatter_data(
                db=db,
                entity_type=entity_type, # Передаем для валидации внутри crud
                feature_x=feature_x,
                feature_y=feature_y,
                limit=limit
            )
            # В crud уже приводит к float
            # return [ScatterPoint(id=p["id"], x=p["x"], y=p["y"], is_anomaly=p["is_anomaly"]) for p in data]
            return data # Возвращаем как есть, т.к. формат совпадает
        elif entity_type == 'activity_session':
            data = crud.get_session_scatter_data(
                db=db,
                feature_x=feature_x,
                feature_y=feature_y,
                limit=limit
            )
            # ID в этом случае - это session_id (строка), но схема ожидает int.
            # Пока оставим так, на фронтенде ID не критичен для отображения.
            # В идеале, схема ScatterPoint должна быть более гибкой или использовать Union.
            # Преобразуем session_id в 0, чтобы соответствовать схеме (временное решение)
            return [ScatterPoint(id=0, x=p["x"], y=p["y"], is_anomaly=p["is_anomaly"]) for p in data]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported entity_type: {entity_type}")

    except (NotImplementedError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========================================== 