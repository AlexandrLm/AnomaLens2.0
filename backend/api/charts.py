from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any

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