from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from .. import crud, schemas
from ..database import get_db

router = APIRouter()

# --- Схема ответа для списка активностей с общим количеством ---
class UserActivityResponse(BaseModel):
    totalCount: int
    data: List[schemas.UserActivity]
# -----------------------------------------------------------

@router.post("/", response_model=schemas.UserActivity, status_code=status.HTTP_201_CREATED)
def create_new_user_activity(activity: schemas.UserActivityCreate, db: Session = Depends(get_db)):
    """
    Записывает новое событие активности пользователя.
    Принимает `action_type` и опционально `customer_id`, `session_id`, `details`, `ip_address`, `user_agent`.
    """
    # Дополнительная валидация может быть добавлена здесь (например, типы action_type)
    try:
        return crud.create_user_activity(db=db, activity=activity)
    except Exception as e:
        # Обработка потенциальных ошибок при создании
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating the user activity.")

@router.get("/", response_model=UserActivityResponse)
def read_activities(
    skip: int = Query(0, ge=0, description="Number of activities to skip"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of activities to return"),
    start_time: Optional[datetime] = Query(None, description="Filter by start time (ISO 8601 format)"),
    end_time: Optional[datetime] = Query(None, description="Filter by end time (ISO 8601 format)"),
    db: Session = Depends(get_db)
):
    """
    Получает список активностей пользователей с пагинацией и фильтрацией по времени.
    Возвращает объект с общим количеством записей и списком активностей для текущей страницы.
    """
    total_count, activities = crud.get_user_activities(
        db,
        skip=skip,
        limit=limit,
        start_time=start_time,
        end_time=end_time
    )

    return UserActivityResponse(totalCount=total_count, data=activities)

@router.get("/{activity_id}", response_model=schemas.UserActivity)
def read_single_user_activity(activity_id: int, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретной активности по ее ID.
    """
    db_activity = crud.get_user_activity(db, activity_id=activity_id)
    if db_activity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User activity not found")
    return db_activity

# --- НОВЫЙ ЭНДПОИНТ: Получение контекста - история сессии --- 
@router.get(
    "/{activity_id}/context/session_history",
    response_model=List[schemas.SimpleActivityHistoryItem],
    tags=["Activities Context"] # Добавляем отдельный тег для контекста
)
def get_activity_context_session_history(
    activity_id: int,
    db: Session = Depends(get_db)
):
    """
    Для указанной активности возвращает список всех действий пользователя
    в рамках той же сессии для контекста.
    """
    # 1. Получаем исходную активность (нужна для session_id)
    db_activity = crud.get_user_activity(db, activity_id=activity_id)
    if db_activity is None:
        raise HTTPException(status_code=404, detail=f"Activity (ID: {activity_id}) not found")
    if db_activity.session_id is None:
        raise HTTPException(status_code=400, detail=f"Activity (ID: {activity_id}) does not have a session ID")

    session_id = db_activity.session_id
    current_activity_id = db_activity.id # ID активности, для которой запрашиваем контекст

    # 2. Получаем все активности сессии через CRUD
    session_activities = crud.get_activities_by_session_id(db, session_id=session_id)

    # 3. Преобразуем в схему ответа и отмечаем текущую аномальную активность
    history_items = []
    for activity in session_activities:
        history_item = schemas.SimpleActivityHistoryItem(
            id=activity.id,
            timestamp=activity.timestamp,
            action_type=activity.action_type,
            details=activity.details,
            session_id=activity.session_id,
            is_current_anomaly=(activity.id == current_activity_id) # Отмечаем текущую
        )
        history_items.append(history_item)
        
    return history_items
# ------------------------------------------------------------ 