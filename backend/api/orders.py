from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import crud, schemas, models # Убедимся, что импортируем все необходимое
from ..database import get_db

router = APIRouter(
    prefix="/orders",
    tags=["Orders Context"], # Используем отдельный тег для контекста
    responses={404: {"description": "Not found"}},
)

# Здесь будут эндпоинты, связанные с заказами, включая контекстные 

@router.get(
    "/{order_id}/context/customer_history", 
    response_model=List[schemas.SimpleOrderHistoryItem], 
    # tags уже определен для роутера
)
def get_order_context_customer_history(
    order_id: int, 
    limit: int = Query(5, ge=1, le=20, description="Количество недавних заказов для показа"),
    db: Session = Depends(get_db)
):
    """
    Для указанного заказа возвращает список недавних заказов того же клиента 
    для контекста.
    """
    # 1. Получаем заказ (он нам нужен для customer_id)
    db_order = crud.get_order(db, order_id=order_id)
    if db_order is None:
        raise HTTPException(status_code=404, detail=f"Order (ID: {order_id}) not found")
    if db_order.customer is None:
        # Это странная ситуация
        raise HTTPException(status_code=404, detail=f"Customer not found for order (ID: {db_order.id})")
        
    customer_id = db_order.customer_id
    current_order_id = db_order.id # ID заказа, для которого запрашиваем контекст

    # 2. Получаем недавние заказы клиента через CRUD
    recent_orders = crud.get_recent_orders_by_customer_id(db, customer_id=customer_id, limit=limit)

    # 3. Преобразуем в схему ответа и отмечаем текущий заказ
    history_items = []
    for order in recent_orders:
        # Считаем количество позиций (items должны быть загружены через subqueryload в CRUD)
        item_count = len(order.items) if order.items else 0 
        history_item = schemas.SimpleOrderHistoryItem(
            id=order.id,
            created_at=order.created_at,
            total_amount=order.total_amount,
            item_count=item_count,
            is_current_anomaly=(order.id == current_order_id) # Отмечаем текущий
        )
        history_items.append(history_item)
        
    return history_items 