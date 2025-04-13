# backend/api/dashboard.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func # Required for count function

from .. import models
from ..database import get_db

router = APIRouter()

@router.get("/summary", summary="Get data summary counts")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Retrieves the total counts for key entities in the database.
    """
    try:
        total_customers = db.query(func.count(models.Customer.id)).scalar()
        total_products = db.query(func.count(models.Product.id)).scalar()
        total_orders = db.query(func.count(models.Order.id)).scalar()
        total_activities = db.query(func.count(models.UserActivity.id)).scalar()
        total_categories = db.query(func.count(models.Category.id)).scalar()

        return {
            "total_customers": total_customers,
            "total_products": total_products,
            "total_orders": total_orders,
            "total_activities": total_activities,
            "total_categories": total_categories,
            # Можно добавить больше метрик, например, средний чек, кол-во аномалий и т.д.
        }
    except Exception as e:
        # Log the exception e
        print(f"Error fetching dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch dashboard summary data") 