from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Dict

from ..database import get_db
from ..simulator import generator # Импортируем наш генератор

router = APIRouter()

class GenerationConfig(BaseModel):
    num_customers: int = Field(100, ge=0, description="Number of customers to generate")
    num_products: int = Field(50, ge=0, description="Number of products to generate")
    num_orders: int = Field(200, ge=0, description="Number of orders to generate")
    num_activities: int = Field(1000, ge=0, description="Number of user activities to generate")
    order_anomaly_rate: float = Field(0.05, ge=0.0, le=1.0, description="Approximate fraction of orders with anomalous amounts (0.0 to 1.0)")
    activity_anomaly_rate: float = Field(0.05, ge=0.0, le=1.0, description="Approximate fraction of activity generation attempts that result in anomalies (bursts, failed logins) (0.0 to 1.0)")

@router.post("/generate", response_model=Dict[str, int], status_code=status.HTTP_201_CREATED)
def generate_test_data(
    config: GenerationConfig = Body(..., embed=True, description="Configuration for data generation including anomaly rates"),
    db: Session = Depends(get_db)
):
    """
    Generates test data (categories, products, customers, orders, activities)
    based on the provided configuration, including intentional anomalies.
    Returns a summary of generated items.
    """
    print(f"Received generation request with config: {config}")
    try:
        summary = generator.generate_all_data(
            db=db,
            num_customers=config.num_customers,
            num_products=config.num_products,
            num_orders=config.num_orders,
            num_activities=config.num_activities,
            order_anomaly_rate=config.order_anomaly_rate,
            activity_anomaly_rate=config.activity_anomaly_rate
        )
        return summary
    except Exception as e:
        print(f"Error during data generation: {e}")
        # Можно добавить более детальную обработку ошибок
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during data generation: {str(e)}"
        ) 