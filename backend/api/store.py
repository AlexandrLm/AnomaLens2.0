from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import crud, models, schemas
from ..database import get_db

router = APIRouter()

# === Простой тестовый эндпоинт ===
@router.get("/ping", tags=["Test"]) # Добавим временный тег для ясности
async def ping():
    return {"ping": "pong from store router"}

# === Category Endpoints ===

@router.post("/categories/", response_model=schemas.Category, status_code=status.HTTP_201_CREATED)
def create_new_category(category: schemas.CategoryCreate, db: Session = Depends(get_db)):
    """
    Создает новую категорию товаров.
    Проверяет, существует ли уже категория с таким именем.
    """
    db_category = crud.get_category_by_name(db, name=category.name)
    if db_category:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Category with this name already exists")
    return crud.create_category(db=db, category=category)

@router.get("/categories/", response_model=List[schemas.Category])
def read_all_categories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Получает список всех категорий с пагинацией.
    """
    categories = crud.get_categories(db, skip=skip, limit=limit)
    return categories

@router.get("/categories/{category_id}", response_model=schemas.Category)
def read_single_category(category_id: int, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретной категории по ее ID.
    """
    db_category = crud.get_category(db, category_id=category_id)
    if db_category is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")
    return db_category

# === Product Endpoints ===

@router.post("/products/", response_model=schemas.Product, status_code=status.HTTP_201_CREATED)
def create_new_product(product: schemas.ProductCreate, db: Session = Depends(get_db)):
    """
    Создает новый товар.
    Требует указания `name` и `price`.
    `description` и `category_id` опциональны.
    Если указан `category_id`, проверяет существование категории.
    """
    try:
        return crud.create_product(db=db, product=product)
    except ValueError as e:
        # Перехватываем ошибку из crud.create_product, если категория не найдена
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/products/", response_model=List[schemas.Product])
def read_all_products(
    skip: int = Query(0, ge=0, description="Number of products to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of products to return"),
    db: Session = Depends(get_db)
):
    """
    Получает список всех товаров с пагинацией.
    Включает информацию о категории товара (если она есть).
    """
    products = crud.get_products(db, skip=skip, limit=limit)
    return products

@router.get("/products/{product_id}", response_model=schemas.Product)
def read_single_product(product_id: int, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретном товаре по его ID.
    Включает информацию о категории товара (если она есть).
    """
    db_product = crud.get_product(db, product_id=product_id)
    if db_product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    # Явно загружаем категорию, если она не была загружена (хотя get_product ее не загружает)
    # На самом деле, схема Product ожидает Category, так что SQLAlchemy (в зависимости от настроек)
    # может попытаться загрузить ее лениво при формировании ответа.
    # Явное указание в get_products через joinedload - более надежный способ.
    # Для get_product отдельного, если нужна категория, ее тоже лучше явно загружать.
    # Пока оставим так, схема Product включает Optional[Category]
    return db_product

# === Customer Endpoints ===

@router.post("/customers/", response_model=schemas.Customer, status_code=status.HTTP_201_CREATED)
def create_new_customer(customer: schemas.CustomerCreate, db: Session = Depends(get_db)):
    """
    Создает нового клиента (пользователя).
    Проверяет, существует ли уже клиент с таким email.
    """
    db_customer = crud.get_customer_by_email(db, email=customer.email)
    if db_customer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return crud.create_customer(db=db, customer=customer)

@router.get("/customers/", response_model=List[schemas.Customer])
def read_all_customers(
    skip: int = Query(0, ge=0, description="Number of customers to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of customers to return"),
    db: Session = Depends(get_db)
):
    """
    Получает список всех клиентов с пагинацией.
    """
    customers = crud.get_customers(db, skip=skip, limit=limit)
    return customers

@router.get("/customers/{customer_id}", response_model=schemas.Customer)
def read_single_customer(customer_id: int, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретном клиенте по его ID.
    """
    db_customer = crud.get_customer(db, customer_id=customer_id)
    if db_customer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")
    return db_customer

# === Order Endpoints ===

@router.post("/orders/", response_model=schemas.Order, status_code=status.HTTP_201_CREATED)
def create_new_order(order: schemas.OrderCreate, db: Session = Depends(get_db)):
    """
    Создает новый заказ.
    Принимает ID клиента и список позиций заказа (`product_id`, `quantity`).
    Рассчитывает общую сумму и сохраняет заказ вместе с позициями.
    Возвращает созданный заказ с деталями клиента и позиций.
    """
    try:
        created_order = crud.create_order(db=db, order=order)
        return created_order
    except ValueError as e:
        # Перехватываем ошибки валидации из crud (клиент/товар не найден, неверное кол-во)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Общий обработчик на случай других ошибок
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating the order.")

@router.get("/orders/", response_model=List[schemas.Order])
def read_all_orders(
    skip: int = Query(0, ge=0, description="Number of orders to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders to return"),
    customer_id: Optional[int] = None, # Добавляем опциональный query parameter
    db: Session = Depends(get_db)
):
    """
    Получает список всех заказов с пагинацией.
    Позволяет фильтровать заказы по `customer_id`.
    Включает информацию о клиенте и позициях заказа (с товарами).
    """
    orders = crud.get_orders(db, skip=skip, limit=limit, customer_id=customer_id)
    return orders

@router.get("/orders/{order_id}", response_model=schemas.Order)
def read_single_order(order_id: int, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретном заказе по его ID.
    Включает информацию о клиенте и позициях заказа (с товарами).
    """
    db_order = crud.get_order(db, order_id=order_id)
    if db_order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
# === UserActivity Endpoints (заготовка) ===
# ... 