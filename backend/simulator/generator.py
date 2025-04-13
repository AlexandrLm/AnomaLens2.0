import random
import time # Для генерации близких временных меток
from faker import Faker
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from decimal import Decimal

from .. import crud, schemas, models

fake = Faker('ru_RU') # Используем русскую локаль для имен, адресов и т.д.

# --- Конфигурация генерации ---
DEFAULT_CATEGORIES = ["Электроника", "Книги", "Одежда", "Дом и сад", "Спорт и отдых", "Детские товары", "Автотовары", "Продукты питания"]
ACTION_TYPES = ["view_page", "view_product", "search", "add_to_cart", "remove_from_cart", "login", "failed_login", "logout", "purchase", "update_profile"]
IP_ADDRESSES = [fake.ipv4() for _ in range(50)] # Пул IP-адресов
USER_AGENTS = [fake.user_agent() for _ in range(50)] # Пул User-Agent

# --- Функции-помощники ---
def _get_random_or_none(item_list: List[Any], none_chance=0.1):
    if item_list and random.random() > none_chance:
        return random.choice(item_list)
    return None

def _create_activity(db: Session, activity_data: schemas.UserActivityCreate) -> Optional[models.UserActivity]:
    """Вспомогательная функция для создания активности с обработкой ошибок."""
    try:
        # Проверка customer_id перенесена сюда из crud для удобства генератора
        if activity_data.customer_id:
            db_customer = crud.get_customer(db, customer_id=activity_data.customer_id)
            if not db_customer:
                activity_data.customer_id = None # Если клиент не найден, делаем анонимной

        created = crud.create_user_activity(db, activity=activity_data)
        return created
    except Exception as e:
        print(f"Error creating activity: {e}. Data: {activity_data}")
        return None

# --- Основные функции генерации ---

def generate_categories(db: Session, count: int = len(DEFAULT_CATEGORIES)) -> List[models.Category]:
    created_categories = []
    for name in DEFAULT_CATEGORIES[:count]:
        existing = crud.get_category_by_name(db, name=name)
        if not existing:
            category_schema = schemas.CategoryCreate(name=name, description=fake.sentence())
            created = crud.create_category(db, category=category_schema)
            created_categories.append(created)
        else:
            created_categories.append(existing)
    print(f"Generated/found {len(created_categories)} categories.")
    return created_categories

def generate_products(db: Session, count: int, categories: List[models.Category]) -> List[models.Product]:
    created_products = []
    if not categories:
        print("Cannot generate products without categories.")
        return []
    for _ in range(count):
        category = random.choice(categories)
        product_schema = schemas.ProductCreate(
            name=fake.company() + " " + fake.word().capitalize(), # Более уникальные названия
            description=fake.text(max_nb_chars=200),
            price=round(random.uniform(100.0, 50000.0), 2),
            category_id=category.id
        )
        # Не проверяем на уникальность имени, т.к. могут быть похожие товары
        created = crud.create_product(db, product=product_schema)
        created_products.append(created)
    print(f"Generated {len(created_products)} products.")
    return created_products

def generate_customers(db: Session, count: int) -> List[models.Customer]:
    created_customers = []
    for _ in range(count):
        email = fake.unique.email()
        customer_schema = schemas.CustomerCreate(
            email=email,
            first_name=fake.first_name(),
            last_name=fake.last_name()
        )
        existing = crud.get_customer_by_email(db, email=email)
        if not existing:
             created = crud.create_customer(db, customer=customer_schema)
             created_customers.append(created)
        # Если существует, просто пропускаем
    print(f"Generated {len(created_customers)} customers.")
    return created_customers

def generate_orders(db: Session, count: int, customers: List[models.Customer], products: List[models.Product], anomaly_rate: float = 0.05) -> List[models.Order]:
    created_orders = []
    num_anomalous = 0
    if not customers or not products:
        print("Cannot generate orders without customers and products.")
        return []

    # Оценим "нормальную" среднюю цену товара для аномалий суммы
    avg_price = sum(p.price for p in products) / len(products) if products else 1000.0

    for i in range(count):
        is_anomalous_order = random.random() < anomaly_rate
        customer = random.choice(customers)
        num_items = random.randint(1, 5)
        order_items_create = []
        product_selection = random.sample(products, min(num_items, len(products)))

        calculated_total = Decimal("0.0")

        for product in product_selection:
             item_quantity = random.randint(1, 3)
             item_price = Decimal(str(product.price))
             calculated_total += item_price * item_quantity
             order_items_create.append(schemas.OrderItemCreate(
                 product_id=product.id,
                 quantity=item_quantity
             ))

        if not order_items_create:
             continue

        # Аномалия суммы заказа
        final_total = calculated_total
        if is_anomalous_order:
             num_anomalous += 1
             if random.random() < 0.7: # Аномально большая сумма (70% аномальных)
                 final_total = calculated_total * Decimal(random.uniform(10, 100))
                 print(f"Generating anomalous HIGH amount order for customer {customer.id}. Original: {calculated_total:.2f}, Anomalous: {final_total:.2f}")
             else: # Аномально маленькая сумма
                 final_total = Decimal(random.uniform(0.1, 10.0))
                 print(f"Generating anomalous LOW amount order for customer {customer.id}. Original: {calculated_total:.2f}, Anomalous: {final_total:.2f}")


        order_schema = schemas.OrderCreate(
            customer_id=customer.id,
            items=order_items_create
        )

        try:
            # Передаем total_amount в CRUD функцию, если она это поддерживает
            # Если нет, CRUD должен сам считать. Но для симуляции аномалии,
            # нам НУЖНО передать аномальную сумму. Доработаем CRUD позже, если нужно.
            # Пока создаем заказ стандартно, а потом ОБНОВИМ сумму, если аномальная.
            created = crud.create_order(db, order=order_schema)
            if is_anomalous_order and created:
                created.total_amount = float(final_total) # Обновляем сумму на аномальную
                db.add(created)
                db.commit()
                db.refresh(created)
            if created:
                 created_orders.append(created)

        except ValueError as e:
            print(f"Skipped order creation due to error: {e}")
        except Exception as e:
            print(f"Unexpected error during order creation: {e}")

    print(f"Generated {len(created_orders)} orders. Attempted to generate {num_anomalous} anomalous amount orders.")
    return created_orders

def generate_activities(db: Session, count: int, customers: List[models.Customer], products: List[models.Product], anomaly_rate: float = 0.05) -> List[models.UserActivity]:
    created_activities = []
    generated_count = 0
    num_anomalous_bursts = 0
    num_anomalous_logins = 0

    while generated_count < count:
        is_anomaly_attempt = random.random() < anomaly_rate
        anomaly_type = None
        if is_anomaly_attempt:
            anomaly_type = random.choice(["burst", "failed_login"])

        # --- Генерация Аномалии: Пачка действий (Burst) ---
        if anomaly_type == "burst":
            num_anomalous_bursts += 1
            burst_size = random.randint(5, 15) # Кол-во одинаковых действий
            burst_action = random.choice(["view_product", "add_to_cart", "search"]) # Тип действия для пачки
            burst_ip = random.choice(IP_ADDRESSES)
            burst_agent = random.choice(USER_AGENTS)
            burst_session = fake.uuid4()
            # Клиент может быть анонимным или зарегистрированным
            burst_customer = _get_random_or_none(customers, none_chance=0.5)
            burst_product = _get_random_or_none(products) if burst_action != "search" else None
            print(f"Generating anomalous BURST: {burst_size} x '{burst_action}' from IP {burst_ip}")

            for i in range(burst_size):
                if generated_count >= count: break
                details = {}
                if burst_action == "view_product" and burst_product:
                    details["product_id"] = burst_product.id
                elif burst_action == "add_to_cart" and burst_product:
                    details["product_id"] = burst_product.id
                    details["quantity"] = 1
                elif burst_action == "search":
                    details["query"] = fake.word()

                activity_data = schemas.UserActivityCreate(
                    customer_id=burst_customer.id if burst_customer else None,
                    session_id=burst_session,
                    action_type=burst_action,
                    details=details if details else None,
                    ip_address=burst_ip,
                    user_agent=burst_agent
                    # timestamp будет текущим при создании в БД, что близко друг к другу
                )
                created = _create_activity(db, activity_data)
                if created: created_activities.append(created)
                generated_count += 1
                time.sleep(0.01) # Небольшая пауза для имитации разницы времени

        # --- Генерация Аномалии: Неудачные Логины ---
        elif anomaly_type == "failed_login":
            num_anomalous_logins += 1
            burst_size = random.randint(5, 20) # Кол-во попыток
            burst_ip = random.choice(IP_ADDRESSES)
            burst_agent = random.choice(USER_AGENTS)
            burst_session = fake.uuid4()
            target_email = fake.email() # Email, который пытаются подобрать
            print(f"Generating anomalous FAILED LOGINS: {burst_size} attempts for '{target_email}' from IP {burst_ip}")

            for i in range(burst_size):
                if generated_count >= count: break
                activity_data = schemas.UserActivityCreate(
                    customer_id=None, # Неудачный логин обычно без customer_id
                    session_id=burst_session,
                    action_type="failed_login",
                    details={"attempted_email": target_email, "success": False},
                    ip_address=burst_ip,
                    user_agent=burst_agent
                )
                created = _create_activity(db, activity_data)
                if created: created_activities.append(created)
                generated_count += 1
                time.sleep(0.02) # Небольшая пауза

        # --- Генерация Нормальной Активности ---
        else:
            customer = _get_random_or_none(customers)
            product = _get_random_or_none(products)
            action_type = random.choice([act for act in ACTION_TYPES if act != "failed_login"]) # Исключаем аномальный тип
            details = {}

            if action_type == "view_product" and product:
                details["product_id"] = product.id
                details["product_name"] = product.name
            elif action_type == "search":
                details["query"] = fake.word()
            elif action_type == "add_to_cart" and product:
                details["product_id"] = product.id
                details["quantity"] = random.randint(1, 2)
            elif action_type == "remove_from_cart" and product:
                 details["product_id"] = product.id
            elif action_type == "view_page":
                 details["url"] = "/" + fake.uri_path()
            elif action_type == "login" and customer:
                 details["success"] = True
            elif action_type == "purchase":
                 details["order_total"] = round(random.uniform(500.0, 10000.0), 2)

            activity_data = schemas.UserActivityCreate(
                customer_id=customer.id if customer else None,
                session_id=fake.uuid4() if random.random() < 0.9 else None, # 90% шанс иметь сессию
                action_type=action_type,
                details=details if details else None,
                ip_address=random.choice(IP_ADDRESSES),
                user_agent=random.choice(USER_AGENTS)
            )
            created = _create_activity(db, activity_data)
            if created: created_activities.append(created)
            generated_count += 1

    print(f"Generated {len(created_activities)} activities (target was {count}). "
          f"Attempted {num_anomalous_bursts} anomalous bursts, "
          f"{num_anomalous_logins} anomalous login attempt sequences.")
    return created_activities

def generate_all_data(
    db: Session,
    num_customers: int,
    num_products: int,
    num_orders: int,
    num_activities: int,
    order_anomaly_rate: float = 0.05, # Доля аномальных заказов
    activity_anomaly_rate: float = 0.05 # Доля попыток создать аномальную активность/пачку
) -> Dict[str, int]:
    """Генерирует все типы данных с заданной долей аномалий."""
    print(f"Starting data generation with Order Anomaly Rate: {order_anomaly_rate*100:.1f}%, Activity Anomaly Rate: {activity_anomaly_rate*100:.1f}%")
    start_time = time.time()
    categories = generate_categories(db)
    products = generate_products(db, num_products, categories)
    # Генерируем немного больше клиентов, т.к. часть может быть пропущена из-за дубликатов email
    customers = generate_customers(db, int(num_customers * 1.1))
    orders = generate_orders(db, num_orders, customers, products, anomaly_rate=order_anomaly_rate)
    # Убедимся, что есть клиенты и товары перед генерацией активностей
    if not customers: customers = crud.get_customers(db, limit=10) # Загрузим немного, если не сгенерировались
    if not products: products = crud.get_products(db, limit=10)
    activities = generate_activities(db, num_activities, customers, products, anomaly_rate=activity_anomaly_rate)
    end_time = time.time()
    print(f"Data generation finished in {end_time - start_time:.2f} seconds.")

    # Возвращаем фактическое количество созданных/найденных объектов
    return {
        "categories": db.query(models.Category).count(), # Считаем фактическое кол-во в БД
        "products": db.query(models.Product).count(),
        "customers": db.query(models.Customer).count(),
        "orders": db.query(models.Order).count(),
        "activities": db.query(models.UserActivity).count() # Это может быть больше num_activities из-за пачек
    } 