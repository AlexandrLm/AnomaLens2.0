import random
import time # Для генерации близких временных меток
from faker import Faker
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import traceback
import re
from datetime import datetime, timezone

# --- ДОБАВЛЯЕМ КОД ДЛЯ OLLAMA --- 
import requests
import json

# Адрес Ollama API по умолчанию
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Модель по умолчанию
OLLAMA_DEFAULT_MODEL = "gemma3:latest" # Используем другое имя, чтобы не конфликтовать с моделями БД

def generate_via_ollama(prompt: str, model: str = OLLAMA_DEFAULT_MODEL, ollama_url: str = OLLAMA_API_URL) -> Optional[str]:
    """
    Отправляет промпт в Ollama API и возвращает сгенерированный текст.
    Args:
        prompt: Текст промпта для модели.
        model: Имя модели Ollama.
        ollama_url: URL Ollama API.
    Returns:
        Сгенерированный текст или None в случае ошибки.
    """
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        print(f"[Ollama] Sending prompt ({model}): {prompt[:100]}...")
        response = requests.post(ollama_url, headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        result = response.json()
        generated_text = result.get('response')
        if generated_text:
            print(f"[Ollama] Response received: {generated_text[:100]}...")
            return generated_text.strip()
        else:
            print(f"[Ollama] Response did not contain 'response' field: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Error connecting to Ollama API at {ollama_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[Ollama] Error decoding JSON response: {e}")
        print(f"[Ollama] Raw response text: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"[Ollama] An unexpected error occurred: {e}")
        return None

def generate_product_details(category: str, model: str = OLLAMA_DEFAULT_MODEL, ollama_url: str = OLLAMA_API_URL) -> Optional[Tuple[str, str]]:
    """
    Генерирует название и описание продукта для заданной категории с помощью Ollama.
    Returns: Кортеж (название, описание) или None.
    """
    prompt = (
        f"Придумай реалистичное название и краткое (1-2 предложения) маркетинговое описание для нового товара "
        f"в интернет-магазине в категории '{category}'. "
        f"Ответ дай в формате JSON с ключами 'name' и 'description'. Например: "
        f'{{"name": "Название товара", "description": "Описание товара."}}'
    )
    generated_json_str = generate_via_ollama(prompt, model, ollama_url)
    if not generated_json_str: return None
    try:
        # --- Улучшенный поиск JSON в ответе --- \
        match = re.search(r'\{.*\}', generated_json_str, re.DOTALL) # Ищем JSON объект (DOTALL для многострочности)
        if match:
            json_str_cleaned = match.group(0)
            print(f"[Ollama] Found potential JSON: {json_str_cleaned[:150]}...")
        # ---------------------------------------
        # --- Старый метод (если re не найдет) - оставляем как фоллбэк?
        # json_start = generated_json_str.find('{')\n        # json_end = generated_json_str.rfind('}')\n        # if json_start != -1 and json_end != -1:\n        #     json_str_cleaned = generated_json_str[json_start : json_end + 1]\n        # --- Убираем старый метод --- \n        else:\n             print(f\"[Ollama] Could not find JSON object using regex in response: {generated_json_str}\")\n             return None\n\n        # --- Дальнейшая обработка json_str_cleaned --- \n        data = json.loads(json_str_cleaned)\n        name = data.get('name')\n
        data = json.loads(json_str_cleaned)
        name = data.get('name')
        description = data.get('description')
        if name and description and isinstance(name, str) and isinstance(description, str):
            print(f"[Ollama] Parsed product: Name='{name}', Desc='{description[:50]}...'")
            return name, description
        else:
            print(f"[Ollama] Parsed JSON invalid: {data}")
            return None
    except Exception as e:
        print(f"[Ollama] Error processing product details response: {e}")
        return None
# ----------------------------------

from .. import crud, schemas, models

fake = Faker('ru_RU') # Используем русскую локаль для имен, адресов и т.д.

# --- Конфигурация генерации ---
DEFAULT_CATEGORIES = ["Электроника", "Книги", "Одежда", "Дом и сад", "Спорт и отдых", "Детские товары", "Автотовары", "Продукты питания"]
ACTION_TYPES = ["view_page", "view_product", "search", "add_to_cart", "remove_from_cart", "login", "failed_login", "logout", "purchase", "update_profile"]
IP_ADDRESSES = [fake.ipv4() for _ in range(50)] # Пул IP-адресов
USER_AGENTS = [fake.user_agent() for _ in range(50)] # Пул User-Agent

# --- Обновляем схему --- 
class UserActivityCreateWithTimestamp(schemas.UserActivityCreate):
    timestamp: Optional[datetime] = None
# ---------------------

# --- Функции-помощники ---
def _get_random_or_none(item_list: List[Any], none_chance=0.1):
    if item_list and random.random() > none_chance:
        return random.choice(item_list)
    return None

def _create_activity(db: Session, activity_data: UserActivityCreateWithTimestamp) -> Optional[models.UserActivity]:
    """Вспомогательная функция для создания активности с обработкой ошибок и возможностью задать timestamp."""
    try:
        # Проверка customer_id 
        if activity_data.customer_id:
            db_customer = crud.get_customer(db, customer_id=activity_data.customer_id)
            if not db_customer:
                activity_data.customer_id = None

        # --- Передаем timestamp в модель --- 
        activity_dict = activity_data.model_dump(exclude_unset=True) # Не передаем None поля, если они не заданы
        db_activity = models.UserActivity(**activity_dict)
        # -----------------------------------
        
        db.add(db_activity)
        db.commit()
        db.refresh(db_activity)
        if db_activity.customer_id:
            db.refresh(db_activity, attribute_names=['customer'])
        return db_activity
    except Exception as e:
        print(f"Error creating activity: {e}. Data: {activity_data.model_dump()}")
        db.rollback() # Откатываем транзакцию при ошибке создания
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
        # Выбираем случайную категорию
        category = random.choice(categories)

        # --- ПОПЫТКА ГЕНЕРАЦИИ ЧЕРЕЗ OLLAMA --- 
        product_name_llm = None
        description_llm = None
        try:
            llm_details = generate_product_details(category.name)
            if llm_details:
                product_name_llm, description_llm = llm_details
            else:
                 print(f"[Simulator] Ollama failed for category '{category.name}'. Will use Faker.")
        except Exception as e:
            print(f"[Simulator] Error calling Ollama, will use Faker: {e}")
        # ---------------------------------------

        # --- Используем результат LLM или Faker --- 
        final_product_name = product_name_llm if product_name_llm else fake.catch_phrase()
        final_description = description_llm if description_llm else fake.text(max_nb_chars=150)
        # -----------------------------------------

        product_data = schemas.ProductCreate(
            name=final_product_name, # Используем итоговое имя
            description=final_description, # Используем итоговое описание
            price=round(random.uniform(5.0, 1000.0), 2),
            category_id=category.id
        )
        # Не проверяем на уникальность имени, т.к. могут быть похожие товары
        created = crud.create_product(db, product=product_data)
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

def generate_orders(
        db: Session, 
        count: int, 
        customers: List[models.Customer], 
        products: List[models.Product], 
        # --- Принимаем новые параметры --- 
        enable_anomaly: bool = True,
        anomaly_rate: float = 0.03 
        # -------------------------------
    ) -> List[models.Order]:
    created_orders = []
    num_anomalous = 0
    if not customers or not products:
        print("Cannot generate orders without customers and products.")
        return []

    # Оценим "нормальную" среднюю цену товара для аномалий суммы
    avg_price = sum(p.price for p in products) / len(products) if products else 1000.0

    for i in range(count):
        # --- УСЛОВНАЯ ГЕНЕРАЦИЯ АНОМАЛИИ --- 
        is_anomalous_order = enable_anomaly and random.random() < anomaly_rate
        # -----------------------------------
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
            items=[]
        )

        try:
            created = crud.create_order(
                db=db, 
                order=order_schema, 
                items_data=order_items_create
            )
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
            traceback.print_exc()

    print(f"Generated {len(created_orders)} orders. Attempted to generate {num_anomalous} anomalous amount orders.")
    return created_orders

def generate_activities(
        db: Session, 
        count: int, 
        customers: List[models.Customer], 
        products: List[models.Product], 
        # --- Принимаем новые параметры --- 
        enable_burst: bool = True,
        burst_rate: float = 0.02,
        enable_failed_login: bool = True,
        failed_login_rate: float = 0.01,
        # --- Принимаем даты --- 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
        # ---------------------
    ) -> List[models.UserActivity]:
    created_activities = []
    generated_count = 0
    num_anomalous_bursts = 0
    num_anomalous_logins = 0

    # Определяем общий шанс на любую аномалию (нужно для random.random() < total_anomaly_chance)
    # Учитываем флаги включения
    total_anomaly_chance = 0
    if enable_burst: total_anomaly_chance += burst_rate
    if enable_failed_login: total_anomaly_chance += failed_login_rate

    while generated_count < count:
        # --- УСЛОВНАЯ ГЕНЕРАЦИЯ АНОМАЛИИ --- 
        is_anomaly_attempt = random.random() < total_anomaly_chance
        anomaly_type = None
        if is_anomaly_attempt:
            # Выбираем тип аномалии пропорционально их долям
            rand_val = random.random() * total_anomaly_chance
            if enable_burst and rand_val < burst_rate:
                 anomaly_type = "burst"
            elif enable_failed_login and rand_val < burst_rate + failed_login_rate:
                 anomaly_type = "failed_login"
        # -------------------------------------

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

                activity_data = UserActivityCreateWithTimestamp(
                    customer_id=burst_customer.id if burst_customer else None,
                    session_id=burst_session,
                    action_type=burst_action,
                    details=details if details else None,
                    ip_address=burst_ip,
                    user_agent=burst_agent,
                    timestamp=None
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
                activity_data = UserActivityCreateWithTimestamp(
                    customer_id=None, # Неудачный логин обычно без customer_id
                    session_id=burst_session,
                    action_type="failed_login",
                    details={"attempted_email": target_email, "success": False},
                    ip_address=burst_ip,
                    user_agent=burst_agent,
                    timestamp=None
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
            
            # --- ГЕНЕРАЦИЯ ВРЕМЕНИ В ДИАПАЗОНЕ --- 
            activity_timestamp = None
            try:
                if start_date and end_date:
                    activity_timestamp = fake.date_time_between(start_date=start_date, end_date=end_date, tzinfo=timezone.utc)
                elif start_date:
                    # Если есть только начальная дата, генерируем до "сейчас"
                    activity_timestamp = fake.date_time_between(start_date=start_date, end_date="now", tzinfo=timezone.utc)
                # Если дат нет, timestamp будет установлен БД по умолчанию
            except Exception as date_exc:
                print(f"[Simulator] Error generating date between {start_date} and {end_date}: {date_exc}")
                # Оставляем None, БД поставит текущее время
            # -------------------------------------

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

            activity_data = UserActivityCreateWithTimestamp(
                customer_id=customer.id if customer else None,
                session_id=fake.uuid4() if random.random() < 0.9 else None, # 90% шанс иметь сессию
                action_type=action_type,
                details=details if details else None,
                ip_address=random.choice(IP_ADDRESSES),
                user_agent=random.choice(USER_AGENTS),
                timestamp=activity_timestamp 
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
    enable_order_amount_anomaly: bool,
    order_amount_anomaly_rate: float,
    enable_activity_burst_anomaly: bool,
    activity_burst_anomaly_rate: float,
    enable_failed_login_anomaly: bool,
    failed_login_anomaly_rate: float,
    # --- Принимаем даты --- 
    activity_start_date: Optional[datetime] = None,
    activity_end_date: Optional[datetime] = None
    # --------------------
) -> Dict[str, int]:
    """Генерирует все типы данных с заданной долей аномалий."""
    print(f"Starting data generation...") # Убрали старые rate из лога
    start_time = time.time()
    categories = generate_categories(db)
    products = generate_products(db, num_products, categories)
    # Генерируем немного больше клиентов, т.к. часть может быть пропущена из-за дубликатов email
    customers = generate_customers(db, int(num_customers * 1.1))
    # --- Передаем новые параметры в generate_orders --- 
    orders = generate_orders(db, num_orders, customers, products, 
                             enable_anomaly=enable_order_amount_anomaly, 
                             anomaly_rate=order_amount_anomaly_rate)
    # ------------------------------------------------
    # Убедимся, что есть клиенты и товары перед генерацией активностей
    if not customers: customers = crud.get_customers(db, limit=10) # Загрузим немного, если не сгенерировались
    if not products: products = crud.get_products(db, limit=10)
    # --- Передаем новые параметры в generate_activities --- 
    activities = generate_activities(db, num_activities, customers, products, 
                                     enable_burst=enable_activity_burst_anomaly, 
                                     burst_rate=activity_burst_anomaly_rate, 
                                     enable_failed_login=enable_failed_login_anomaly, 
                                     failed_login_rate=failed_login_anomaly_rate,
                                     # --- Передаем даты --- 
                                     start_date=activity_start_date,
                                     end_date=activity_end_date
                                     # --------------------
                                     )
    # -----------------------------------------------------
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