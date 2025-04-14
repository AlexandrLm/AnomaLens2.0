import os
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional

# Импортируем модели, если они нужны внутри engineer_features
from .. import models # Относительный импорт из родительской директории

# --- Общие константы и настройки ---

SAVED_MODELS_DIR = "backend/ml_service/saved_models"
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)

# Список числовых признаков, которые будем рассчитывать/использовать
NUMERICAL_FEATURES = [
    'customer_id', 'timestamp_hour', 'timestamp_dayofweek',
    'details_order_total', 'details_quantity',
    'time_since_last_activity_ip', 'actions_in_last_5min_ip',
    'failed_logins_in_last_15min_ip'
]
# Категориальный признак для OHE
CATEGORICAL_FEATURE = 'action_type'

# --- Общая функция Feature Engineering --- 
def engineer_features(activities: List[models.UserActivity]) -> Optional[pd.DataFrame]:
    """
    Преобразует список активностей в DataFrame и рассчитывает признаки.
    Возвращает DataFrame с ID, timestamp и всеми рассчитанными признаками (числовыми и категориальными).
    (Код функции полностью скопирован из detector.py)
    """
    if not activities: return None
    data_list = [{"id": act.id, "customer_id": act.customer_id,
                  "timestamp": pd.to_datetime(act.timestamp), "action_type": act.action_type,
                  "ip_address": act.ip_address,
                  "details_order_total": float(act.details.get("order_total", 0)) if act.details else 0,
                  "details_quantity": int(act.details.get("quantity", 0)) if act.details else 0}
                 for act in activities]
    df = pd.DataFrame(data_list)
    if df.empty: return None

    print(f"Engineering features for {len(df)} activities...")
    df = df.sort_values(by="timestamp")
    # --- Уникальность timestamp ---
    if df['timestamp'].duplicated().any():
        print("Warning: Duplicate timestamps detected. Adding nanosecond offset.")
        df['timestamp'] = df['timestamp'] + pd.to_timedelta(np.arange(len(df)), unit='ns')
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    # --------------------------

    start_fe = time.time()

    print("\n--- Starting individual feature calculations ---")

    # Time since last activity
    try:
        print("Calculating 'time_since_last_activity_ip'...")
        df['time_since_last_activity_ip'] = df.groupby('ip_address')['timestamp'].diff().dt.total_seconds().fillna(0)
        print("'time_since_last_activity_ip' calculated.")
    except Exception as e:
        print(f"\n!!! ERROR calculating 'time_since_last_activity_ip': {e} !!!")
        try: df.info()
        except: print("Could not print df info on error.")
        return None

    # Rolling count
    try:
        print("Calculating 'actions_in_last_5min_ip'...")
        df_indexed = df.set_index('timestamp')
        if not df_indexed.index.is_unique: print("Warning: Timestamp index is not unique before rolling count!")
        rolling_count = df_indexed.groupby('ip_address')['id'].rolling('5min').count()
        df['actions_in_last_5min_ip'] = rolling_count.reset_index(level=0, drop=True).fillna(0)
        print("'actions_in_last_5min_ip' calculated.")
    except Exception as e:
        print(f"\n!!! ERROR calculating 'actions_in_last_5min_ip': {e} !!!")
        try: df.info()
        except: print("Could not print df info on error.")
        return None

    # Failed logins flag
    try:
        print("Calculating 'is_failed_login'...")
        df['is_failed_login'] = (df['action_type'] == 'failed_login').astype(int)
        print("'is_failed_login' calculated.")
    except Exception as e:
         print(f"\n!!! ERROR calculating 'is_failed_login': {e} !!!")
         try: df.info()
         except: print("Could not print df info on error.")
         return None

    # Rolling sum for failed logins
    try:
        print("Calculating 'failed_logins_in_last_15min_ip'...")
        df_indexed = df.set_index('timestamp')
        if not df_indexed.index.is_unique: print("Warning: Timestamp index is not unique before rolling sum!")
        print(f"  dtype of 'is_failed_login' before rolling sum: {df_indexed['is_failed_login'].dtype}")
        rolling_sum = df_indexed.groupby('ip_address')['is_failed_login'].rolling('15min').sum()
        df['failed_logins_in_last_15min_ip'] = rolling_sum.reset_index(level=0, drop=True).fillna(0)
        print("'failed_logins_in_last_15min_ip' calculated.")
    except Exception as e:
        print(f"\n!!! ERROR calculating 'failed_logins_in_last_15min_ip': {e} !!!")
        print("--- Error Details ---")
        try:
            print("df dtypes:")
            df.info()
            print("\ndf['is_failed_login'].value_counts():")
            print(df['is_failed_login'].value_counts())
        except: print("Could not print detailed error info.")
        print("--- End Error Details ---")
        return None

    # Timestamp features
    try:
        print("Calculating timestamp features...")
        df['timestamp_hour'] = df['timestamp'].dt.hour
        df['timestamp_dayofweek'] = df['timestamp'].dt.weekday
        print("Timestamp features calculated.")
    except Exception as e:
        print(f"\n!!! ERROR calculating timestamp features: {e} !!!")
        try: df.info()
        except: print("Could not print df info on error.")
        return None

    # Customer ID fillna
    try:
        print("Processing 'customer_id'...")
        df['customer_id'] = df['customer_id'].fillna(-1)
        print("'customer_id' processed.")
    except Exception as e:
        print(f"\n!!! ERROR processing 'customer_id': {e} !!!")
        try: df.info()
        except: print("Could not print df info on error.")
        return None

    print("--- Finished individual feature calculations ---\n")

    # --- Финальная очистка типов ---
    print("\n--- Starting final type conversion and cleanup ---")
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            # ... (весь код очистки типов без изменений)
            print(f"\nProcessing column: '{col}'")
            print(f"  Initial - dtype: {df[col].dtype}, non-nulls: {df[col].notna().sum()}/{len(df)}, head(5):\n{df[col].head().to_string()}")
            try:
                original_dtype_before_coerce = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count_after_coerce = df[col].isna().sum()
                print(f"  After pd.to_numeric(errors='coerce') - dtype: {df[col].dtype}, NaNs created: {nan_count_after_coerce} (from original type {original_dtype_before_coerce})")
            except Exception as e:
                 print(f"  ERROR during pd.to_numeric for {col}: {e}")
                 continue
            try:
                inf_mask = df[col].isin([np.inf, -np.inf])
                inf_count = inf_mask.sum()
                if inf_count > 0:
                    print(f"  Replacing {inf_count} infinite values in '{col}' with NaN")
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                print(f"  ERROR during infinity check/replace for {col}: {e}")
            try:
                nan_count_before_fillna = df[col].isna().sum()
                if nan_count_before_fillna > 0:
                    fill_value = 0.0
                    print(f"  Filling {nan_count_before_fillna} NaN values in '{col}' with {fill_value}")
                    df[col] = df[col].fillna(fill_value)
            except Exception as e:
                print(f"  ERROR during fillna for {col}: {e}")
            try:
                df[col] = df[col].astype(float)
                print(f"  Final - dtype: {df[col].dtype}, non-nulls: {df[col].notna().sum()}, head(5):\n{df[col].head().to_string()}")
            except Exception as e:
                print(f"  ERROR during final .astype(float) for {col}: {e}")
        else:
             print(f"Warning: Expected numerical feature '{col}' not found after engineering.")
    print("--- Finished final type conversion and cleanup ---\n")
    # ------------------------------------------------------------

    # Drop helper column
    df = df.drop(columns=['is_failed_login'], errors='ignore')

    print(f"Feature engineering took {time.time() - start_fe:.2f} seconds.")
    return df 