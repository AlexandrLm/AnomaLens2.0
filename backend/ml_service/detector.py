import pandas as pd
import joblib # Для сохранения/загрузки модели
import os
import time # Для расчета времени FE
import numpy as np # Для std, mean, abs
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN # <--- Импорт DBSCAN
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple, Set

from .. import crud, models
from .. import schemas # <-- Добавляем или проверяем импорт schemas
from ..schemas import DetectionParams

# --- Импортируем общие элементы --- 
from .common import SAVED_MODELS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURE, engineer_features
# --- Импортируем AutoencoderDetector --- 
from .autoencoder_detector import AutoencoderDetector

# --- Удаляем определения, перенесенные в common.py ---
# SAVED_MODELS_DIR = "backend/ml_service/saved_models"
# NUMERICAL_FEATURES = [...] 
# CATEGORICAL_FEATURE = 'action_type'
# def engineer_features(...): ...
# ----------------------------------------------------

# Путь для сохранения модели по умолчанию
DEFAULT_MODEL_DIR = "backend/ml_service/saved_models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "isolation_forest_detector.joblib")

# --- Общие настройки и функции ---
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)

def engineer_features(activities: List[models.UserActivity]) -> Optional[pd.DataFrame]:
    """
    Преобразует список активностей в DataFrame и рассчитывает признаки.
    Возвращает DataFrame с ID, timestamp и всеми рассчитанными признаками (числовыми и категориальными).
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
        # Выведем информацию о df перед ошибкой
        try: df.info()
        except: print("Could not print df info on error.")
        return None # Прерываем, если этот базовый признак не рассчитался

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
        df_indexed = df.set_index('timestamp') # Re-index
        if not df_indexed.index.is_unique: print("Warning: Timestamp index is not unique before rolling sum!")
        # --- Ключевой момент: Проверим dtype 'is_failed_login' перед sum ---
        print(f"  dtype of 'is_failed_login' before rolling sum: {df_indexed['is_failed_login'].dtype}")
        rolling_sum = df_indexed.groupby('ip_address')['is_failed_login'].rolling('15min').sum()
        df['failed_logins_in_last_15min_ip'] = rolling_sum.reset_index(level=0, drop=True).fillna(0)
        print("'failed_logins_in_last_15min_ip' calculated.")
    except Exception as e:
        print(f"\n!!! ERROR calculating 'failed_logins_in_last_15min_ip': {e} !!!")
        # Если ошибка здесь, выведем типы и сам столбец is_failed_login
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

    # --- Финальная очистка типов (оставляем) ---
    print("\n--- Starting final type conversion and cleanup ---") # New print
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            print(f"\nProcessing column: '{col}'") # New print
            # Check initial state
            print(f"  Initial - dtype: {df[col].dtype}, non-nulls: {df[col].notna().sum()}/{len(df)}, head(5):\n{df[col].head().to_string()}") # New print

            # 1. Coerce to numeric
            try:
                original_dtype_before_coerce = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count_after_coerce = df[col].isna().sum()
                print(f"  After pd.to_numeric(errors='coerce') - dtype: {df[col].dtype}, NaNs created: {nan_count_after_coerce} (from original type {original_dtype_before_coerce})") # New print
            except Exception as e:
                 print(f"  ERROR during pd.to_numeric for {col}: {e}")
                 continue # Skip to next column if coercion fails

            # 2. Handle infinities
            try:
                inf_mask = df[col].isin([np.inf, -np.inf])
                inf_count = inf_mask.sum()
                if inf_count > 0:
                    print(f"  Replacing {inf_count} infinite values in '{col}' with NaN") # New print
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                print(f"  ERROR during infinity check/replace for {col}: {e}")

            # 3. Fill NaN values
            try:
                nan_count_before_fillna = df[col].isna().sum()
                if nan_count_before_fillna > 0:
                    fill_value = 0.0 # Default fill value
                    print(f"  Filling {nan_count_before_fillna} NaN values in '{col}' with {fill_value}") # New print
                    df[col] = df[col].fillna(fill_value)
            except Exception as e:
                print(f"  ERROR during fillna for {col}: {e}")

            # 4. Final cast to float
            try:
                df[col] = df[col].astype(float)
                print(f"  Final - dtype: {df[col].dtype}, non-nulls: {df[col].notna().sum()}, head(5):\n{df[col].head().to_string()}") # New print
            except Exception as e:
                print(f"  ERROR during final .astype(float) for {col}: {e}")

        else:
             print(f"Warning: Expected numerical feature '{col}' not found after engineering.")
    print("--- Finished final type conversion and cleanup ---\n") # New print
    # ------------------------------------------------------------

    # Drop helper column
    df = df.drop(columns=['is_failed_login'], errors='ignore')

    print(f"Feature engineering took {time.time() - start_fe:.2f} seconds.")
    # УДАЛЕНО: Лог типов в конце engineer_features, так как он теперь внутри цикла
    # print(f"Final dtypes after engineer_features:\n{df.dtypes}") 
    return df

# --- Детектор: Isolation Forest ---
class IsolationForestDetector:
    DEFAULT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "iforest_detector.joblib")

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, contamination: float = 'auto', random_state: int = 42):
        self.model_path = model_path
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = [] # Будет включать числовые и OHE признаки
        self.ohe_categories: Dict[str, List[Any]] = {} # Для сохранения категорий OHE
        self._load_model()
        if self.model is None: self._initialize_new_model()

    def _initialize_new_model(self):
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state, n_estimators=100)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.ohe_categories = {}

    def _save_model(self):
        if self.model and self.scaler and self.feature_names and self.ohe_categories is not None:
            payload = {'model': self.model, 'scaler': self.scaler,
                       'feature_names': self.feature_names, 'ohe_categories': self.ohe_categories}
            try:
                print(f"Attempting to save IF model to: {self.model_path}")
                joblib.dump(payload, self.model_path)
                print(f"Successfully saved IF model to {self.model_path}")
            except Exception as e: print(f"Error saving IF model: {e}")
        else: print("Cannot save IF model: State is incomplete.")

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                payload = joblib.load(self.model_path)
                self.model = payload['model']; self.scaler = payload['scaler']
                self.feature_names = payload['feature_names']; self.ohe_categories = payload['ohe_categories']
                print(f"IF model loaded from {self.model_path}")
            except Exception as e: print(f"Error loading IF model: {e}. Init new."); self._initialize_new_model()
        else: print(f"IF model file not found at {self.model_path}."); self._initialize_new_model()

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Optional[pd.DataFrame]:
        """Выбирает признаки, кодирует категориальные, масштабирует числовые."""
        print("Preparing features for Isolation Forest...")
        # 1. One-Hot Encode категориальный признак
        ohe_df = pd.get_dummies(df[[CATEGORICAL_FEATURE]], prefix=CATEGORICAL_FEATURE, dummy_na=False) # Не создаем колонку для NaN

        if is_training:
            # Запоминаем категории, которые были при обучении
            self.ohe_categories[CATEGORICAL_FEATURE] = ohe_df.columns.tolist()
            print(f"OHE training categories for {CATEGORICAL_FEATURE}: {self.ohe_categories[CATEGORICAL_FEATURE]}")
        else:
            # Применяем OHE с учетом известных категорий
            known_categories = self.ohe_categories.get(CATEGORICAL_FEATURE, [])
            if not known_categories: print("Warning: OHE categories not loaded/trained."); return None
            # Добавляем недостающие колонки (если в новых данных нет какой-то категории) и убираем лишние
            ohe_df = ohe_df.reindex(columns=known_categories, fill_value=0)
            print(f"Applied OHE using known categories.")


        # 2. Выбираем и масштабируем числовые признаки
        num_features_available = [f for f in NUMERICAL_FEATURES if f in df.columns]
        if not num_features_available: print("No numerical features found."); return None
        numeric_df = df[num_features_available].copy().fillna(0)

        if is_training:
            print(f"Fitting scaler using numerical features: {num_features_available}")
            print(f"Numeric DF dtypes before scaling:\n{numeric_df.dtypes}")
            # --- Добавлено: Вывод head() для отладки ---
            print(f"Numeric DF head() before scaling:\n{numeric_df.head()}")
            # ------------------------------------------
            try:
                scaled_numeric = self.scaler.fit_transform(numeric_df)
                self.feature_names = num_features_available + self.ohe_categories.get(CATEGORICAL_FEATURE, [])
                scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=num_features_available, index=df.index)
            except Exception as e:
                print(f"Error during scaler fitting/transform: {e}") # Добавим лог ошибки здесь
                return None # Прерываем, если масштабирование не удалось
        else:
            if not self.feature_names or not self.scaler: print("Scaler/features not ready."); return None
            # Используем только те числовые признаки, что были при обучении
            expected_num_features = [f for f in self.feature_names if f in num_features_available]
            print(f"Scaling numerical features using existing scaler: {expected_num_features}")
            numeric_df = numeric_df.reindex(columns=expected_num_features, fill_value=0)
            scaled_numeric = self.scaler.transform(numeric_df)
            scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=expected_num_features, index=df.index)

        # 3. Объединяем масштабированные числовые и OHE признаки
        final_df = pd.concat([scaled_numeric_df, ohe_df], axis=1)

        # Убедимся, что порядок и состав колонок соответствует self.feature_names
        final_df = final_df.reindex(columns=self.feature_names, fill_value=0)

        print(f"Prepared {len(final_df.columns)} features in total. Final columns: {final_df.columns.tolist()}")
        return final_df

    def train(self, db: Session, entity_type: str, limit: int):
        """
        Обучает модель Isolation Forest на активностях пользователя.
        Теперь загружает данные внутри с использованием db и limit.
        """
        print("IF Training: Starting...")
        if entity_type != 'activity':
            print(f"IF Training: Skipped. Supports only 'activity', got '{entity_type}'.")
            return

        try:
            print(f"IF Training: Loading last {limit} activities...")
            activities = crud.get_user_activities(db, skip=0, limit=limit)
            if not activities:
                print("IF Training: No activities found to train on.")
                return
            print(f"IF Training: Loaded {len(activities)} activities.")

            df = engineer_features(activities)
            if df is None or df.empty:
                print("IF Training: DataFrame is empty after feature engineering.")
                return

            features_df = self._prepare_features(df, is_training=True)
            if features_df is None or features_df.empty:
                print("IF Training: No features to train on after preparation.")
                return

            print(f"IF Training: Fitting model on {len(features_df)} samples with {len(self.feature_names)} features...")
            self.model.fit(features_df)
            print("IF Training: Model fit complete.")
            self._save_model() # Сохраняем обученную модель и скейлер

        except Exception as e:
            print(f"IF Training Error: {e}")
            # Возможно, стоит сбросить модель в исходное состояние при ошибке?
            # self._initialize_new_model()

    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии в новых данных, используя обученную модель.
        Возвращает список словарей с информацией об аномалиях, включая severity.
        """
        print("IF Detector: Starting detection...")
        if not self.model or not self.scaler or not self.feature_names:
            print("IF Detector: Model or scaler or features not loaded/trained. Cannot detect.")
            return []

        activities = crud.get_user_activities(db, limit=limit)
        if not activities:
            print("IF Detector: No activities found for detection.")
            return []

        df = engineer_features(activities)
        if df is None or df.empty:
            print("IF Detector: Feature engineering returned empty DataFrame.")
            return []

        # Подготовка признаков (OHE + Масштабирование) как при обучении
        X = self._prepare_features(df, is_training=False)
        if X is None:
            print("IF Detector: Feature preparation failed during detection.")
            return []

        # --- Выполняем предсказание ---
        try:
            # scores: чем меньше, тем более аномально (ближе к -1)
            # predict: 1 - норма, -1 - аномалия (согласно contamination)
            scores = self.model.decision_function(X) # Anomaly scores
            predictions = self.model.predict(X) # Anomaly labels (1 or -1)
            print(f"IF Detector: Prediction completed. Found {sum(predictions == -1)} potential anomalies based on contamination.")
        except Exception as e:
            print(f"IF Detector: Error during prediction: {e}")
            return []

        # --- Формируем результат ---
        anomalies = []
        anomaly_indices = np.where(predictions == -1)[0] # Индексы аномалий

        for idx in anomaly_indices:
            activity_id = df.iloc[idx]['id']
            current_score = scores[idx]

            # --- ОПРЕДЕЛЕНИЕ СЕРЬЕЗНОСТИ ---
            severity = "Low" # Default
            if current_score <= -0.1:
                 severity = "High"
            elif current_score <= 0.05:
                 severity = "Medium"
            # -----------------------------

            # --- УЛУЧШЕНО: Формирование Reason --- 
            reason_str = (
                f"Point easily isolated by the model (score: {current_score:.4f}), indicating it differs from the majority based on its features. "
                f"Scores closer to -1 are more anomalous, closer to 1 are normal."
            )
            # ------------------------------------

            anomaly_info = {
                "activity_id": activity_id,
                "reason": reason_str, # <--- Используем улучшенный reason
                "details": {"anomaly_score": float(current_score)},
                "severity": severity # <--- Добавляем серьезность
            }
            anomalies.append(anomaly_info)

        print(f"IF Detector: Formatted {len(anomalies)} anomalies.")
        return anomalies

# --- Детектор: Статистический (Z-score) ---
class StatisticalDetector:
    """
    Обнаруживает аномалии на основе статистических показателей (Z-Score).
    """
    STATS_FILE_TEMPLATE = os.path.join(SAVED_MODELS_DIR, "{entity_type}_stats.joblib")

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        # Словарь для хранения статистики {entity_type: {column: {'mean': ..., 'std': ...}}}
        self.stats = {}
        self._load_all_stats() # Загружаем всю имеющуюся статистику при инициализации

    def _get_stats_path(self, entity_type: str) -> str:
        """Возвращает путь к файлу статистики для данного типа сущности."""
        return self.STATS_FILE_TEMPLATE.format(entity_type=entity_type)

    def _save_stats(self, entity_type: str):
        """Сохраняет статистику для указанного типа сущности."""
        if entity_type in self.stats:
            path = self._get_stats_path(entity_type)
            try:
                joblib.dump(self.stats[entity_type], path)
                print(f"Saved stats for '{entity_type}' to {path}")
            except Exception as e:
                print(f"Error saving stats for '{entity_type}': {e}")

    def _load_stats(self, entity_type: str):
        """Загружает статистику для указанного типа сущности."""
        path = self._get_stats_path(entity_type)
        if os.path.exists(path):
            try:
                self.stats[entity_type] = joblib.load(path)
                print(f"Loaded stats for '{entity_type}' from {path}")
                # Добавим проверку, что все нужные ключи ('mean', 'std') на месте
                for col, col_stats in self.stats[entity_type].items():
                    if 'mean' not in col_stats or 'std' not in col_stats:
                        print(f"Warning: Stats for column '{col}' in '{entity_type}' are incomplete. Missing 'mean' or 'std'.")
                        # Можно удалить неполную статистику
                        # del self.stats[entity_type][col]
            except Exception as e:
                print(f"Error loading stats for '{entity_type}': {e}. Initializing empty stats.")
                self.stats[entity_type] = {}
        else:
            print(f"No stats file found for '{entity_type}'. Initializing empty stats.")
            self.stats[entity_type] = {}

    def _load_all_stats(self):
        """Загружает статистику для всех известных типов сущностей."""
        # Пока жестко зададим типы, можно расширить
        for entity_type in ['activity', 'order']:
            self._load_stats(entity_type)

    def train(self, db: Session, entity_type: str, limit: int = 10000):
        """
        Рассчитывает и сохраняет статистику (среднее и стандартное отклонение)
        для числовых признаков указанного типа сущности.
        """
        print(f"Starting statistical training for '{entity_type}'...")
        data = None
        feature_columns = []

        if entity_type == 'activity':
            activities = crud.get_user_activities(db, skip=0, limit=limit)
            if not activities:
                print("No activity data found for statistical training.")
                return
            df_features = engineer_features(activities)
            if df_features is None or df_features.empty:
                print("Feature engineering failed or returned empty DataFrame for activities.")
                return
            data = df_features
            # Используем числовые признаки, определенные глобально или специфичные для активностей
            feature_columns = [col for col in NUMERICAL_FEATURES if col in df_features.columns]

        elif entity_type == 'order':
            orders = crud.get_orders(db, skip=0, limit=limit)
            if not orders:
                print("No order data found for statistical training.")
                return
            # Преобразуем заказы в DataFrame
            order_data = [{
                'id': order.id,
                'customer_id': order.customer_id,
                'created_at': pd.to_datetime(order.created_at),
                'total_amount': order.total_amount,
                'item_count': len(order.items) # Пример доп. признака для заказов
            } for order in orders]
            df_orders = pd.DataFrame(order_data)
            if df_orders.empty:
                print("DataFrame creation failed or returned empty for orders.")
                return
            data = df_orders
            # Определяем числовые признаки для заказов
            feature_columns = ['total_amount', 'item_count']
            # Убедимся, что эти колонки существуют
            feature_columns = [col for col in feature_columns if col in data.columns]
        else:
            print(f"Unsupported entity_type '{entity_type}' for statistical training.")
            return

        if data is None or data.empty:
            print(f"No data available for statistical training for '{entity_type}'.")
            return

        if not feature_columns:
            print(f"No numerical feature columns found for '{entity_type}' in the data.")
            return

        print(f"Calculating stats for columns: {feature_columns}")
        entity_stats = {}
        try:
            for col in feature_columns:
                # --- Добавляем проверку на наличие числовых данных перед агрегацией ---
                if pd.api.types.is_numeric_dtype(data[col]):
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    # Добавляем проверку на NaN/Inf в статистике
                    if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                        print(f"  Warning: Could not calculate valid stats for '{col}' (mean={mean_val}, std={std_val}). Skipping.")
                        continue
                    entity_stats[col] = {'mean': mean_val, 'std': std_val}
                    print(f"  Calculated stats for '{col}': mean={mean_val:.4f}, std={std_val:.4f}")
                else:
                    print(f"  Warning: Column '{col}' is not numeric. Skipping stats calculation.")
            # -------------------------------------------------------------------------

            self.stats[entity_type] = entity_stats
            self._save_stats(entity_type)
            print(f"Successfully completed statistical training for '{entity_type}'.")
        except Exception as e:
            print(f"Error during statistical calculation for '{entity_type}': {e}")
            # Выведем информацию о типах данных при ошибке
            print("Data types causing potential issue:")
            print(data[feature_columns].dtypes)


    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии на основе Z-score.
        Возвращает список словарей с информацией об аномалиях, включая severity.
        """
        print(f"Statistical Detector ({entity_type}): Starting detection...")
        df = None
        feature_to_check = None
        id_field = 'id' # Default id field

        if entity_type == 'activity':
            data = crud.get_user_activities(db, limit=limit)
            # Define the primary feature to check for anomalies in activities
            feature_to_check = "time_since_last_activity_ip" # Or another key engineered feature
            if not data:
                 print(f"Statistical Detector ({entity_type}): No data found."); return []
            # Feature Engineering для активностей
            df = engineer_features(data)
            if df is None or df.empty:
                print(f"Statistical Detector ({entity_type}): Feature eng. failed or returned empty."); return []

        elif entity_type == 'order':
            data = crud.get_orders(db, limit=limit)
            # Define the primary feature to check for anomalies in orders
            feature_to_check = "total_amount"
            if not data:
                 print(f"Statistical Detector ({entity_type}): No data found."); return []
            # Преобразуем заказы в DataFrame
            order_list = [o.__dict__ for o in data] # Use __dict__ or a custom method if needed
            df = pd.DataFrame(order_list)
            # Ensure required columns exist after potential __dict__ conversion issues
            if 'id' in df.columns: df.rename(columns={'id': 'order_id'}, inplace=True) # Example rename if needed
            id_field = 'order_id' # ID field for orders
            if df.empty:
                 print(f"Statistical Detector ({entity_type}): DataFrame empty."); return []
        else:
            raise ValueError(f"Unsupported entity_type for StatisticalDetector: {entity_type}")

        # --- ИСПРАВЛЕНО: Загрузка и использование статистики ---
        # 1. Убедимся, что статистика для entity_type загружена (или пытаемся загрузить снова)
        if entity_type not in self.stats:
            print(f"Stats for {entity_type} not pre-loaded. Attempting load now.")
            self._load_stats(entity_type) # Load stats for the specific entity type

        # 2. Проверяем наличие статистики и нужной колонки
        if entity_type not in self.stats:
             print(f"Statistical Detector ({entity_type}): Stats failed to load. Cannot detect.")
             return []
        if feature_to_check not in self.stats[entity_type]:
            print(f"Statistical Detector ({entity_type}): Stats not available for feature '{feature_to_check}'. Cannot detect.")
            return []
        if feature_to_check not in df.columns:
             print(f"Statistical Detector ({entity_type}): Feature '{feature_to_check}' not found in DataFrame after processing. Cannot detect.")
             return []

        # 3. Получаем mean и std из загруженной статистики
        stats_for_feature = self.stats[entity_type][feature_to_check]
        mean = stats_for_feature.get('mean')
        std = stats_for_feature.get('std')

        # 4. Проверяем валидность mean и std
        if mean is None or std is None or std == 0 or pd.isna(mean) or pd.isna(std):
             print(f"Statistical Detector ({entity_type}): Invalid stats (mean={mean}, std={std}) for '{feature_to_check}'. Cannot detect.")
             return []
        # -------------------------------------------------------

        print(f"Statistical Detector ({entity_type}): Using stats for '{feature_to_check}': mean={mean:.2f}, std={std:.2f}, z_threshold={self.z_threshold}")

        anomalies = []
        try:
            # Убедимся, что колонка числовая перед расчетом
            if not pd.api.types.is_numeric_dtype(df[feature_to_check]):
                print(f"Warning: Column '{feature_to_check}' is not numeric. Attempting conversion.")
                df[feature_to_check] = pd.to_numeric(df[feature_to_check], errors='coerce')
                df.dropna(subset=[feature_to_check], inplace=True) # Удаляем строки, где конвертация не удалась

            if df.empty:
                print(f"Statistical Detector ({entity_type}): DataFrame is empty after numeric conversion/dropna for '{feature_to_check}'.")
                return []

            z_scores = (df[feature_to_check] - mean) / std
            potential_anomaly_indices = df.index[np.abs(z_scores) >= self.z_threshold]

            print(f"Statistical Detector ({entity_type}): Found {len(potential_anomaly_indices)} potential anomalies based on z-score >= {self.z_threshold}.")
            # --- ДОБАВЛЕН ЛОГ: Вывод распределения Z-score перед фильтрацией ---
            print(f"Statistical Detector ({entity_type}): Z-score distribution summary:\n{z_scores.describe()}")
            # ------------------------------------------------------------------

            for idx in potential_anomaly_indices:
                row = df.loc[idx]
                entity_id = row[id_field]
                current_z = z_scores.loc[idx]
                current_value = row[feature_to_check]

                # Определение Severity (логика без изменений)
                severity = "Medium"
                if abs(current_z) >= self.z_threshold * 1.5:
                    severity = "High"
                elif abs(current_z) >= self.z_threshold:
                    severity = "Medium"

                # --- УЛУЧШЕНО: Формирование Reason --- 
                reason_str = (
                    f"Value of '{feature_to_check}' ({current_value:.4f}) deviates significantly from the mean ({mean:.4f}). "
                    f"Z-score = {current_z:.2f}, which is {('above' if current_z > 0 else 'below')} the threshold of {self.z_threshold}. "
                    f"(std={std:.4f})"
                )
                # ------------------------------------

                anomaly_info = {
                    "entity_type": entity_type,
                    "entity_id": int(entity_id),
                    "reason": reason_str, # <--- Используем улучшенный reason
                    "details": {
                        "feature": feature_to_check,
                        "value": float(current_value),
                        "mean": float(mean),
                        "std": float(std),
                        "z_score": float(current_z)
                    },
                    "severity": severity,
                    "anomaly_score": float(abs(current_z))
                }
                if entity_type == 'activity': anomaly_info['activity_id'] = int(entity_id)
                elif entity_type == 'order': anomaly_info['order_id'] = int(entity_id)

                anomalies.append(anomaly_info)

        except KeyError as e:
            print(f"Statistical Detector ({entity_type}): KeyError during Z-score calculation or row access: {e}. Check if '{feature_to_check}' or '{id_field}' exists.")
            # print(f"DataFrame columns: {df.columns.tolist()}") # Раскомментировать для отладки
            return []
        except Exception as e:
            print(f"Statistical Detector ({entity_type}): Error during Z-score calculation/anomaly formatting: {e}")
            # import traceback
            # traceback.print_exc() # Раскомментировать для детальной отладки
            return []

        print(f"Statistical Detector ({entity_type}): Formatted {len(anomalies)} anomalies.")
        return anomalies

# --- Детектор: DBSCAN ---
class DbscanDetector:
    """Обнаруживает аномалии с помощью DBSCAN (шумовые точки)."""

    # ИСПРАВЛЕНО: Принимаем eps и min_samples в __init__
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler() # Масштабировщик для DBSCAN
        # Атрибуты для хранения информации об обучении (хотя DBSCAN не "обучается")
        self.feature_names: List[str] = []
        self.ohe_categories: Dict[str, List[Any]] = {}
        print(f"Creating DbscanDetector with eps={self.eps}, min_samples={self.min_samples}") # Лог параметров

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Готовит признаки для DBSCAN: OHE + Scaling."""
        print("Preparing features for DBSCAN...")
        # 1. OHE для категориального признака
        try:
            ohe_df = pd.get_dummies(df[[CATEGORICAL_FEATURE]], prefix=CATEGORICAL_FEATURE, dummy_na=False)
            self.ohe_categories[CATEGORICAL_FEATURE] = ohe_df.columns.tolist() # Просто запоминаем использованные
            print(f"  Applied One-Hot Encoding for '{CATEGORICAL_FEATURE}'. New columns: {ohe_df.columns.tolist()}")
        except KeyError:
             print(f"Warning: Categorical feature '{CATEGORICAL_FEATURE}' not found for OHE.")
             ohe_df = pd.DataFrame(index=df.index) # Создаем пустой DF, если категории нет
        except Exception as e:
            print(f"Error during OHE: {e}")
            return None

        # 2. Выбор и масштабирование числовых признаков
        num_features_available = [f for f in NUMERICAL_FEATURES if f in df.columns]
        if not num_features_available: print("No numerical features found for scaling."); return None
        numeric_df = df[num_features_available].copy().fillna(0) # Заполняем NaN перед масштабированием

        try:
            scaled_numeric = self.scaler.fit_transform(numeric_df) # Обучаем и применяем scaler
            scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=num_features_available, index=df.index)
            self.feature_names = num_features_available + self.ohe_categories.get(CATEGORICAL_FEATURE, []) # Сохраняем имена всех признаков
        except Exception as e:
            print(f"Error during scaling: {e}")
            return None

        # 3. Объединение признаков
        final_df = pd.concat([scaled_numeric_df, ohe_df], axis=1)
        print(f"Prepared {len(final_df.columns)} features for DBSCAN (before scaling).")
        return final_df

    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print(f"DBSCAN Detector: Starting detection for {entity_type}...")
        if entity_type != 'activity':
             print(f"DBSCAN currently only supports 'activity'. Skipping for '{entity_type}'.")
             return []

        activities = crud.get_user_activities(db, limit=limit)
        if not activities: print("DBSCAN Detector: No activity data found."); return []

        df_engineered = engineer_features(activities)
        if df_engineered is None or df_engineered.empty: print("DBSCAN: Feature eng. failed."); return []

        # Используем OHE+Scaling специфичный для DBSCAN
        df_prepared = self._prepare_features(df_engineered)
        if df_prepared is None or df_prepared.empty: print("DBSCAN: Feature prep. failed."); return []

        print(f"DBSCAN Detector: Running DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        try:
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1) # Используем все ядра
            clusters = dbscan.fit_predict(df_prepared)
        except Exception as e:
            print(f"Error during DBSCAN fitting: {e}")
            return []

        noise_indices = df_engineered.index[clusters == -1]
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        print(f"DBSCAN Detector: Found {num_clusters} clusters and {len(noise_indices)} noise points.")

        anomalies = []
        for idx in noise_indices:
            original_activity = df_engineered.loc[idx]
            activity_id = original_activity.get('id') # Получаем ID из оригинального (инженерного) DF
            if activity_id is None: continue # Пропускаем, если нет ID

            # --- УЛУЧШЕНО: Формирование Reason --- 
            reason_str = (
                f"Point classified as noise (cluster=-1) by DBSCAN. "
                f"It doesn't belong to any dense cluster based on current settings (eps={self.eps}, min_samples={self.min_samples}) when analyzing its features."
            )
            # ------------------------------------

            anomaly_info = {
                "activity_id": int(activity_id),
                "reason": reason_str, # <--- Используем улучшенный reason
                "details": {"cluster": -1},
                "severity": "Medium" # DBSCAN шум считаем Medium
            }
            anomalies.append(anomaly_info)

        print(f"DBSCAN Detector: Formatted {len(anomalies)} noise points as anomalies.")
        return anomalies

# --- Фабрика детекторов (get_detector) ---
def get_detector(algorithm: str, params: schemas.DetectionParams):
    """Фабричная функция для создания экземпляров детекторов."""
    if algorithm == 'isolation_forest':
        # IF не использует параметры из DetectionParams напрямую в __init__ (пока)
        return IsolationForestDetector()
    elif algorithm == 'statistical_zscore':
        # Передаем z_threshold из параметров запроса
        return StatisticalDetector(z_threshold=params.z_threshold)
    elif algorithm == 'dbscan':
        # ИСПРАВЛЕНО: Передаем eps и min_samples из параметров запроса
        return DbscanDetector(eps=params.dbscan_eps, min_samples=params.dbscan_min_samples)
    elif algorithm == 'autoencoder':
         # Используем импортированный класс
         return AutoencoderDetector() # Correctly references the imported class
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")



# --- Главная функция для запуска детекции --- (устарела, логика перенесена в API)
# def run_detection_and_save(db: Session, detector: Any, entity_type: str, limit: int):
#     # ... (код из старой версии)

# --- Главная функция для запуска обучения --- (устарела, логика перенесена в API)
# def train_models(db: Session, detectors: List[Any], entity_type: str, limit: int):
#     # ... (код из старой версии) 