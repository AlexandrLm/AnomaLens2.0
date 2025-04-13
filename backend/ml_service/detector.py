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

# Путь для сохранения модели по умолчанию
DEFAULT_MODEL_DIR = "backend/ml_service/saved_models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "isolation_forest_detector.joblib")

# --- Общие настройки и функции ---
SAVED_MODELS_DIR = "backend/ml_service/saved_models"
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)

# Список числовых признаков, которые будем рассчитывать/использовать
NUMERICAL_FEATURES = [
    'customer_id', 'timestamp_hour', 'timestamp_dayofweek',
    'details_order_total', 'details_quantity',
    'time_since_last_activity_ip', 'actions_in_last_5min_ip',
    'failed_logins_in_last_15min_ip'#, 'distinct_actions_per_ip_5min' # <-- Временно уберем
]
# Категориальный признак для OHE
CATEGORICAL_FEATURE = 'action_type'

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
    DEFAULT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "iforest_detector.joblib") # Изменил имя файла

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
        Обнаруживает аномалии с использованием обученной модели Isolation Forest.
        Возвращает список словарей с информацией об аномалиях, включая severity.
        """
        print(f"Starting Isolation Forest detection for '{entity_type}'...")
        if self.model is None:
            print("Isolation Forest model not loaded. Cannot detect.")
            return []
        if entity_type != 'activity':
            print(f"Isolation Forest detector currently only supports 'activity'. Skipping for '{entity_type}'.")
            return []

        activities = crud.get_user_activities(db, skip=0, limit=limit)
        if not activities:
            print("No activity data found for Isolation Forest detection.")
            return []

        df_engineered = engineer_features(activities)
        if df_engineered is None or df_engineered.empty:
            print("Feature engineering failed or returned empty DataFrame.")
            return []

        df_features = self._prepare_features(df_engineered)
        if df_features is None or df_features.empty:
            print("Feature preparation failed or returned empty DataFrame.")
            return []

        # Убедимся, что колонки совпадают с теми, на которых обучались
        if set(self.feature_names) != set(df_features.columns):
            print("Warning: Feature mismatch between training and detection.")
            # Попробуем привести к нужному набору колонок
            missing_cols = list(set(self.feature_names) - set(df_features.columns))
            extra_cols = list(set(df_features.columns) - set(self.feature_names))
            print(f"  Missing columns: {missing_cols}")
            print(f"  Extra columns: {extra_cols}")
            for col in missing_cols:
                df_features[col] = 0 # Добавляем недостающие с нулями
            df_features = df_features[self.feature_names] # Оставляем только нужные и в правильном порядке
            print("  Adjusted columns for detection.")

        try:
            # decision_function возвращает anomaly score
            scores = self.model.decision_function(df_features)
            # predict возвращает -1 для аномалий, 1 для нормальных
            predictions = self.model.predict(df_features)
        except Exception as e:
            print(f"Error during Isolation Forest prediction: {e}")
            return []

        anomalies = []
        anomaly_indices = df_features.index[predictions == -1]

        print(f"Isolation Forest raw scores calculated. Found {len(anomaly_indices)} potential anomalies.")

        for index in anomaly_indices:
            original_activity_id = df_engineered.loc[index, 'id']
            anomaly_score = scores[index]

            # Определение уровня серьезности на основе anomaly score
            # Чем score ниже (больше отрицательное значение), тем выше серьезность
            # Пороги подбираются экспериментально
            severity = "Low"
            if anomaly_score < -0.1: # Примерный порог для Medium
                severity = "Medium"
            if anomaly_score < -0.25: # Примерный порог для High
                severity = "High"

            anomaly_info = {
                "detector_name": "isolation_forest",
                "entity_type": entity_type,
                "entity_id": int(original_activity_id),
                "anomaly_score": float(anomaly_score),
                "severity": severity, # <-- Добавили серьезность
                "details": {
                    # Можно добавить сюда значения признаков, если нужно
                    # feature_values = df_features.loc[index].to_dict()
                }
            }
            anomalies.append(anomaly_info)
            # print(f"    Anomaly detected: ID={original_activity_id}, Score={anomaly_score:.4f}, Severity={severity}")

        print(f"Isolation Forest detection finished. Anomalies prepared: {len(anomalies)}")
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


    def detect(self, db: Session, entity_type: str, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии в данных, используя сохраненную статистику.
        Возвращает список словарей с информацией об аномалиях, включая severity.
        """
        print(f"Starting statistical detection for '{entity_type}'...")
        anomalies = []
        entity_stats = self.stats.get(entity_type)

        if not entity_stats:
            print(f"No statistics found for '{entity_type}'. Cannot perform detection. Please train first.")
            return []

        data = None
        feature_columns = list(entity_stats.keys()) # Колонки, для которых есть статистика
        id_column = 'id' # Имя колонки с ID

        if entity_type == 'activity':
            activities = crud.get_user_activities(db, skip=0, limit=limit)
            if not activities: return []
            df_features = engineer_features(activities)
            if df_features is None or df_features.empty:
                 print("Feature engineering failed or returned empty DataFrame for activity detection.")
                 return []
            data = df_features
            # Убедимся, что все нужные колонки есть
            feature_columns = [col for col in feature_columns if col in data.columns]

        elif entity_type == 'order':
            orders = crud.get_orders(db, skip=0, limit=limit)
            if not orders: return []
            order_data = [{
                'id': order.id,
                'customer_id': order.customer_id,
                'created_at': pd.to_datetime(order.created_at),
                'total_amount': order.total_amount,
                'item_count': len(order.items)
            } for order in orders]
            df_orders = pd.DataFrame(order_data)
            if df_orders.empty:
                print("DataFrame creation failed or returned empty for order detection.")
                return []
            data = df_orders
            id_column = 'id' # В данном случае совпадает
            # Убедимся, что все нужные колонки есть
            feature_columns = [col for col in feature_columns if col in data.columns]
        else:
            print(f"Unsupported entity_type '{entity_type}' for statistical detection.")
            return []

        if data is None or data.empty or not feature_columns:
            print(f"Insufficient data or features for detection for '{entity_type}'.")
            return []

        print(f"Detecting anomalies using Z-score (threshold={self.z_threshold}) for columns: {feature_columns}")

        for col in feature_columns:
            if col not in data.columns:
                print(f"  Warning: Column '{col}' not found in detection data. Skipping.")
                continue

            mean_val = entity_stats[col]['mean']
            std_val = entity_stats[col]['std']

            if std_val == 0: # Пропускаем, если стандартное отклонение 0
                print(f"  Warning: Standard deviation for '{col}' is 0. Skipping z-score calculation.")
                continue

            # Рассчитываем Z-score
            z_scores = (data[col] - mean_val) / std_val
            # Находим аномальные индексы
            anomaly_indices = data[np.abs(z_scores) > self.z_threshold].index

            print(f"  Found {len(anomaly_indices)} potential anomalies for column '{col}'")

            for index in anomaly_indices:
                entity_id = data.loc[index, id_column]
                actual_value = data.loc[index, col]
                anomaly_z_score = z_scores.loc[index]

                # Определение уровня серьезности на основе Z-Score
                severity = "Low"
                if abs(anomaly_z_score) > self.z_threshold * 1.5: # Например, в 1.5 раза выше порога
                    severity = "Medium"
                if abs(anomaly_z_score) > self.z_threshold * 2.5: # Например, в 2.5 раза выше порога
                    severity = "High"

                anomaly_info = {
                    "detector_name": "statistical_zscore",
                    "entity_type": entity_type,
                    "entity_id": int(entity_id), # Убедимся, что ID - это int
                    "anomaly_score": float(abs(anomaly_z_score)), # Используем абсолютное значение Z-Score как score
                    "severity": severity, # <-- Добавили серьезность
                    "details": {
                        "feature": col,
                        "value": float(actual_value),
                        "mean": float(mean_val),
                        "std_dev": float(std_val),
                        "z_score": float(anomaly_z_score),
                        "threshold": self.z_threshold
                    }
                }
                anomalies.append(anomaly_info)
                # print(f"    Anomaly detected: ID={entity_id}, Feature='{col}', Value={actual_value:.2f}, Z-Score={anomaly_z_score:.2f}, Severity={severity}")

        print(f"Statistical detection finished. Total anomalies found: {len(anomalies)}")
        return anomalies

# --- Детектор: DBSCAN ---
class DbscanDetector:
    # Параметры DBSCAN нужно подбирать, это просто примерные значения
    # eps: Макс. расстояние между образцами для одного соседства.
    # min_samples: Кол-во образцов в окрестности точки, чтобы считать ее основной.
    def __init__(self, eps=0.5, min_samples=5):
         self.eps = eps
         self.min_samples = min_samples
         self.scaler = StandardScaler() # DBSCAN чувствителен к масштабу
         self.feature_names: List[str] = [] # Признаки, на которых обучался скейлер
         self.ohe_categories: Dict[str, List[Any]] = {} # OHE категории

    # DBSCAN не требует явного сохранения модели, но скейлер и признаки - да.
    # Можно использовать ту же логику, что и в IF, или передавать их при детекции.
    # Для простоты, будем переобучать скейлер каждый раз перед detect.
    # Это менее эффективно, но проще в реализации без сохранения состояния скейлера.

    def train(self, db: Session, entity_type: str, limit: int):
         # Обучение не требуется для DBSCAN, но можем использовать для fit скейлера, если хотим
         # Пока оставляем пустым, но с новой сигнатурой
         print(f"DBSCAN Training: Not implemented/needed for entity '{entity_type}'. Skipping.")
         pass

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Подготавливает признаки для DBSCAN, включая One-Hot Encoding.
        Возвращает DataFrame только с числовыми и OHE признаками.
        """
        if df is None or df.empty:
            return None

        print("Preparing features for DBSCAN...")
        # Выбираем числовые признаки, которые есть в DataFrame
        numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
        features = df[numerical_cols].copy()

        # Применяем One-Hot Encoding к категориальному признаку
        if CATEGORICAL_FEATURE in df.columns:
            try:
                ohe_df = pd.get_dummies(df[CATEGORICAL_FEATURE], prefix=CATEGORICAL_FEATURE, dummy_na=False)
                # Важно: используем df.index для соединения, чтобы сохранить соответствие строк
                features = features.join(ohe_df)
                print(f"  Applied One-Hot Encoding for '{CATEGORICAL_FEATURE}'. New columns: {list(ohe_df.columns)}")
            except Exception as e:
                print(f"Error during One-Hot Encoding for DBSCAN: {e}")
                # В случае ошибки OHE, продолжаем только с числовыми
        else:
            print(f"  Categorical feature '{CATEGORICAL_FEATURE}' not found. Using only numerical features.")

        # Заполняем пропуски нулями (важно перед масштабированием)
        features = features.fillna(0.0)

        # Просто возвращаем признаки до масштабирования
        print(f"Prepared {len(features.columns)} features for DBSCAN (before scaling).")
        return features

    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии как выбросы с помощью DBSCAN.
        Возвращает список словарей с информацией об аномалиях, включая severity.
        Масштабирование выполняется внутри этого метода.
        """
        print(f"Starting DBSCAN detection for '{entity_type}'...")
        if entity_type != 'activity':
            print(f"DBSCAN detector currently only supports 'activity'. Skipping for '{entity_type}'.")
            return []

        # Используем правильное имя функции: get_user_activities
        activities = crud.get_user_activities(db, skip=0, limit=limit)
        if not activities:
            print("No activity data found for DBSCAN detection.")
            return []

        df_engineered = engineer_features(activities)
        if df_engineered is None or df_engineered.empty:
            print("Feature engineering failed or returned empty DataFrame.")
            return []

        # Получаем признаки (включая OHE)
        df_features = self._prepare_features(df_engineered)
        if df_features is None or df_features.empty:
            print("Feature preparation failed or returned empty DataFrame.")
            return []

        # Стандартизация данных - создаем и обучаем scaler здесь!
        try:
            scaler = StandardScaler()
            print(f"Scaling features for DBSCAN using {len(df_features.columns)} columns: {list(df_features.columns)}")
            X_scaled = scaler.fit_transform(df_features) 
            print("Features scaled successfully.")
        except Exception as e:
            print(f"Error scaling features for DBSCAN: {e}")
            # Выведем детали для отладки
            print("Features DataFrame info:")
            try: df_features.info()
            except: print("Could not print df_features info.")
            print("\nFeature columns:", list(df_features.columns))
            print("\nNaN counts per column:", df_features.isna().sum().to_dict())
            return []

        # Применение DBSCAN
        try:
            print(f"Running DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(X_scaled)
        except Exception as e:
            print(f"Error running DBSCAN fit_predict: {e}")
            return []

        # -1 означает выбросы (аномалии)
        anomaly_indices = df_features.index[labels == -1]
        print(f"DBSCAN clustering complete. Found {len(anomaly_indices)} outliers (anomalies)." )

        anomalies = []
        for index in anomaly_indices:
            original_activity_id = df_engineered.loc[index, 'id']

            anomaly_info = {
                "detector_name": "dbscan",
                "entity_type": entity_type,
                "entity_id": int(original_activity_id),
                "anomaly_score": -1.0, # Условное значение для выбросов DBSCAN
                "severity": "Medium", # <-- Устанавливаем среднюю серьезность для всех выбросов DBSCAN
                "details": {
                    "dbscan_label": -1,
                    # Можно добавить значения признаков
                    # "features": df_features.loc[index].to_dict()
                }
            }
            anomalies.append(anomaly_info)
            # print(f"    Anomaly detected: ID={original_activity_id}, Label=-1, Severity=Medium")

        print(f"DBSCAN detection finished. Anomalies prepared: {len(anomalies)}")
        return anomalies

# --- Фабрика детекторов ---

def get_detector(algorithm_name: str, params: schemas.DetectionParams) -> Any:
    """Возвращает инстанс детектора по имени и параметрам."""
    if algorithm_name == "isolation_forest":
        return IsolationForestDetector()
    elif algorithm_name == "statistical_zscore":
        # Передаем z_threshold из параметров запроса
        return StatisticalDetector(z_threshold=params.z_threshold or 3.0)
    elif algorithm_name == "dbscan":
        # Используем параметры из localStorage/SettingsPage, переданные в params
        # Загружаем настройки из localStorage (они должны быть переданы в DetectionParams)
        # TODO: Передать параметры dbscan_eps и dbscan_min_samples в DetectionParams
        # Пока используем дефолтные значения, если они не переданы
        eps = params.dbscan_eps if hasattr(params, 'dbscan_eps') and params.dbscan_eps is not None else 0.5
        min_samples = params.dbscan_min_samples if hasattr(params, 'dbscan_min_samples') and params.dbscan_min_samples is not None else 5
        print(f"Creating DbscanDetector with eps={eps}, min_samples={min_samples}")
        return DbscanDetector(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unknown detection algorithm: {algorithm_name}")



# --- Главная функция для запуска детекции --- (устарела, логика перенесена в API)
# def run_detection_and_save(db: Session, detector: Any, entity_type: str, limit: int):
#     # ... (код из старой версии)

# --- Главная функция для запуска обучения --- (устарела, логика перенесена в API)
# def train_models(db: Session, detectors: List[Any], entity_type: str, limit: int):
#     # ... (код из старой версии) 