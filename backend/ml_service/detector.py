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
# Восстанавливаем импорт необходимых констант
from .common import SAVED_MODELS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURE, engineer_features
# --- Импортируем AutoencoderDetector --- 
from .autoencoder_detector import AutoencoderDetector

# Путь для сохранения модели по умолчанию
DEFAULT_MODEL_DIR = "backend/ml_service/saved_models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "isolation_forest_detector.joblib")

# --- Общие настройки и функции ---
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)

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
        print("IF Training: Starting...")
        if entity_type != 'activity':
            print(f"IF Training: Currently only supports 'activity'. Skipping training for '{entity_type}'.")
            return

        print(f"IF Training: Loading last {limit} activities...")
        # --- FIX: Unpack the tuple returned by crud.get_user_activities ---
        _total_count, activities = crud.get_user_activities(db, limit=limit)
        if not activities:
            print(f"IF Training: No activities found.")
            return
        print(f"IF Training: Loaded {len(activities)} activities.")
        # -------------------------------------------------------------------

        try:
            df_engineered = engineer_features(activities)
            if df_engineered is None: raise ValueError("Feature engineering failed")

            features_df = self._prepare_features(df_engineered, is_training=True)
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
        print("IF Detector: Starting detection...")
        if entity_type != 'activity':
            print(f"IF Detector: Currently only supports 'activity'. Skipping detection for '{entity_type}'.")
            return []
        if not self.model or not self.scaler or not self.feature_names or self.ohe_categories is None:
            print("IF model/scaler/features not loaded. Cannot detect.")
            return []

        # --- FIX: Unpack the tuple returned by crud.get_user_activities ---
        _total_count, activities = crud.get_user_activities(db, limit=limit)
        if not activities: return [] # No data to check
        # -------------------------------------------------------------------
        df_engineered = engineer_features(activities)
        if df_engineered is None: return []

        # Подготовка признаков (OHE + Масштабирование) как при обучении
        X = self._prepare_features(df_engineered, is_training=False)
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
            activity_id = df_engineered.iloc[idx]['id']
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
        print(f"Starting statistical training for '{entity_type}'...")
        if entity_type == 'activity':
            # --- FIX: Unpack the tuple returned by crud.get_user_activities ---
            _total_count, activities = crud.get_user_activities(db, limit=limit)
            if not activities: print("Statistical Training (activity): No data found."); return
            # -------------------------------------------------------------------
            df_features = engineer_features(activities)
            if df_features is None or df_features.empty:
                print("Statistical Training (activity): Feature engineering failed or returned empty.")
                return
            columns_to_process = NUMERICAL_FEATURES

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
            df_features = df_orders
            # Определяем числовые признаки для заказов
            columns_to_process = ['total_amount', 'item_count']
            # Убедимся, что эти колонки существуют
            columns_to_process = [col for col in columns_to_process if col in df_features.columns]
        else:
            print(f"Unsupported entity_type '{entity_type}' for statistical training.")
            return

        if not columns_to_process:
            print(f"No numerical feature columns found for '{entity_type}' in the data.")
            return

        print(f"Calculating stats for columns: {columns_to_process}")
        entity_stats = {}
        try:
            for col in columns_to_process:
                # --- Добавляем проверку на наличие числовых данных перед агрегацией ---
                if pd.api.types.is_numeric_dtype(df_features[col]):
                    mean_val = df_features[col].mean()
                    std_val = df_features[col].std()
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
            print(df_features[columns_to_process].dtypes)


    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print(f"Statistical Detector ({entity_type}): Starting detection...")
        if not self.stats.get(entity_type):
            print(f"Warning: Stats not loaded for entity_type '{entity_type}'. Detection might be inaccurate. Trying load...")
            self._load_stats(entity_type)
            if not self.stats.get(entity_type):
                print(f"Error: Stats for '{entity_type}' could not be loaded. Cannot perform detection.")
                return []

        all_anomalies = [] # Initialize list to store all found anomalies
        df_features = None
        id_column_name = 'id' # Default ID column name

        if entity_type == 'activity':
            _total_count, activities = crud.get_user_activities(db, limit=limit)
            if not activities: return []
            df_features = engineer_features(activities)
            if df_features is None or df_features.empty: return []
            # Features to check for activities
            features_to_check = [f for f in NUMERICAL_FEATURES if f in df_features.columns]

        elif entity_type == 'order':
            # --- Order logic remains largely the same, but we adapt it slightly --- 
            data = crud.get_orders(db, limit=limit)
            if not data: print(f"Statistical Detector ({entity_type}): No data found."); return []
            order_list = [o.__dict__ for o in data] 
            df_features = pd.DataFrame(order_list)
            if df_features.empty: print(f"Statistical Detector ({entity_type}): DataFrame empty."); return []
            # Ensure required columns exist 
            if 'id' in df_features.columns: df_features.rename(columns={'id': 'order_id'}, inplace=True) 
            id_column_name = 'order_id' # Use the correct ID column for orders
            # Features to check for orders
            features_to_check = ['total_amount'] # Check only total_amount for orders for now
            features_to_check = [col for col in features_to_check if col in df_features.columns] 
            # ---------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported entity_type for StatisticalDetector: {entity_type}")

        if df_features is None or df_features.empty:
             print(f"Statistical Detector ({entity_type}): No features DataFrame to process.")
             return []
        if not features_to_check:
            print(f"Statistical Detector ({entity_type}): No features identified to check.")
            return []

        # --- Loop through each feature to check --- 
        for feature_name in features_to_check:
            print(f"\nStatistical Detector ({entity_type}): Checking feature '{feature_name}'...")
            
            # 1. Check if stats exist for this feature
            if feature_name not in self.stats.get(entity_type, {}):
                print(f"  Stats not available for feature '{feature_name}'. Skipping.")
                continue

            stats_for_feature = self.stats[entity_type][feature_name]
            mean = stats_for_feature.get('mean')
            std = stats_for_feature.get('std')

            if mean is None or std is None or std == 0 or pd.isna(mean) or pd.isna(std):
                 print(f"  Invalid stats (mean={mean}, std={std}) for '{feature_name}'. Skipping.")
                 continue

            print(f"  Using stats: mean={mean:.4f}, std={std:.4f}, z_threshold={self.z_threshold}")

            # 2. Calculate Z-scores for this feature
            try:
                 # Ensure the column is numeric
                 if not pd.api.types.is_numeric_dtype(df_features[feature_name]):
                    print(f"  Warning: Column '{feature_name}' is not numeric. Attempting conversion.")
                    # Create a copy to avoid SettingWithCopyWarning if conversion modifies the original df_features slice
                    feature_series = pd.to_numeric(df_features[feature_name], errors='coerce')
                    # We need the original indices even after dropping NaNs for matching later
                    valid_indices = feature_series.dropna().index 
                    if valid_indices.empty:
                        print(f"  DataFrame is empty after numeric conversion/dropna for '{feature_name}'. Skipping feature.")
                        continue
                 else:
                     feature_series = df_features[feature_name]
                     valid_indices = feature_series.index # All indices are valid initially

                 # Calculate Z-scores only for valid numeric entries
                 z_scores = (feature_series.loc[valid_indices] - mean) / std
                 potential_anomaly_indices = z_scores.index[np.abs(z_scores) >= self.z_threshold]

                 print(f"  Found {len(potential_anomaly_indices)} potential anomalies for '{feature_name}'.")
                 if not potential_anomaly_indices.empty:
                     print(f"  Z-score distribution for '{feature_name}':\n{z_scores.describe()}")

                 # 3. Format anomalies found for this feature
                 for idx in potential_anomaly_indices:
                    row = df_features.loc[idx]
                    # Use the correct ID column name determined earlier
                    entity_id = row.get(id_column_name) 
                    if entity_id is None: continue # Skip if ID is missing
                    
                    current_z = z_scores.loc[idx]
                    current_value = feature_series.loc[idx] # Use the numeric series

                    severity = "Medium"
                    if abs(current_z) >= self.z_threshold * 1.5: severity = "High"

                    # --- Перевод Reason на русский --- 
                    deviation_direction = "выше" if current_z > 0 else "ниже"
                    reason_str = (
                        f"Значение признака '{feature_name}' ({current_value:.4f}) значительно отклоняется от среднего ({mean:.4f}). "
                        f"Z-оценка = {current_z:.2f}, что {deviation_direction} порога {self.z_threshold}. "
                        f"(Ст. откл.={std:.4f})"
                    )
                    # ---------------------------------

                    anomaly_info = {
                        "entity_type": entity_type,
                        "entity_id": int(entity_id),
                        "reason": reason_str,
                        "details": {
                            "feature": feature_name,
                            "value": float(current_value),
                            "mean": float(mean),
                            "std": float(std),
                            "z_score": float(current_z)
                        },
                        "severity": severity,
                        "anomaly_score": float(abs(current_z))
                    }
                    # Add specific ID for convenience if needed (might be redundant with entity_id)
                    # if entity_type == 'activity': anomaly_info['activity_id'] = int(entity_id)
                    # elif entity_type == 'order': anomaly_info['order_id'] = int(entity_id)

                    all_anomalies.append(anomaly_info)

            except KeyError as e:
                 print(f"  KeyError for feature '{feature_name}': {e}. Check if feature or '{id_column_name}' exists.")
                 # print(f"DataFrame columns: {df_features.columns.tolist()}")
                 continue # Skip to the next feature
            except Exception as e:
                 print(f"  Error during Z-score calculation/anomaly formatting for '{feature_name}': {e}")
                 # import traceback
                 # traceback.print_exc()
                 continue # Skip to the next feature
        # --- End of loop through features --- 

        print(f"\nStatistical Detector ({entity_type}): Completed checks. Total anomalies found across all features: {len(all_anomalies)}.")
        return all_anomalies

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
            print(f"DBSCAN currently only supports 'activity'. Skipping detection for '{entity_type}'.")
            return []

        # --- FIX: Unpack the tuple returned by crud.get_user_activities ---
        _total_count, activities = crud.get_user_activities(db, limit=limit)
        if not activities: return [] # No data to check
        # -------------------------------------------------------------------
        df_engineered = engineer_features(activities)
        if df_engineered is None: return []

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
        print(f"Unsupported algorithm: {algorithm}")
        return None
