import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from sklearn.cluster import DBSCAN

# --- Нужно импортировать CRUD и модели --- 
from .. import crud, models # Относительный импорт из родительской директории

# --- Общие константы и настройки для ДЕТЕКТОРОВ ЗАКАЗОВ ---
# Используем поддиректорию, чтобы не смешивать модели
SAVED_ORDER_MODELS_DIR = "backend/ml_service/saved_models/order_detectors"
if not os.path.exists(SAVED_ORDER_MODELS_DIR):
    os.makedirs(SAVED_ORDER_MODELS_DIR)

# Список числовых признаков, которые будем использовать для ЗАКАЗОВ
ORDER_NUMERICAL_FEATURES = [
    'total_amount', 
    'item_count', 
    'total_quantity',
    'hour_of_day', 
    'day_of_week'
]
# Категориальные признаки для заказов пока не используем
# ORDER_CATEGORICAL_FEATURE = None 

# --- Функция для подготовки признаков ЗАКАЗОВ --- 
def engineer_order_features(orders: List[models.Order]) -> Optional[pd.DataFrame]:
    """
    Преобразует список заказов в DataFrame и рассчитывает базовые признаки.
    """
    if not orders: return None
    
    data_list = []
    for order in orders:
        item_count = len(order.items) if order.items else 0
        total_quantity = sum(item.quantity for item in order.items) if order.items else 0
        hour = order.created_at.hour if order.created_at else -1 # Используем -1 для отсутствующих дат
        day_week = order.created_at.weekday() if order.created_at else -1
        
        data_list.append({
            "id": order.id, # ID заказа
            "customer_id": order.customer_id, # ID клиента
            "created_at": pd.to_datetime(order.created_at) if order.created_at else None,
            "total_amount": float(order.total_amount) if order.total_amount is not None else 0.0,
            "item_count": item_count,
            "total_quantity": total_quantity,
            "hour_of_day": hour,
            "day_of_week": day_week
        })
        
    df = pd.DataFrame(data_list)
    if df.empty: return None

    # Оставляем только нужные колонки + id для связи
    cols_to_keep = ['id'] + ORDER_NUMERICAL_FEATURES
    df = df[[col for col in cols_to_keep if col in df.columns]]
    
    # Обработка пропусков (заполняем нулями или -1)
    for col in ORDER_NUMERICAL_FEATURES:
        if col in df.columns:
            fill_val = -1 if col in ['hour_of_day', 'day_of_week'] else 0
            df[col] = df[col].fillna(fill_val)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val)
        else:
            print(f"[Order FE] Warning: Feature '{col}' not found.")
            df[col] = 0 # Добавляем колонку с нулями, если ее нет
            
    print(f"[Order FE] Engineered features for {len(df)} orders. Columns: {df.columns.tolist()}")
    return df


# --- Детектор Заказов: Isolation Forest --- 
class OrderIsolationForestDetector:
    # Указываем новые пути для модели и скейлера
    DEFAULT_MODEL_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "iforest_order_detector.joblib")
    DEFAULT_SCALER_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "iforest_order_scaler.joblib")

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH, contamination: float = 'auto', random_state: int = 42):
        self.model_path = model_path
        self.scaler_path = scaler_path # Новый путь для скейлера
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = ORDER_NUMERICAL_FEATURES # Используем признаки заказов
        self._load_model()
        if self.model is None: self._initialize_new_model()

    def _initialize_new_model(self):
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state, n_estimators=100)
        self.scaler = StandardScaler()
        print("Initialized new Order Isolation Forest model and scaler.")

    def _save_model(self):
        # Сохраняем модель и скейлер ОТДЕЛЬНО
        if self.model:
            try:
                joblib.dump(self.model, self.model_path)
                print(f"Order IF model saved to {self.model_path}")
            except Exception as e: print(f"Error saving Order IF model: {e}")
        if self.scaler:
            try:
                joblib.dump(self.scaler, self.scaler_path)
                print(f"Order IF scaler saved to {self.scaler_path}")
            except Exception as e: print(f"Error saving Order IF scaler: {e}")
            
    def _load_model(self):
        # Загружаем модель
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"Order IF model loaded from {self.model_path}")
            except Exception as e: print(f"Error loading Order IF model: {e}. Init new."); self._initialize_new_model()
        else: print(f"Order IF model file not found at {self.model_path}."); self._initialize_new_model()
        
        # Загружаем скейлер ОТДЕЛЬНО
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"Order IF scaler loaded from {self.scaler_path}")
            except Exception as e: 
                print(f"Error loading Order IF scaler: {e}. Creating new.")
                # Если модель загрузилась, а скейлер нет - создаем новый скейлер
                if self.model is not None: self.scaler = StandardScaler()
        else: 
            print(f"Order IF scaler file not found at {self.scaler_path}. Creating new.")
            if self.model is not None: self.scaler = StandardScaler() # Создаем, если не найден

    # --- Адаптированный _prepare_features --- 
    def _prepare_features(self, df_engineered: pd.DataFrame, is_training: bool = False) -> Optional[pd.DataFrame]:
        """Масштабирует числовые признаки заказа."""
        print("[Order IF] Preparing features...")
        if df_engineered is None or df_engineered.empty:
            return None
            
        # Выбираем только числовые признаки (уже определены в self.feature_names)
        numeric_df = df_engineered[self.feature_names].copy()
        
        if is_training:
            print(f"[Order IF] Fitting scaler using features: {self.feature_names}")
            try:
                scaled_numeric = self.scaler.fit_transform(numeric_df)
                # Сохраняем обученный скейлер
                self._save_model() # Сохраняем и модель, и скейлер
            except Exception as e:
                print(f"[Order IF] Error during scaler fitting/transform: {e}") 
                return None
        else:
            if not self.scaler:
                 print("[Order IF] Scaler not ready for prediction."); return None
            print(f"[Order IF] Scaling features using existing scaler: {self.feature_names}")
            try:
                scaled_numeric = self.scaler.transform(numeric_df)
            except Exception as e:
                print(f"[Order IF] Error during scaling transform: {e}")
                return None

        scaled_df = pd.DataFrame(scaled_numeric, columns=self.feature_names, index=df_engineered.index)
        print(f"[Order IF] Prepared {len(scaled_df.columns)} features.")
        return scaled_df

    # --- Адаптированный train --- 
    def train(self, db: Session, entity_type: str, limit: int):
        print("[Order IF] Training starting...")
        if entity_type != 'order': # Этот детектор только для заказов
            print(f"Order IF Training: Expected 'order', got '{entity_type}'. Skipping.")
            return

        print(f"[Order IF] Loading last {limit} orders...")
        orders = crud.get_orders(db, limit=limit) # Используем get_orders
        if not orders:
            print(f"[Order IF] Training: No orders found."); return
        print(f"[Order IF] Loaded {len(orders)} orders.")

        try:
            df_engineered = engineer_order_features(orders) # Используем новый FE
            if df_engineered is None: raise ValueError("Order Feature engineering failed")

            features_df = self._prepare_features(df_engineered, is_training=True)
            if features_df is None or features_df.empty:
                print("[Order IF] Training: No features to train on after preparation."); return

            print(f"[Order IF] Fitting model on {len(features_df)} samples...")
            self.model.fit(features_df)
            print("[Order IF] Training: Model fit complete.")
            # self._save_model() # Модель и скейлер уже сохранились в _prepare_features

        except Exception as e:
            print(f"[Order IF] Training Error: {e}")

    # --- Адаптированный detect --- 
    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print("[Order IF] Detection starting...")
        if entity_type != 'order': 
            print(f"[Order IF] Detection: Expected 'order', got '{entity_type}'. Skipping."); return []
        if not self.model or not self.scaler:
            print("[Order IF] Model/scaler not loaded. Cannot detect."); return []

        orders = crud.get_orders(db, limit=limit)
        if not orders: print("[Order IF] No order data to detect."); return []
        
        df_engineered = engineer_order_features(orders)
        if df_engineered is None: print("[Order IF] Feature engineering failed."); return []

        X = self._prepare_features(df_engineered, is_training=False)
        if X is None: print("[Order IF] Feature preparation failed."); return []

        try:
            scores = self.model.decision_function(X) 
            predictions = self.model.predict(X) 
            print(f"[Order IF] Prediction completed. Found {sum(predictions == -1)} potential anomalies.")
        except Exception as e:
            print(f"[Order IF] Error during prediction: {e}"); return []

        anomalies = []
        anomaly_indices = np.where(predictions == -1)[0]
        original_indices = df_engineered.index[anomaly_indices] # Получаем исходные индексы пандас

        # --- SHAP (Опционально, можно добавить позже, если нужно) --- 
        # ... (код для SHAP, если адаптировать) ...
        
        for i, original_idx in enumerate(original_indices):
            order_id = df_engineered.loc[original_idx]['id']
            current_score = scores[anomaly_indices[i]] # Используем индекс из anomaly_indices

            severity = "Low"
            if current_score <= -0.1: severity = "High"
            elif current_score <= 0.05: severity = "Medium"

            reason_str = (
                f"Заказ (ID: {order_id}) выделен Isolation Forest для заказов (оценка: {current_score:.4f}). "
                f"Указывает на необычное сочетание признаков заказа (сумма, кол-во и т.д.)."
            )
            
            details_dict = {"anomaly_score": float(current_score)}
            # --- Добавление SHAP в details (если будет реализовано) --- 
            # ...

            anomaly_info = {
                "entity_type": "order", # Указываем правильный тип
                "entity_id": int(order_id),
                "reason": reason_str,
                "details": details_dict, 
                "severity": severity, 
                "anomaly_score": float(current_score) # Score для заказов
            }
            anomalies.append(anomaly_info)

        print(f"[Order IF] Formatted {len(anomalies)} anomalies.")
        return anomalies

# --- Детектор Заказов: DBSCAN --- 
class OrderDbscanDetector:
    # Путь для скейлера DBSCAN для заказов
    DEFAULT_SCALER_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "dbscan_order_scaler.joblib")

    def __init__(self, scaler_path: str = DEFAULT_SCALER_PATH, eps: float = 0.5, min_samples: int = 5):
        self.scaler_path = scaler_path
        self.eps = eps
        self.min_samples = min_samples
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = ORDER_NUMERICAL_FEATURES # Признаки для заказов
        self._load_scaler()
        if self.scaler is None: self.scaler = StandardScaler()
        print(f"[Order DBSCAN] Initialized with eps={self.eps}, min_samples={self.min_samples}")

    def _save_scaler(self):
        if self.scaler:
            try:
                joblib.dump(self.scaler, self.scaler_path)
                print(f"[Order DBSCAN] Scaler saved to {self.scaler_path}")
            except Exception as e: print(f"[Order DBSCAN] Error saving scaler: {e}")

    def _load_scaler(self):
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"[Order DBSCAN] Scaler loaded from {self.scaler_path}")
            except Exception as e: 
                print(f"[Order DBSCAN] Error loading scaler: {e}. Creating new.")
                self.scaler = StandardScaler()
        else: 
            print(f"[Order DBSCAN] Scaler file not found at {self.scaler_path}. Creating new.")
            self.scaler = StandardScaler()

    # --- Адаптированный _prepare_features --- 
    def _prepare_features(self, df_engineered: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Масштабирует числовые признаки заказа для DBSCAN."""
        print("[Order DBSCAN] Preparing features...")
        if df_engineered is None or df_engineered.empty: return None
        numeric_df = df_engineered[self.feature_names].copy()
        try:
            # DBSCAN не "тренируется", но скейлер нужно обучить при первом прогоне или если его нет
            # Считаем, что train вызываться не будет, а скейлер обучится при первом detect
            if getattr(self.scaler, 'n_samples_seen_', 0) == 0: # Проверяем, обучен ли scaler
                 print("[Order DBSCAN] Fitting scaler...")
                 scaled_numeric = self.scaler.fit_transform(numeric_df)
                 self._save_scaler() # Сохраняем обученный скейлер
            else:
                print("[Order DBSCAN] Scaling features using existing scaler...")
                scaled_numeric = self.scaler.transform(numeric_df)
        except Exception as e:
            print(f"[Order DBSCAN] Error during scaling: {e}")
            return None
        scaled_df = pd.DataFrame(scaled_numeric, columns=self.feature_names, index=df_engineered.index)
        print(f"[Order DBSCAN] Prepared {len(scaled_df.columns)} features.")
        return scaled_df

    # --- DBSCAN не требует метода train --- 

    # --- Адаптированный detect --- 
    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print("[Order DBSCAN] Detection starting...")
        if entity_type != 'order': 
             print(f"[Order DBSCAN] Detection: Expected 'order', got '{entity_type}'. Skipping."); return []
        if not self.scaler:
             print("[Order DBSCAN] Scaler not loaded. Cannot detect."); return []
             
        orders = crud.get_orders(db, limit=limit)
        if not orders: print("[Order DBSCAN] No order data to detect."); return []
        
        df_engineered = engineer_order_features(orders)
        if df_engineered is None: print("[Order DBSCAN] Feature engineering failed."); return []

        X = self._prepare_features(df_engineered)
        if X is None: print("[Order DBSCAN] Feature preparation failed."); return []

        print(f"[Order DBSCAN] Running DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        try:
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
            clusters = dbscan.fit_predict(X)
        except Exception as e:
            print(f"[Order DBSCAN] Error during DBSCAN fitting: {e}"); return []

        noise_indices = np.where(clusters == -1)[0]
        original_indices = df_engineered.index[noise_indices]
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        print(f"[Order DBSCAN] Found {num_clusters} clusters and {len(noise_indices)} noise points (anomalies).")

        anomalies = []
        for original_idx in original_indices:
            order_id = df_engineered.loc[original_idx]['id']

            reason_str = (
                f"Заказ (ID: {order_id}) классифицирован как шум DBSCAN для заказов (eps={self.eps}, min_samples={self.min_samples}). "
                f"Это может указывать на редкое сочетание признаков заказа."
            )

            anomaly_info = {
                "entity_type": "order",
                "entity_id": int(order_id),
                "reason": reason_str, 
                "details": {"eps": self.eps, "min_samples": self.min_samples},
                "severity": "Medium", # Шум DBSCAN часто умеренной серьезности
                "anomaly_score": -1.0 # У DBSCAN нет прямого скора, ставим -1 для шума
            }
            anomalies.append(anomaly_info)

        print(f"[Order DBSCAN] Formatted {len(anomalies)} noise points as anomalies.")
        return anomalies

# --- Детектор Заказов: Autoencoder --- 
# Импорты Keras/TensorFlow (предполагается, что они уже есть выше или будут добавлены)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, losses, models as keras_models
    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not found. OrderAutoencoderDetector will not be available.")
    KERAS_AVAILABLE = False

class OrderAutoencoderDetector:
    DEFAULT_MODEL_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "autoencoder_order.keras")
    DEFAULT_SCALER_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "autoencoder_order_scaler.joblib")
    # OHE пока не используем
    # DEFAULT_OHE_PATH = os.path.join(SAVED_ORDER_MODELS_DIR, "autoencoder_order_ohe.joblib") 

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 scaler_path: str = DEFAULT_SCALER_PATH,
                 encoding_dim: int = 3, # Уменьшим размерность для меньшего кол-ва признаков
                 epochs: int = 30,
                 batch_size: int = 32,
                 anomaly_threshold: float = 0.5 ):
        if not KERAS_AVAILABLE: 
             raise ImportError("TensorFlow/Keras required for OrderAutoencoderDetector")
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold
        self.model: Optional[keras_models.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = ORDER_NUMERICAL_FEATURES
        self._load_model_and_scaler()

    def _build_model(self, input_dim: int):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(input_dim * 2, activation='relu'), # Простая архитектура
            layers.Dense(self.encoding_dim, activation='relu'),
            layers.Dense(input_dim * 2, activation='relu'),
            layers.Dense(input_dim, activation='linear') # Linear для StandardScaler
        ])
        model.compile(optimizer='adam', loss='mse')
        print(f"[Order AE] Built Autoencoder model with input_dim={input_dim}, encoding_dim={self.encoding_dim}")
        return model

    def _save_model(self):
        if self.model: self.model.save(self.model_path)
        if self.scaler: joblib.dump(self.scaler, self.scaler_path)
        print(f"[Order AE] Saved model to {self.model_path} and scaler to {self.scaler_path}")

    def _load_model_and_scaler(self):
        if os.path.exists(self.model_path):
             try: self.model = keras_models.load_model(self.model_path); print("[Order AE] Model loaded.")
             except Exception as e: print(f"[Order AE] Error loading model: {e}"); self.model = None
        else: print("[Order AE] Model file not found."); self.model = None
        if os.path.exists(self.scaler_path):
             try: self.scaler = joblib.load(self.scaler_path); print("[Order AE] Scaler loaded.")
             except Exception as e: print(f"[Order AE] Error loading scaler: {e}"); self.scaler = None
        else: print("[Order AE] Scaler file not found."); self.scaler = None

    # --- Адаптированный _prepare_features (только масштабирование) --- 
    def _prepare_features(self, df_engineered: pd.DataFrame, is_training: bool = False) -> Optional[pd.DataFrame]:
        print("[Order AE] Preparing features...")
        if df_engineered is None or df_engineered.empty: return None
        numeric_df = df_engineered[self.feature_names].copy()
        if is_training:
            if self.scaler is None: self.scaler = StandardScaler() # Создаем, если не загрузился
            print("[Order AE] Fitting scaler...")
            try: 
                scaled_numeric = self.scaler.fit_transform(numeric_df)
                # Сохраняем только scaler, модель сохранится после fit
                if self.scaler: joblib.dump(self.scaler, self.scaler_path); print(f"[Order AE] Scaler saved to {self.scaler_path}")
            except Exception as e: print(f"[Order AE] Scaler fit error: {e}"); return None
        else:
            if not self.scaler: print("[Order AE] Scaler not ready."); return None
            try: scaled_numeric = self.scaler.transform(numeric_df)
            except Exception as e: print(f"[Order AE] Scaler transform error: {e}"); return None
        scaled_df = pd.DataFrame(scaled_numeric, columns=self.feature_names, index=df_engineered.index)
        print(f"[Order AE] Prepared {len(scaled_df.columns)} features.")
        return scaled_df

    # --- Адаптированный train --- 
    def train(self, db: Session, entity_type: str, limit: int):
        print("[Order AE] Training starting...")
        if entity_type != 'order': print(f"[Order AE] Expected 'order', got '{entity_type}'. Skipping."); return
        orders = crud.get_orders(db, limit=limit)
        if not orders: print("[Order AE] No orders found."); return
        df_engineered = engineer_order_features(orders)
        if df_engineered is None: print("[Order AE] Feature engineering failed."); return
        df_prepared = self._prepare_features(df_engineered, is_training=True)
        if df_prepared is None: print("[Order AE] Feature preparation failed."); return
        input_dim = df_prepared.shape[1]
        self.model = self._build_model(input_dim)
        try:
            history = self.model.fit(df_prepared, df_prepared, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
            print(f"[Order AE] Training complete. Final val_loss: {history.history['val_loss'][-1]:.4f}")
            self._save_model() # Сохраняем модель и скейлер
        except Exception as e: print(f"[Order AE] Training error: {e}")

    # --- Адаптированный detect --- 
    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print("[Order AE] Detection starting...")
        if entity_type != 'order': print(f"[Order AE] Expected 'order', got '{entity_type}'. Skipping."); return []
        if not self.model or not self.scaler: print("[Order AE] Model/scaler not loaded."); return []
        orders = crud.get_orders(db, limit=limit)
        if not orders: print("[Order AE] No order data."); return []
        df_engineered = engineer_order_features(orders)
        if df_engineered is None: print("[Order AE] FE failed."); return []
        df_prepared = self._prepare_features(df_engineered, is_training=False)
        if df_prepared is None: print("[Order AE] Feature prep failed."); return []
        try:
            reconstructions = self.model.predict(df_prepared)
            mse = np.mean(np.power(df_prepared - reconstructions, 2), axis=1)
            threshold = self.anomaly_threshold
            anomaly_mask = mse > threshold
            anomalies_df = df_engineered[anomaly_mask]
            anomalies_errors = pd.Series(mse[anomaly_mask], index=anomalies_df.index)
            print(f"[Order AE] Found {len(anomalies_df)} anomalies (threshold={threshold:.4f}).")
            output_anomalies = []
            for idx, row in anomalies_df.iterrows():
                order_id = row.get('id')
                if order_id is None: continue
                error_value = anomalies_errors.loc[idx]
                # --- Ошибки по признакам (опционально, аналогично detector.py) ---
                original_features = df_prepared.loc[idx]
                reconstructed_features = reconstructions[df_prepared.index.get_loc(idx)]
                feature_mae_errors = np.abs(original_features - reconstructed_features)
                feature_errors_dict = {k: round(float(v), 4) for k, v in feature_mae_errors.to_dict().items()}
                # --------------------------------------------------------------------
                # --- Добавляем извлечение топ-признаков с ошибками --- 
                top_error_features_str = ""
                if feature_errors_dict:
                    sorted_errors = sorted(feature_errors_dict.items(), key=lambda item: item[1], reverse=True)
                    top_features = [f"{k} (ош: {v:.3f})" for k, v in sorted_errors[:2]]
                    top_error_features_str = f", в основном из-за: {', '.join(top_features)}"
                # -----------------------------------------------------
                
                severity = "Medium"
                if error_value > threshold * 1.5: severity = "High"
                
                # --- ИЗМЕНЕНО: Добавляем топ признаки в reason --- 
                reason_str = (
                    f"Заказ (ID: {order_id}) имеет высокую ошибку реконструкции ({error_value:.4f}) Автоэнкодером для заказов, превышает порог ({threshold:.4f}). "
                    f"Указывает на необычное сочетание признаков заказа{top_error_features_str}."
                )
                # ---------------------------------------------------
                output_anomalies.append({
                    "entity_type": "order",
                    "entity_id": int(order_id),
                    "reason": reason_str,
                    "details": {"reconstruction_error": float(error_value), "threshold": threshold, "feature_errors": feature_errors_dict},
                    "severity": severity,
                    "anomaly_score": float(error_value)
                })
            return output_anomalies
        except Exception as e: print(f"[Order AE] Detection error: {e}"); return []

# --- Детектор Заказов: Statistical (Z-Score) --- 
class OrderStatisticalDetector:
    """
    Обнаруживает аномалии в данных заказов на основе Z-оценки числовых признаков.
    """
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.feature_names: List[str] = ORDER_NUMERICAL_FEATURES # Используем признаки заказов
        print(f"[Order Z-Score] Initialized with threshold={self.z_threshold}")

    # Метод train не требуется, т.к. статистика (mean, std) рассчитывается на лету
    # def train(self, ...): pass 

    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        print("[Order Z-Score] Detection starting...")
        if entity_type != 'order': 
            print(f"[Order Z-Score] Detection: Expected 'order', got '{entity_type}'. Skipping."); return []

        orders = crud.get_orders(db, limit=limit)
        if not orders: print("[Order Z-Score] No order data to detect."); return []
        
        df_engineered = engineer_order_features(orders)
        if df_engineered is None or df_engineered.empty: 
             print("[Order Z-Score] Feature engineering failed or resulted in empty DataFrame."); return []

        anomalies = []
        numeric_df = df_engineered[self.feature_names].copy()
        
        # Проверяем, что в numeric_df есть данные
        if numeric_df.empty:
            print("[Order Z-Score] No numeric features found after selection."); return []
            
        # Проверяем наличие ненулевой дисперсии перед вычислением Z-оценки
        if numeric_df.shape[0] < 2 or numeric_df.std().sum() == 0:
             print("[Order Z-Score] Insufficient data or zero standard deviation for Z-score calculation. Skipping detection.")
             return []

        try:
            # Рассчитываем Z-оценки
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
            
            # Ищем строки, где хотя бы одна Z-оценка превышает порог
            # Используем .any(axis=1), чтобы найти строки с хотя бы одним выбросом
            outlier_rows_mask = z_scores.gt(self.z_threshold).any(axis=1) 
            
            anomalous_indices = numeric_df[outlier_rows_mask].index

            print(f"[Order Z-Score] Found {len(anomalous_indices)} potential anomalies based on Z-score > {self.z_threshold}.")

            for idx in anomalous_indices:
                order_id = df_engineered.loc[idx]['id']
                row_z_scores = z_scores.loc[idx] # Z-оценки для этой строки
                
                # Находим признаки, превысившие порог в этой строке
                violating_features = row_z_scores[row_z_scores > self.z_threshold]
                
                # Максимальная Z-оценка как прокси "серьезности"
                max_z = violating_features.max() 
                
                # Формируем строку причины
                reason_parts = []
                details_dict = {"z_threshold": self.z_threshold, "max_z_score": float(max_z), "violating_features": {}}
                for feature, z_val in violating_features.items():
                    original_value = numeric_df.loc[idx, feature]
                    mean_val = numeric_df[feature].mean()
                    std_val = numeric_df[feature].std()
                    reason_parts.append(f"{feature}={original_value:.2f} (Z={z_val:.2f})")
                    details_dict["violating_features"][feature] = {
                        "value": float(original_value),
                        "z_score": float(z_val),
                        "mean": float(mean_val),
                        "std_dev": float(std_val)
                    }
                
                reason_str = (
                    f"Заказ (ID: {order_id}) имеет аномальные значения Z-Score: "
                    f"{', '.join(reason_parts)}. Порог={self.z_threshold}. "
                    f"Расчет на основе средних и ст.откл. по {len(numeric_df)} заказам."
                )

                # Определяем серьезность на основе максимальной Z-оценки
                severity = "Low"
                if max_z > self.z_threshold * 2: severity = "High"
                elif max_z > self.z_threshold * 1.5: severity = "Medium"

                anomaly_info = {
                    "entity_type": "order",
                    "entity_id": int(order_id),
                    "reason": reason_str,
                    "details": details_dict,
                    "severity": severity,
                    "anomaly_score": float(max_z) # Используем max Z-score как скор
                }
                anomalies.append(anomaly_info)

        except Exception as e:
            print(f"[Order Z-Score] Error during Z-score calculation or processing: {e}")
            # Добавить traceback для отладки
            import traceback
            traceback.print_exc() 
            return [] # Возвращаем пустой список в случае ошибки

        print(f"[Order Z-Score] Formatted {len(anomalies)} Z-score anomalies.")
        return anomalies

# --- Фабрика для детекторов ЗАКАЗОВ --- 
def get_order_detector(algorithm: str, params: Any): # params пока Any
    """Фабричная функция для создания экземпляров детекторов ЗАКАЗОВ."""
    print(f"[Order Factory] Getting detector for algorithm: {algorithm}")
    if algorithm == 'isolation_forest':
        # Передаем contamination из params, если он там есть, иначе auto
        contamination = getattr(params, 'contamination', 'auto') 
        return OrderIsolationForestDetector(contamination=contamination)
    elif algorithm == 'dbscan':
        # Получаем параметры DBSCAN из params (нужно будет их туда добавить)
        eps = getattr(params, 'dbscan_eps', 0.5)
        min_samples = getattr(params, 'dbscan_min_samples', 5)
        return OrderDbscanDetector(eps=eps, min_samples=min_samples)
    elif algorithm == 'autoencoder':
        if not KERAS_AVAILABLE: return None # Возвращаем None, если Keras недоступен
        # Получаем параметры AE из params (нужно будет их туда добавить)
        threshold = getattr(params, 'autoencoder_threshold', 0.5)
        # Другие параметры (encoding_dim, epochs) пока используем дефолтные из конструктора
        return OrderAutoencoderDetector(anomaly_threshold=threshold)
    # --- ДОБАВЛЯЕМ ОБРАБОТКУ Z-SCORE ---
    elif algorithm == 'statistical_zscore':
        z_threshold = getattr(params, 'z_threshold', 3.0) # Берем порог из params
        return OrderStatisticalDetector(z_threshold=z_threshold)
    # -------------------------------------
    else:
        print(f"[Order Factory] Unsupported algorithm for orders: {algorithm}")
        return None 