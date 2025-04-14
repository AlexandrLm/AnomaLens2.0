import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

# --- Осторожно: Зависимость от TensorFlow/Keras ---
# Убедитесь, что tensorflow установлен в окружении
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, losses, models as keras_models
except ImportError:
    print("TensorFlow/Keras not found. AutoencoderDetector will not be available.")
    # Можно определить заглушку класса, чтобы код не падал при импорте
    class AutoencoderDetector:
        def __init__(self, *args, **kwargs):
             raise ImportError("TensorFlow/Keras is required for AutoencoderDetector but not installed.")
        def train(self, *args, **kwargs):
            raise ImportError("TensorFlow/Keras is required for AutoencoderDetector but not installed.")
        def detect(self, *args, **kwargs):
            raise ImportError("TensorFlow/Keras is required for AutoencoderDetector but not installed.")
# ---------------------------------------------------

from .. import crud, models, schemas
# Импортируем общие элементы из common.py
try:
    from .common import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURE, SAVED_MODELS_DIR
except ImportError as e:
     print(f"Could not import from .common: {e}. Check import paths.")
     # Определяем заглушки, чтобы избежать ошибок ниже, но функциональность будет нарушена
     def engineer_features(*args, **kwargs): return None
     NUMERICAL_FEATURES = []
     CATEGORICAL_FEATURE = ''
     SAVED_MODELS_DIR = "."


class AutoencoderDetector:
    """
    Детектор аномалий на основе автоэнкодера Keras.
    Использует ошибку реконструкции для выявления аномалий.
    """
    DEFAULT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "autoencoder_detector.keras")
    DEFAULT_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "autoencoder_scaler.joblib")
    DEFAULT_OHE_PATH = os.path.join(SAVED_MODELS_DIR, "autoencoder_ohe.joblib")

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 scaler_path: str = DEFAULT_SCALER_PATH,
                 ohe_path: str = DEFAULT_OHE_PATH,
                 encoding_dim: int = 16, # Пример: размерность кодированного представления
                 epochs: int = 50,
                 batch_size: int = 32,
                 anomaly_threshold: float = 0.5 # Пример: Порог для ошибки реконструкции
                 ):
        if 'keras' not in globals(): # Проверяем, был ли Keras импортирован
             raise ImportError("TensorFlow/Keras is required for AutoencoderDetector but not installed.")

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.ohe_path = ohe_path
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold # Порог MSE

        self.model: Optional[keras_models.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.ohe_categories: Dict[str, List[Any]] = {}

        self._load_model_and_scaler() # Пытаемся загрузить при инициализации

    def _build_model(self, input_dim: int):
        """Строит модель автоэнкодера Keras."""
        # Простая модель FCNN Autoencoder
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            # Encoder
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.encoding_dim, activation='relu', name='encoder_output'),
            # Decoder
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid') # Используем sigmoid, если данные масштабированы в [0,1]
                                                         # или linear, если в другом диапазоне
        ])
        model.compile(optimizer='adam', loss='mse') # Используем Mean Squared Error как функцию потерь
        print(f"Built Autoencoder model with input_dim={input_dim}, encoding_dim={self.encoding_dim}")
        model.summary() # Выводим структуру модели
        return model

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Optional[pd.DataFrame]:
        """
        Подготовка признаков: OHE + Scaling.
        Аналогично IF/DBSCAN, но сохраняет OHE категории отдельно.
        Масштабирует данные в диапазон [0, 1] с помощью MinMaxScaler (лучше для sigmoid в последнем слое AE).
        """
        print("Preparing features for Autoencoder...")
        if df is None or df.empty: return None

        # 1. One-Hot Encode категориальный признак
        ohe_df = pd.DataFrame(index=df.index) # Начнем с пустого
        if CATEGORICAL_FEATURE in df.columns:
            try:
                current_ohe = pd.get_dummies(df[[CATEGORICAL_FEATURE]], prefix=CATEGORICAL_FEATURE, dummy_na=False)
                if is_training:
                    self.ohe_categories[CATEGORICAL_FEATURE] = current_ohe.columns.tolist()
                    joblib.dump(self.ohe_categories, self.ohe_path) # Сохраняем OHE категории
                    print(f"OHE training categories saved to {self.ohe_path}")
                    ohe_df = current_ohe
                else:
                    if not self.ohe_categories:
                         print("Warning: OHE categories not loaded for prediction. Trying to load...")
                         self._load_ohe() # Попробуем загрузить
                    known_categories = self.ohe_categories.get(CATEGORICAL_FEATURE, [])
                    if not known_categories:
                        print("Error: OHE categories not available for prediction.")
                        return None
                    ohe_df = current_ohe.reindex(columns=known_categories, fill_value=0)
                    print(f"Applied OHE using known categories.")
            except Exception as e:
                print(f"Error during OHE: {e}")
                return None
        else:
            print(f"Warning: Categorical feature '{CATEGORICAL_FEATURE}' not found.")


        # 2. Выбираем и масштабируем числовые признаки
        num_features_available = [f for f in NUMERICAL_FEATURES if f in df.columns]
        if not num_features_available:
            print("No numerical features found.")
            # Если есть только OHE признаки, возможно, продолжить? Зависит от задачи.
            if ohe_df.empty: return None # Точно нечего обрабатывать
            else: scaled_numeric_df = pd.DataFrame(index=df.index) # Пустой, если нет числовых

        else:
             numeric_df = df[num_features_available].copy().fillna(0) # Заполняем NaN перед масштабированием

             if is_training:
                 print(f"Fitting scaler using numerical features: {num_features_available}")
                 self.scaler = StandardScaler() # Используем StandardScaler
                 try:
                     scaled_numeric = self.scaler.fit_transform(numeric_df)
                     joblib.dump(self.scaler, self.scaler_path) # Сохраняем scaler
                     print(f"Scaler saved to {self.scaler_path}")
                     # Сохраняем имена признаков (числовые + OHE)
                     self.feature_names = num_features_available + self.ohe_categories.get(CATEGORICAL_FEATURE, [])
                     scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=num_features_available, index=df.index)
                 except Exception as e:
                     print(f"Error during scaler fitting/transform: {e}")
                     return None
             else:
                 if not self.scaler or not self.feature_names:
                     print("Scaler/features not ready for prediction.")
                     # Попытка загрузить scaler, если он не был загружен ранее
                     if not self.scaler: self._load_model_and_scaler() 
                     if not self.scaler: return None # Если загрузка не удалась

                 # Используем только те числовые признаки, что были при обучении
                 expected_num_features = [f for f in self.feature_names if f in num_features_available]
                 print(f"Scaling numerical features using existing scaler: {expected_num_features}")
                 # Выбираем и упорядочиваем колонки согласно self.feature_names
                 numeric_df_aligned = numeric_df.reindex(columns=expected_num_features, fill_value=0)
                 try:
                     scaled_numeric = self.scaler.transform(numeric_df_aligned)
                     scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=expected_num_features, index=df.index)
                 except Exception as e:
                     print(f"Error during scaling transform: {e}")
                     return None


        # 3. Объединяем масштабированные числовые и OHE признаки
        final_df = pd.concat([scaled_numeric_df, ohe_df], axis=1)

        # 4. Убедимся, что порядок и состав колонок соответствует self.feature_names (важно для Keras)
        if self.feature_names: # Если feature_names уже определены (при обучении или загрузке)
            final_df = final_df.reindex(columns=self.feature_names, fill_value=0)
        else: # Если это первый проход (обучение без OHE), установим feature_names
            self.feature_names = final_df.columns.tolist()


        print(f"Prepared {len(final_df.columns)} features for Autoencoder. Columns: {final_df.columns.tolist()}")
        return final_df


    def train(self, db: Session, entity_type: str, limit: int):
        """Обучает модель автоэнкодера."""
        print(f"Autoencoder Training started for {entity_type}...")
        if entity_type != 'activity':
            print(f"Autoencoder currently only supports 'activity'. Skipping training for '{entity_type}'.")
            return

        _total_count, activities = crud.get_user_activities(db, limit=limit)
        if not activities: print("Autoencoder: No activity data found for training."); return

        df_engineered = engineer_features(activities)
        if df_engineered is None: print("Autoencoder: Feature engineering failed."); return

        df_prepared = self._prepare_features(df_engineered, is_training=True)
        if df_prepared is None: print("Autoencoder: Feature preparation failed."); return

        if not self.feature_names:
             print("Error: Feature names were not set during preparation.")
             return

        input_dim = df_prepared.shape[1]
        self.model = self._build_model(input_dim)

        try:
            # Обучаем модель
            history = self.model.fit(
                df_prepared, df_prepared, # Автоэнкодер учится восстанавливать вход
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1, # Используем часть данных для валидации
                shuffle=True,
                callbacks=[ # Остановка, если улучшение прекратилось
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
            )
            print(f"Autoencoder training completed. Final validation loss: {history.history['val_loss'][-1]:.4f}")
            self._save_model()

        except Exception as e:
            print(f"Error during Autoencoder model training: {e}")
            import traceback
            traceback.print_exc()


    def detect(self, db: Session, entity_type: str, limit: int) -> List[Dict[str, Any]]:
        """Обнаруживает аномалии с помощью обученного автоэнкодера."""
        print(f"Autoencoder Detection started for {entity_type}...")
        if entity_type != 'activity':
            print(f"Autoencoder currently only supports 'activity'. Skipping detection for '{entity_type}'.")
            return []

        if self.model is None or self.scaler is None or not self.feature_names:
            print("Autoencoder model/scaler not loaded or features unknown. Cannot detect.")
            # Пытаемся загрузить снова
            self._load_model_and_scaler()
            if self.model is None or self.scaler is None or not self.feature_names:
                 print("Failed to load model/scaler. Aborting detection.")
                 return []

        _total_count, activities = crud.get_user_activities(db, limit=limit)
        if not activities: print("Autoencoder: No activity data to detect."); return []

        df_engineered = engineer_features(activities)
        if df_engineered is None: print("Autoencoder: Feature engineering failed."); return []

        df_prepared = self._prepare_features(df_engineered, is_training=False)
        if df_prepared is None: print("Autoencoder: Feature preparation failed."); return []

        try:
            # Получаем реконструкции
            reconstructions = self.model.predict(df_prepared)
            # Считаем ошибку реконструкции (MSE) для каждой точки
            mse = np.mean(np.power(df_prepared - reconstructions, 2), axis=1)
            
            # Преобразуем обратно в DataFrame для удобства
            reconstruction_error_df = pd.DataFrame({'reconstruction_error': mse.values}, index=df_prepared.index)
            
            # --- Определяем порог для аномалий ---
            # Можно использовать фиксированный порог (self.anomaly_threshold)
            # Или динамический (например, перцентиль ошибки реконструкции)
            # Пока используем фиксированный
            threshold = self.anomaly_threshold 
            print(f"Using anomaly threshold (MSE): {threshold}")
            
            # Находим аномалии
            anomaly_mask = reconstruction_error_df['reconstruction_error'] > threshold
            anomalies_df = df_engineered[anomaly_mask]
            anomalies_errors = reconstruction_error_df[anomaly_mask]

            print(f"Autoencoder found {len(anomalies_df)} anomalies based on threshold {threshold:.4f}.")

            # Формируем результат
            output_anomalies = []
            for idx, row in anomalies_df.iterrows():
                activity_id = row.get('id')
                if activity_id is None: continue
                
                error_value = anomalies_errors.loc[idx, 'reconstruction_error']
                
                # --- Определение Severity ---
                # Можно сделать зависимым от величины ошибки относительно порога
                severity = "Medium" # По умолчанию
                if error_value > threshold * 2.0: # Пример: ошибка вдвое выше порога = High
                     severity = "High"
                     
                # --- ИСПРАВЛЕНО: Reason на русском --- 
                reason_str = (
                    f"Высокая ошибка реконструкции ({error_value:.4f}) с помощью Автоэнкодера, превышает порог ({threshold:.4f}). "
                    f"Указывает, что признаки точки необычны по сравнению с изученными нормальными паттернами."
                )
                # -------------------------------------

                output_anomalies.append({
                    "activity_id": int(activity_id),
                    "reason": reason_str,
                    "details": {"reconstruction_error": float(error_value), "threshold": threshold},
                    "severity": severity,
                    "anomaly_score": float(error_value) # Используем саму ошибку как score
                })

            return output_anomalies

        except Exception as e:
            print(f"Error during Autoencoder detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _save_model(self):
        """Сохраняет модель Keras и scaler."""
        if self.model:
            try:
                print(f"Saving Autoencoder Keras model to {self.model_path}")
                self.model.save(self.model_path) # Keras способ сохранения
                print("Autoencoder Keras model saved.")
            except Exception as e:
                print(f"Error saving Keras model: {e}")
        # Scaler и OHE сохраняются в _prepare_features при обучении

    def _load_model_and_scaler(self):
        """Загружает модель Keras, scaler и OHE категории."""
        # Загрузка модели Keras
        if os.path.exists(self.model_path):
            try:
                print(f"Loading Autoencoder Keras model from {self.model_path}")
                self.model = keras_models.load_model(self.model_path)
                print("Autoencoder Keras model loaded.")
                 # После загрузки модели нужно получить input_dim и feature_names
                 # Это немного сложнее, т.к. Keras не хранит их явно в .keras файле
                 # Можно сохранять их отдельно или пытаться извлечь из структуры модели
                 # Пока оставляем self.feature_names пустым при загрузке, 
                 # он будет установлен при первом вызове _prepare_features
                 
            except Exception as e:
                print(f"Error loading Keras model: {e}. Model set to None.")
                self.model = None
        else:
            print(f"Keras model file not found at {self.model_path}")
            self.model = None

        # Загрузка Scaler
        if os.path.exists(self.scaler_path):
            try:
                print(f"Loading Autoencoder scaler from {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                print("Autoencoder scaler loaded.")
            except Exception as e:
                print(f"Error loading scaler: {e}. Scaler set to None.")
                self.scaler = None
        else:
             print(f"Scaler file not found at {self.scaler_path}")
             self.scaler = None

        # Загрузка OHE категорий и feature_names (если они сохранялись)
        self._load_ohe() # OHE грузится отдельно
        
        # Важно: Установить feature_names после загрузки OHE и scaler
        # Это можно сделать, если OHE содержит список категорий, а scaler
        # имеет атрибут n_features_in_ или что-то подобное.
        # Или, как вариант, сохранять feature_names отдельным файлом.
        # Пока этот момент требует доработки для надежной загрузки.
        
        # Примерный способ восстановить feature_names (требует проверки):
        if self.scaler and self.ohe_categories:
            try:
                num_features = self.scaler.n_features_in_ # Количество числовых признаков
                # Предполагаем, что порядок в NUMERICAL_FEATURES соответствует обученному scaler
                num_feature_names = NUMERICAL_FEATURES[:num_features] 
                ohe_feature_names = self.ohe_categories.get(CATEGORICAL_FEATURE, [])
                self.feature_names = num_feature_names + ohe_feature_names
                print(f"Reconstructed feature names: {self.feature_names}")
            except Exception as e:
                 print(f"Could not reconstruct feature names after loading: {e}")
                 self.feature_names = [] # Сбрасываем, если не удалось
        else:
             self.feature_names = []


    def _load_ohe(self):
         """Загружает OHE категории."""
         if os.path.exists(self.ohe_path):
            try:
                print(f"Loading OHE categories from {self.ohe_path}")
                self.ohe_categories = joblib.load(self.ohe_path)
                print("OHE categories loaded.")
            except Exception as e:
                print(f"Error loading OHE categories: {e}.")
                self.ohe_categories = {}
         else:
            print(f"OHE categories file not found at {self.ohe_path}")
            self.ohe_categories = {}

# --- Дальнейший код в файле detector.py (фабрика get_detector и т.д.) --- 