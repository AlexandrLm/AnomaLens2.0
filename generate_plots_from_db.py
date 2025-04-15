import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import os
import sys
import time

# --- Настройка Пути ---
# Добавляем корневую папку проекта в sys.path, чтобы импорты работали
# Предполагается, что скрипт запускается из корня проекта (AnomaLens2.0)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # Или os.path.dirname(current_dir) если скрипт в подпапке
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Импорты из вашего backend ---
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend import crud, models
    from backend.database import Base, DATABASE_URL # Используем URL из вашего database.py
    from backend.ml_service.common import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURE
except ImportError as e:
    print(f"Ошибка импорта модулей из backend: {e}")
    print("Убедитесь, что скрипт запущен из корневой директории проекта AnomaLens2.0,")
    print("и все зависимости из backend/requirements.txt установлены.")
    sys.exit(1)

# --- Константы ---
DATA_LIMIT = 10000 # Лимит на загрузку активностей (можно None для всех, но осторожно)
OUTPUT_DIR = "presentation_plots" # Папка для сохранения графиков
# Ограничение на кол-во признаков для PairPlot, чтобы не было слишком медленно/громоздко
PAIRPLOT_MAX_FEATURES = 5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Настройка SQLAlchemy ---
print(f"Подключение к базе данных: {DATABASE_URL}")
# connect_args={\"check_same_thread\": False} может быть не нужен вне FastAPI, но оставим для совместимости
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Функции для загрузки данных ---
def get_data_from_db(db_session, limit=DATA_LIMIT):
    print(f"Загрузка последних {limit or 'всех'} активностей...")
    start_time = time.time()
    _total_count, activities = crud.get_user_activities(db_session, limit=limit or 1_000_000_000) # Используем огромный лимит если None
    print(f"Загружено {len(activities)} активностей за {time.time() - start_time:.2f} сек.")

    if not activities:
        print("Нет активностей для анализа.")
        return None, None, None

    print("Выполнение инженерии признаков...")
    start_time = time.time()
    df_engineered = engineer_features(activities)
    print(f"Инженерия признаков завершена за {time.time() - start_time:.2f} сек.")

    if df_engineered is None or df_engineered.empty:
        print("Ошибка или нет данных после инженерии признаков.")
        return None, None, None

    # Добавим timestamp_dayofweek, если его нет после FE (на всякий случай)
    if 'timestamp' in df_engineered.columns and 'timestamp_dayofweek' not in df_engineered.columns:
        df_engineered['timestamp_dayofweek'] = df_engineered['timestamp'].dt.weekday

    print("Загрузка информации об аномалиях...")
    start_time = time.time()
    # Получаем все записи аномалий, связанные с загруженными активностями
    activity_ids = df_engineered['id'].tolist()
    anomalies_q = db_session.query(models.Anomaly)\
        .filter(models.Anomaly.entity_type == 'activity')\
        .filter(models.Anomaly.entity_id.in_(activity_ids))
    all_anomalies = anomalies_q.all()
    print(f"Загружено {len(all_anomalies)} записей об аномалиях за {time.time() - start_time:.2f} сек.")

    # Создаем удобную структуру для проверки аномальности
    anomaly_map = {} # {activity_id: {detector1, detector2, ...}}
    anomaly_details = {} # {activity_id: {detector1: score1, detector2: score2, ...}}

    for anom in all_anomalies:
        act_id = anom.entity_id
        detector = anom.detector_name
        if act_id not in anomaly_map:
            anomaly_map[act_id] = set()
            anomaly_details[act_id] = {}
        if detector:
            anomaly_map[act_id].add(detector)
            # Сохраняем скор, если он есть
            if anom.anomaly_score is not None:
                 anomaly_details[act_id][detector] = anom.anomaly_score
            elif detector == 'statistical_zscore' and anom.details and 'max_z_score' in anom.details:
                 # Для Z-Score, если нет скора, попробуем взять из details
                 anomaly_details[act_id][detector] = anom.details['max_z_score']


    # Добавляем флаги и детали аномалий в основной DataFrame
    df_engineered['is_anomaly_any'] = df_engineered['id'].apply(lambda x: x in anomaly_map)
    df_engineered['anomaly_detectors'] = df_engineered['id'].apply(lambda x: list(anomaly_map.get(x, set())))

    detectors_found = set(det for dets in anomaly_map.values() for det in dets)
    detector_flags = []
    for det_name in detectors_found:
        det_flag_col = f'is_anomaly_{det_name}'
        det_score_col = f'{det_name}_score'
        detector_flags.append(det_flag_col)
        df_engineered[det_flag_col] = df_engineered['id'].apply(lambda x: det_name in anomaly_map.get(x, set()))
        # Получаем скор, если он есть, иначе NaN
        df_engineered[det_score_col] = df_engineered['id'].apply(lambda x: anomaly_details.get(x, {}).get(det_name))


    print("Данные успешно загружены и объединены.")
    return df_engineered, detectors_found, detector_flags

# --- Основной код ---
db = SessionLocal()
try:
    df_data, detected_detector_names, detector_flags = get_data_from_db(db)

    if df_data is not None and not df_data.empty:
        print(f"\nОбнаружены детекторы в данных: {detected_detector_names}")
        print(f"Всего записей: {len(df_data)}")
        print(f"Всего найдено аномальных активностей (любым детектором): {df_data['is_anomaly_any'].sum()}")

        # Определяем доступные числовые признаки для графиков
        available_num_features = sorted([f for f in NUMERICAL_FEATURES if f in df_data.columns and df_data[f].nunique() > 1]) # Исключаем константы

        # --- Предобработка (для k-distance и DBSCAN) ---
        df_scaled = None
        if not available_num_features:
             print("Нет числовых признаков для масштабирования!")
        else:
            print(f"Масштабирование признаков: {available_num_features}")
            df_prepared = df_data[available_num_features].copy().fillna(0)
            # Проверка на наличие нечисловых данных перед масштабированием
            non_numeric_cols = df_prepared.select_dtypes(exclude=np.number).columns
            if len(non_numeric_cols) > 0:
                 print(f"Предупреждение: Нечисловые колонки найдены перед масштабированием: {non_numeric_cols}. Попытка приведения к числу...")
                 for col in non_numeric_cols:
                     df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce').fillna(0)

            scaler = StandardScaler()
            # Обернем в try-except на случай ошибок в данных
            try:
                df_scaled = scaler.fit_transform(df_prepared)
                print("Масштабирование завершено.")
            except Exception as scale_err:
                print(f"Ошибка при масштабировании: {scale_err}")
                df_scaled = None


        # --- Генерация Графиков ---
        plot_counter = 0
        plot_explanations = {} # Словарь для описаний

        # --- Базовые Распределения ---
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Распределение типов действий...")
        plt.figure(figsize=(10, 6))
        sns.countplot(y='action_type', data=df_data, order = df_data['action_type'].value_counts().index[:15], palette='viridis', hue='action_type', legend=False) # Топ 15
        plt.title('Распределение типов пользовательской активности (Топ 15)')
        plt.xlabel('Количество событий')
        plt.ylabel('Тип действия')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_action_distribution.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = ("Распределение типов действий",
                                            "Показывает, какие действия пользователи совершают чаще всего. Помогает понять общую картину активности.",
                                            "Слайд: Данные")

        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Активность по часам...")
        plt.figure(figsize=(10, 6))
        if 'timestamp_hour' in df_data.columns:
            hourly_counts = df_data.groupby('timestamp_hour')['id'].count().sort_index()
            hourly_counts.plot(kind='bar', color='skyblue')
            plt.title('Общая активность пользователей по часам дня')
            plt.xlabel('Час дня (0-23)')
            plt.ylabel('Количество событий')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--')
        else:
             plt.text(0.5, 0.5, 'Признак timestamp_hour не найден', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_activity_by_hour.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = ("Активность по часам",
                                            "Показывает пики и спады активности в течение суток. Может выявить необычную активность ночью.",
                                            "Слайд: Данные / Инженерия признаков")

        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Активность по дням недели...")
        plt.figure(figsize=(10, 6))
        if 'timestamp_dayofweek' in df_data.columns:
            daily_counts = df_data.groupby('timestamp_dayofweek')['id'].count().sort_index()
            daily_counts.index = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'] # Заменяем цифры на дни
            daily_counts.plot(kind='bar', color='lightcoral')
            plt.title('Общая активность пользователей по дням недели')
            plt.xlabel('День недели')
            plt.ylabel('Количество событий')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--')
        else:
             plt.text(0.5, 0.5, 'Признак timestamp_dayofweek не найден', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_activity_by_weekday.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = ("Активность по дням недели",
                                            "Показывает распределение активности по дням. Может выявить аномальную активность в выходные/будни.",
                                            "Слайд: Данные / Инженерия признаков")

        # --- Анализ Признаков ---
        if available_num_features:
            plot_counter += 1
            print(f"Генерация графика {plot_counter}: Pair Plot для признаков...")
            # Берем ограниченное количество признаков для скорости
            pairplot_features = available_num_features[:PAIRPLOT_MAX_FEATURES]
            pairplot_data = df_data[pairplot_features + ['is_anomaly_any']].copy()
            # Уменьшаем выборку, если данных слишком много, для ускорения pairplot
            if len(pairplot_data) > 2000:
                 print(f"  Уменьшаем выборку для Pair Plot до 2000 точек (было {len(pairplot_data)})")
                 pairplot_data = pairplot_data.sample(n=2000, random_state=42)

            plt.figure(figsize=(15, 15)) # Pairplot требует больше места
            try:
                sns.pairplot(pairplot_data, vars=pairplot_features, hue='is_anomaly_any', diag_kind='kde',
                             palette={True: 'red', False: 'grey'}, plot_kws={'alpha': 0.5})
                plt.suptitle(f'Диаграмма парных отношений для {len(pairplot_features)} признаков (цвет - аномалия)', y=1.02)
                plot_filename = f'plot_{plot_counter}_pairplot.png'
                plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
                plot_explanations[plot_filename] = (f"Pair Plot ({len(pairplot_features)} признака)",
                                                    "Показывает попарные диаграммы рассеяния для выбранных числовых признаков и их распределения (на диагонали). Помогает визуально оценить взаимосвязи и разделение нормальных/аномальных точек.",
                                                    "Слайд: Инженерия признаков / Анализ Результатов")
            except Exception as pp_err:
                print(f"  Ошибка при построении Pair Plot: {pp_err}")
                plot_explanations[f'plot_{plot_counter}_pairplot.png (Ошибка)'] = ("Pair Plot - Ошибка", str(pp_err), "-")
            finally:
                plt.close() # Закрываем в любом случае

            plot_counter += 1
            print(f"Генерация графика {plot_counter}: Тепловая карта корреляций...")
            correlation_matrix = df_data[available_num_features].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Тепловая карта корреляций числовых признаков')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plot_filename = f'plot_{plot_counter}_correlation_heatmap.png'
            plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
            plt.close()
            plot_explanations[plot_filename] = ("Тепловая карта корреляций",
                                                "Показывает коэффициент корреляции Пирсона между всеми парами числовых признаков. Помогает выявить линейные зависимости (синие - отрицательная, красные - положительная).",
                                                "Слайд: Инженерия признаков")

        # --- Графики, связанные с Аномалиями ---

        # График Scatter Plot для признаков + выделение аномалий (общий)
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Scatter Plot (Any Anomaly)...")
        # Выбираем два признака с наибольшей вариативностью (стандартным отклонением)
        feature_x, feature_y = df_data[available_num_features].std().nlargest(2).index.tolist() if len(available_num_features) >= 2 else (None, None)

        plt.figure(figsize=(10, 8))
        if feature_x and feature_y:
            palette = {True: 'red', False: 'grey'}
            legend_title = 'Аномалия (Любая)'
            sns.scatterplot(
                x=feature_x, y=feature_y, hue='is_anomaly_any',
                data=df_data, palette=palette, alpha=0.6
            )
            plt.title(f'{feature_x} vs {feature_y}\n(Красные точки - аномалии по любому детектору)')
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            try: # Лог. шкала если много выбросов
                if df_data[feature_x].max() > df_data[feature_x].median() * 100: plt.xscale('log')
                if df_data[feature_y].max() > df_data[feature_y].median() * 100: plt.yscale('log')
            except Exception: pass # Игнорируем ошибки логарифмирования
            plt.legend(title=legend_title, loc='best')
        else:
             plt.text(0.5, 0.5, f'Недостаточно признаков для Scatter Plot', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_scatter_any_anomaly.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = (f"Scatter Plot ({feature_x} vs {feature_y})",
                                            f"Показывает диаграмму рассеяния для двух признаков ('{feature_x}' и '{feature_y}'). Точки, обнаруженные хотя бы одним детектором как аномалии, выделены красным. Помогает увидеть, как аномалии распределены в пространстве признаков.",
                                            "Слайд: Анализ Результатов / Обзор Детекторов")

        # Box Plot для признаков
        if available_num_features:
            plot_counter += 1
            print(f"Генерация графика {plot_counter}: Box Plot признаков...")
            # Выбираем несколько ключевых признаков
            boxplot_features = [f for f in ['time_since_last_activity_ip', 'actions_in_last_5min_ip', 'failed_logins_in_last_15min_ip'] if f in df_data.columns]
            if boxplot_features:
                num_plots = len(boxplot_features)
                fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5), sharey=False)
                if num_plots == 1: axes = [axes] # Делаем итерируемым, если только один plot
                fig.suptitle('Распределение признаков для Нормальных vs Аномальных активностей', fontsize=16)
                for i, feature in enumerate(boxplot_features):
                    sns.boxplot(x='is_anomaly_any', y=feature, data=df_data, ax=axes[i],
                                palette=['lightblue', 'salmon'], # Список соответствует порядку [False, True]
                                hue='is_anomaly_any', legend=False)
                    axes[i].set_title(feature)
                    axes[i].set_xlabel('Аномалия')
                    axes[i].set_ylabel('Значение')
                    # Логарифмическая шкала Y, если большой разброс
                    try:
                        if df_data[feature].max() > df_data[feature].quantile(0.75) * 10: axes[i].set_yscale('log')
                    except Exception: pass
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                plot_filename = f'plot_{plot_counter}_boxplots_features.png'
                plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
                plot_explanations[plot_filename] = (f"Box Plot признаков ({', '.join(boxplot_features)})",
                                                    "Показывает распределение (медиана, квартили, выбросы) выбранных признаков отдельно для нормальных (False) и аномальных (True) активностей. Помогает увидеть, различаются ли распределения.",
                                                    "Слайд: Анализ Результатов / Обзор Детекторов")
            else:
                plot_explanations[f'plot_{plot_counter}_boxplots_features.png (Пропущено)'] = ("Box Plot признаков - Пропущено", "Не найдены признаки для построения.", "-")
            plt.close()


        # График 4: Пример гистограммы Z-Score (если есть данные)
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Гистограмма Z-Score...")
        zscore_col = 'statistical_zscore_score' # Имя колонки со скором
        plt.figure(figsize=(10, 6))
        if zscore_col in df_data.columns and df_data[zscore_col].notna().any():
            scores_to_plot = df_data[zscore_col].dropna()
            sns.histplot(scores_to_plot, kde=True, bins=50)
            # Можно взять порог из настроек, если возможно, или использовать 3
            z_threshold = 3.0 # Placeholder
            plt.axvline(z_threshold, color='red', linestyle='--', label=f'Примерный порог Z-Score = {z_threshold}')
            plt.axvline(-z_threshold, color='red', linestyle='--')
            plt.title('Распределение Z-оценок (из записей аномалий)')
            plt.xlabel('Z-оценка (или Max Z-score)')
            plt.ylabel('Частота')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Нет данных Z-Score для отображения', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_zscore_distribution.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = ("Распределение Z-Score",
                                            "Показывает гистограмму Z-оценок, сохраненных детектором Z-Score. Пунктирные линии - типичный порог (+/- 3). Помогает оценить, насколько 'аномальны' найденные точки.",
                                            "Слайд: Z-Score")

        # График 5: График k-расстояний для подбора eps DBSCAN
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: k-distance plot (DBSCAN)...")
        plt.figure(figsize=(10, 6))
        if df_scaled is not None and df_scaled.shape[0] > 0:
            k = 5 # Или взять из настроек? min_samples = k (+ 1)
            # Уменьшаем k, если точек меньше чем k
            actual_k = min(k, df_scaled.shape[0])
            if actual_k < k: print(f"  Уменьшено k до {actual_k}, т.к. точек всего {df_scaled.shape[0]}")

            if actual_k > 1: # NearestNeighbors требует n_neighbors > 1
                print(f"Расчет {actual_k}-расстояний...")
                try:
                    nbrs = NearestNeighbors(n_neighbors=actual_k).fit(df_scaled)
                    distances, indices = nbrs.kneighbors(df_scaled)
                    # Берем расстояние до k-го соседа (индекс k-1)
                    k_distances = np.sort(distances[:, actual_k-1], axis=0)
                    plt.plot(k_distances)
                    plt.title(f'График k-расстояний (k={actual_k}) для подбора eps (DBSCAN)')
                    plt.xlabel('Индекс точки (отсортировано по k-расстоянию)')
                    plt.ylabel(f'{actual_k}-е расстояние')
                    plt.grid(True, linestyle='--')
                    # !! ВАЖНО: Здесь нужно вручную найти "локоть" и указать его значение для eps !!
                    # plt.axhline(1.5, color='red', linestyle=':', label='Примерный кандидат для eps = 1.5')
                    # plt.legend()
                except Exception as nn_err:
                    print(f"  Ошибка при расчете k-расстояний: {nn_err}")
                    plt.text(0.5, 0.5, f'Ошибка при расчете k-расстояний: {nn_err}', horizontalalignment='center', verticalalignment='center', wrap=True)
            else:
                 plt.text(0.5, 0.5, f'Недостаточно соседей (k={actual_k}) для расчета', horizontalalignment='center', verticalalignment='center')

        else:
            plt.text(0.5, 0.5, 'Масштабированные данные отсутствуют или пусты', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_k_distance.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = ("График k-расстояний (DBSCAN)",
                                            f"Строится для подбора параметра 'eps' алгоритма DBSCAN. Для каждой точки находится расстояние до k-го соседа (здесь k={k}). Расстояния сортируются. 'Локоть' (резкий изгиб) на графике указывает на оптимальное значение 'eps'.",
                                            "Слайд: DBSCAN")

        # График 6: Scatter Plot + выделение аномалий (DBSCAN) - ИСПРАВЛЕНО
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Scatter Plot (DBSCAN)...")
        dbscan_flag = 'is_anomaly_dbscan' # Реальное имя флага
        plt.figure(figsize=(10, 8))
        if feature_x and feature_y:
            # Используем тот же feature_x, feature_y, что и для общего scatter plot
            hue_col = dbscan_flag if dbscan_flag in df_data.columns else 'is_anomaly_any'
            palette = {True: 'purple', False: 'grey'} if hue_col == dbscan_flag else {True: 'orange', False: 'grey'}
            legend_title = 'Аномалия (DBSCAN)' if hue_col == dbscan_flag else 'Аномалия (Любая)'

            sns.scatterplot(
                x=feature_x,
                y=feature_y,
                hue=hue_col,
                data=df_data,
                palette=palette,
                style=hue_col if hue_col != 'is_anomaly_any' else None, # Стиль, только если есть флаг DBSCAN
                # s=np.where(df_data[hue_col], 80, 40), # УБИРАЕМ динамический размер из-за ошибки
                alpha=0.6
            )
            plt.title(f'{feature_x} vs {feature_y}\n(Выделены аномалии: {legend_title})')
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            try: # Лог. шкала если много выбросов
                if df_data[feature_x].max() > df_data[feature_x].median() * 100: plt.xscale('log')
                if df_data[feature_y].max() > df_data[feature_y].median() * 100: plt.yscale('log')
            except Exception: pass
            plt.legend(title=legend_title, loc='best')
        else:
            plt.text(0.5, 0.5, f'Недостаточно признаков для Scatter Plot', horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_scatter_dbscan_anomalies.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = (f"Scatter Plot ({feature_x} vs {feature_y}, DBSCAN)",
                                            f"Показывает диаграмму рассеяния для признаков '{feature_x}' и '{feature_y}'. Точки, помеченные как аномалии (шум) детектором DBSCAN, выделены цветом/стилем. Помогает оценить, какие точки DBSCAN считает выбросами.",
                                            "Слайд: DBSCAN / Анализ Результатов")


        # График 7: Сравнение IF score и AE MSE (если есть данные)
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Сравнение IF и AE скоров...")
        if_score_col = 'isolation_forest_score' # Реальное имя скора IF
        ae_score_col = 'autoencoder_score' # Реальное имя скора AE (MSE)
        plt.figure(figsize=(10, 8))
        # Проверяем наличие обеих колонок и не-NaN значений в них
        if if_score_col in df_data.columns and ae_score_col in df_data.columns and \
           df_data[[if_score_col, ae_score_col]].dropna().shape[0] > 0:
            # Берем только строки, где оба скора не NaN
            plot_data = df_data[[if_score_col, ae_score_col, 'is_anomaly_any']].dropna()
            sns.scatterplot(
                x=if_score_col,
                y=ae_score_col,
                hue='is_anomaly_any', # Раскрашиваем точки, найденные хоть одним методом
                data=plot_data,
                palette={True: 'red', False: 'grey'},
                alpha=0.6
            )
            plt.title('Сравнение оценок аномальности: Isolation Forest vs Autoencoder')
            plt.xlabel('IF Score (чем ниже, тем аномальнее)')
            plt.ylabel('AE Ошибка реконструкции (MSE) (чем выше, тем аномальнее)')
            plt.legend(title='Аномалия (Любая)', loc='best')
            plt.grid(True, linestyle='--')
            plot_explanation_text = ("Сравнение IF Score vs AE MSE",
                                     "Показывает диаграмму рассеяния, где оси - это оценки аномальности от Isolation Forest (чем ниже, тем хуже) и Autoencoder (чем выше, тем хуже). Позволяет увидеть, согласуются ли детекторы в своих оценках.",
                                     "Слайд: Автоэнкодер / Анализ Результатов")
        else:
            plt.text(0.5, 0.5, 'Нет данных IF Score и/или AE Score для сравнения', horizontalalignment='center', verticalalignment='center')
            plot_explanation_text = ("Сравнение IF Score vs AE MSE - Пропущено",
                                     "Не найдены колонки со скорами для Isolation Forest и/или Autoencoder или в них нет данных.",
                                     "-")
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_if_vs_ae_scores.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = plot_explanation_text


        # График 8: Итоговое количество аномалий по детекторам
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Количество аномалий по детекторам...")
        plt.figure(figsize=(8, 5))
        if detector_flags:
            counts_dict = {flag.replace('is_anomaly_', ''): df_data[flag].sum() for flag in detector_flags}
            # Убираем детекторы с 0 аномалий
            counts_dict = {k: v for k, v in counts_dict.items() if v > 0}
            if counts_dict:
                counts_df = pd.DataFrame(list(counts_dict.items()), columns=['Detector', 'Count'])
                sns.barplot(x='Count', y='Detector', data=counts_df.sort_values('Count', ascending=False), palette='magma')
                plt.title('Общее число обнаруженных аномалий каждым детектором')
                plt.xlabel('Количество найденных аномалий')
                plt.ylabel('Детектор')
                plot_explanation_text = ("Количество аномалий по детекторам",
                                         "Столбчатая диаграмма, показывающая, сколько уникальных активностей было помечено как аномальные каждым из работавших детекторов.",
                                         "Слайд: Выводы / Сравнение методов")
            else:
                plt.text(0.5, 0.5, 'Ни один детектор не нашел аномалий', horizontalalignment='center', verticalalignment='center')
                plot_explanation_text = ("Количество аномалий по детекторам - Пропущено",
                                         "Ни один из детекторов не обнаружил аномалий в загруженных данных.",
                                         "-")
        else:
             plt.text(0.5, 0.5, 'Нет данных по флагам аномалий детекторов', horizontalalignment='center', verticalalignment='center')
             plot_explanation_text = ("Количество аномалий по детекторам - Пропущено",
                                      "Не удалось определить флаги аномалий для детекторов.",
                                      "-")

        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_anomaly_counts_by_detector.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = plot_explanation_text

        # --- Дополнительные графики ---

        # График 9: Временной ряд активности
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Временной ряд активности...")
        plt.figure(figsize=(12, 6))
        if 'timestamp' in df_data.columns:
             # Агрегируем по дням для наглядности
             activity_over_time = df_data.set_index('timestamp').resample('D')['id'].count()
             activity_over_time.plot(color='teal', lw=2)
             plt.title('Динамика общего количества активностей по дням')
             plt.xlabel('Дата')
             plt.ylabel('Количество событий')
             plt.grid(True, linestyle=':')
             plot_explanation_text = ("Временной ряд активности",
                                      "Показывает общее количество пользовательских событий (активностей) по дням. Помогает увидеть общие тренды, сезонность, а также резкие всплески или падения активности.",
                                      "Слайд: Данные")
        else:
            plt.text(0.5, 0.5, 'Нет данных timestamp для временного ряда', horizontalalignment='center', verticalalignment='center')
            plot_explanation_text = ("Временной ряд активности - Пропущено", "Отсутствует колонка timestamp.", "-")
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_activity_timeseries.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = plot_explanation_text

        # График 10: Распределение признака для Normal vs Anomaly
        plot_counter += 1
        feature_to_compare = 'time_since_last_activity_ip' # Пример признака
        print(f"Генерация графика {plot_counter}: Распределение '{feature_to_compare}'...")
        plt.figure(figsize=(10, 6))
        if feature_to_compare in df_data.columns:
            # --- ИСПРАВЛЕНИЕ: Удаляем NaN/inf перед построением графика KDE ---
            plot_data = df_data[[feature_to_compare, 'is_anomaly_any']].replace([np.inf, -np.inf], np.nan).dropna()
            if not plot_data.empty:
                sns.histplot(data=plot_data, x=feature_to_compare, hue='is_anomaly_any', kde=True,
                             palette={True: 'red', False: 'grey'}, log_scale=True) # Используем лог шкалу для времени
                plt.title(f'Сравнение распределения признака "{feature_to_compare}"\nдля Нормальных и Аномальных активностей')
                plt.xlabel(f'{feature_to_compare} (Log Scale)')
                plt.ylabel('Частота')
                plot_explanation_text = (f"Распределение '{feature_to_compare}' (Norm vs Anomaly)",
                                         f"Показывает гистограммы распределения признака '{feature_to_compare}' отдельно для нормальных (серые) и аномальных (красные) активностей (оси X в лог. масштабе). Помогает увидеть, насколько хорошо этот признак разделяет классы.",
                                         "Слайд: Анализ Результатов / Обзор Детекторов")
            else:
                plt.text(0.5, 0.5, f'Нет валидных данных (после удаления NaN/inf) для признака {feature_to_compare}', horizontalalignment='center', verticalalignment='center')
                plot_explanation_text = (f"Распределение '{feature_to_compare}' - Пропущено", f"Нет валидных данных (после удаления NaN/inf) для признака '{feature_to_compare}'.", "-")
        else:
             plt.text(0.5, 0.5, f'Признак {feature_to_compare} не найден', horizontalalignment='center', verticalalignment='center')
             plot_explanation_text = (f"Распределение '{feature_to_compare}' - Пропущено", f"Признак '{feature_to_compare}' отсутствует.", "-")
        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_feature_distribution_comparison.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = plot_explanation_text


        # График 11: Аномалии по часам дня
        plot_counter += 1
        print(f"Генерация графика {plot_counter}: Аномалии по часам...")
        plt.figure(figsize=(10, 6))
        if 'timestamp_hour' in df_data.columns and df_data['is_anomaly_any'].any():
            anomalies_hourly = df_data[df_data['is_anomaly_any']].groupby('timestamp_hour')['id'].count().sort_index()
            anomalies_hourly.plot(kind='bar', color='salmon')
            plt.title('Распределение аномальных активностей по часам дня')
            plt.xlabel('Час дня (0-23)')
            plt.ylabel('Количество аномалий')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--')
            plot_explanation_text = ("Аномалии по часам дня",
                                     "Показывает, в какие часы суток чаще всего обнаруживаются аномалии. Может указывать на временные паттерны аномального поведения (например, ночью).",
                                     "Слайд: Анализ Результатов")
        elif 'timestamp_hour' not in df_data.columns:
             plt.text(0.5, 0.5, 'Признак timestamp_hour не найден', horizontalalignment='center', verticalalignment='center')
             plot_explanation_text = ("Аномалии по часам дня - Пропущено", "Отсутствует признак 'timestamp_hour'.", "-")
        else:
            plt.text(0.5, 0.5, 'Аномалии не найдены', horizontalalignment='center', verticalalignment='center')
            plot_explanation_text = ("Аномалии по часам дня - Пропущено", "В загруженных данных нет аномалий.", "-")

        plt.tight_layout()
        plot_filename = f'plot_{plot_counter}_anomalies_by_hour.png'
        plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
        plt.close()
        plot_explanations[plot_filename] = plot_explanation_text

        print(f"\n{plot_counter} Графиков сгенерированы и сохранены в папку: {OUTPUT_DIR}")

        # --- Создание файла с описаниями ---
        explanation_filename = os.path.join(OUTPUT_DIR, "plots_explanation.txt")
        print(f"Создание файла с описаниями: {explanation_filename}")
        with open(explanation_filename, "w", encoding="utf-8") as f:
            f.write("Описание сгенерированных графиков для презентации AnomaLens 2.0\n")
            f.write("=================================================================\n\n")
            for i, (plot_file, (title, description, slide)) in enumerate(plot_explanations.items()):
                f.write(f"График {i+1}: {plot_file}\n")
                f.write(f"  Название: {title}\n")
                f.write(f"  Описание: {description}\n")
                f.write(f"  Возможный слайд: {slide}\n")
                f.write("-" * 60 + "\n\n")
        print("Файл с описаниями успешно создан.")

    else:
        print("Не удалось загрузить или обработать данные из БД. Графики не сгенерированы.")

except Exception as e:
    print(f"\nПроизошла глобальная ошибка: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'db' in locals() and db.is_active:
        print("Закрытие сессии БД.")
        db.close()
    elif 'db' in locals():
         print("Сессия БД уже была закрыта или неактивна.")

