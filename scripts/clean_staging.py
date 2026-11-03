import os
import pandas as pd
import numpy as np
import logging
from utils import load_config
import matplotlib.pyplot as plt
import seaborn as sns


# Настройка визуализации
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_eda_plots(df, staging_dir):
    """Создание улучшенных визуализаций для EDA"""
    
    # 1. Распределение событий по типам
    plt.figure(figsize=(10, 6))
    event_counts = df['event_type'].value_counts()
    plt.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Распределение типов событий')
    plt.savefig(os.path.join(staging_dir, 'event_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. УЛУЧШЕННЫЙ: Активность по времени с разбивкой по типам событий
    plt.figure(figsize=(15, 8))
    
    # Создаем временные метрики
    df['date'] = df['event_time'].dt.date
    df['day_of_week'] = df['event_time'].dt.day_name()
    df['hour'] = df['event_time'].dt.hour
    
    # Активность по дням с разбивкой по типам событий
    daily_activity = df.groupby(['date', 'event_type']).size().unstack(fill_value=0)
    
    plt.subplot(2, 1, 1)
    daily_activity.plot(kind='line', alpha=0.8, linewidth=2)
    plt.title('Активность пользователей по дням (по типам событий)')
    plt.xlabel('Дата')
    plt.ylabel('Количество событий')
    plt.legend(title='Тип события')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Активность по дням недели
    plt.subplot(2, 1, 2)
    weekday_activity = df.groupby(['day_of_week', 'event_type']).size().unstack(fill_value=0)
    # Упорядочиваем дни недели
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_activity = weekday_activity.reindex(weekdays_order)
    weekday_activity.plot(kind='bar', alpha=0.8)
    plt.title('Активность по дням недели')
    plt.xlabel('День недели')
    plt.ylabel('Количество событий')
    plt.legend(title='Тип события')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(staging_dir, 'daily_activity_improved.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Топ-10 самых популярных брендов
    plt.figure(figsize=(12, 6))
    top_brands = df['brand'].value_counts().head(10)
    sns.barplot(x=top_brands.values, y=top_brands.index)
    plt.title('Топ-10 самых популярных брендов')
    plt.xlabel('Количество событий')
    plt.tight_layout()
    plt.savefig(os.path.join(staging_dir, 'top_brands.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. УЛУЧШЕННЫЙ: Анализ цен с фильтрацией нулевых и аномальных значений
    plt.figure(figsize=(15, 10))
    
    # Фильтруем цены для анализа (убираем нули и выбросы)
    prices_non_zero = df[df['price'] > 0]['price']
    prices_reasonable = prices_non_zero[prices_non_zero <= prices_non_zero.quantile(0.95)]  # Убираем топ 5% как выбросы
    
    plt.subplot(2, 2, 1)
    # Распределение всех цен (включая нули)
    df['price'].hist(bins=50, alpha=0.7, color='red')
    plt.title('Распределение всех цен (включая нулевые)')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Цена = 0')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    # Распределение ненулевых цен
    prices_non_zero.hist(bins=50, alpha=0.7, color='blue')
    plt.title('Распределение ненулевых цен')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    
    plt.subplot(2, 2, 3)
    # Распределение без выбросов
    prices_reasonable.hist(bins=50, alpha=0.7, color='green')
    plt.title('Распределение цен (без выбросов)')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    
    plt.subplot(2, 2, 4)
    # Логарифмическая шкала для ненулевых цен
    np.log1p(prices_non_zero).hist(bins=50, alpha=0.7, color='orange')
    plt.title('Распределение ненулевых цен (логарифм)')
    plt.xlabel('log(Цена + 1)')
    plt.ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig(os.path.join(staging_dir, 'price_distribution_improved.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. УЛУЧШЕННЫЙ: Активность по часам с разбивкой по типам событий
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    # Общая активность по часам
    hourly_activity = df.groupby('hour').size()
    hourly_activity.plot(kind='bar', alpha=0.7, color='skyblue')
    plt.title('Общая активность пользователей по часам')
    plt.xlabel('Час дня')
    plt.ylabel('Количество событий')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Активность по часам с разбивкой по типам событий
    hourly_activity_by_type = df.groupby(['hour', 'event_type']).size().unstack(fill_value=0)
    hourly_activity_by_type.plot(kind='line', alpha=0.8, linewidth=2)
    plt.title('Активность по часам (по типам событий)')
    plt.xlabel('Час дня')
    plt.ylabel('Количество событий')
    plt.legend(title='Тип события')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(staging_dir, 'hourly_activity_improved.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. ДОПОЛНИТЕЛЬНЫЙ: Анализ товаров с нулевой ценой
    plt.figure(figsize=(12, 8))
    
    zero_price_data = df[df['price'] == 0]
    non_zero_price_data = df[df['price'] > 0]
    
    plt.subplot(2, 2, 1)
    # Распределение типов событий для товаров с нулевой ценой
    if len(zero_price_data) > 0:
        zero_price_data['event_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Типы событий для товаров с ценой = 0')
    
    plt.subplot(2, 2, 2)
    # Сравнение распределения событий
    event_comparison = pd.DataFrame({
        'Цена = 0': zero_price_data['event_type'].value_counts(),
        'Цена > 0': non_zero_price_data['event_type'].value_counts()
    }).fillna(0)
    event_comparison.plot(kind='bar', alpha=0.7)
    plt.title('Сравнение типов событий по ценам')
    plt.xlabel('Тип события')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    # Топ бренды с нулевой ценой
    if len(zero_price_data) > 0:
        zero_price_data['brand'].value_counts().head(10).plot(kind='bar', alpha=0.7, color='red')
        plt.title('Топ-10 брендов с ценой = 0')
        plt.xlabel('Бренд')
        plt.ylabel('Количество событий')
        plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    # Топ категории с нулевой ценой
    if len(zero_price_data) > 0 and 'category_code' in zero_price_data.columns:
        zero_price_data['category_code'].value_counts().head(10).plot(kind='bar', alpha=0.7, color='orange')
        plt.title('Топ-10 категорий с ценой = 0')
        plt.xlabel('Категория')
        plt.ylabel('Количество событий')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(staging_dir, 'zero_price_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_eda_report(df, staging_dir):
    """Генерация улучшенного текстового отчета EDA"""
    
    report = []
    report.append("=" * 60)
    report.append("📊 ОТЧЕТ EDA - РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    report.append("=" * 60)
    
    # Основная информация о датасете
    report.append("\n1. ОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ:")
    report.append(f"   • Всего записей: {len(df):,}")
    report.append(f"   • Период данных: {df['event_time'].min()} - {df['event_time'].max()}")
    report.append(f"   • Всего уникальных пользователей: {df['user_id'].nunique():,}")
    report.append(f"   • Всего уникальных товаров: {df['product_id'].nunique():,}")
    report.append(f"   • Всего уникальных брендов: {df['brand'].nunique():,}")
    
    # Информация о типах событий
    report.append("\n2. ТИПЫ СОБЫТИЙ:")
    event_stats = df['event_type'].value_counts()
    for event_type, count in event_stats.items():
        percentage = (count / len(df)) * 100
        report.append(f"   • {event_type}: {count:,} ({percentage:.1f}%)")
    
    # АНАЛИЗ ЦЕН - ОСОБОЕ ВНИМАНИЕ НА НУЛЕВЫЕ ЦЕНЫ
    report.append("\n3. УГЛУБЛЕННЫЙ АНАЛИЗ ЦЕН:")
    zero_price_count = len(df[df['price'] == 0])
    zero_price_percentage = (zero_price_count / len(df)) * 100
    report.append(f"   • Товаров с ценой = 0: {zero_price_count:,} ({zero_price_percentage:.2f}%)")
    
    if zero_price_count > 0:
        zero_price_events = df[df['price'] == 0]['event_type'].value_counts()
        report.append("   • Распределение событий для товаров с ценой = 0:")
        for event_type, count in zero_price_events.items():
            pct = (count / zero_price_count) * 100
            report.append(f"     - {event_type}: {count:,} ({pct:.1f}%)")
    
    # Статистика по ненулевым ценам
    non_zero_prices = df[df['price'] > 0]['price']
    if len(non_zero_prices) > 0:
        report.append(f"   • Медиана цены (без нулей): {non_zero_prices.median():.2f}")
        report.append(f"   • Средняя цена (без нулей): {non_zero_prices.mean():.2f}")
        report.append(f"   • Минимальная цена (без нулей): {non_zero_prices.min():.2f}")
        report.append(f"   • Максимальная цена (без нулей): {non_zero_prices.max():.2f}")
        report.append(f"   • Стандартное отклонение: {non_zero_prices.std():.2f}")
    
    # Анализ временных паттернов
    report.append("\n4. ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ:")
    report.append(f"   • Самый активный час: {df['hour'].mode().iloc[0]}:00")
    report.append(f"   • Самый активный день недели: {df['day_of_week'].mode().iloc[0]}")
    
    # Анализ категорий
    report.append("\n5. АНАЛИЗ КАТЕГОРИЙ:")
    if 'category_code' in df.columns:
        top_categories = df['category_code'].value_counts().head(5)
        report.append("   • Топ-5 категорий:")
        for i, (category, count) in enumerate(top_categories.items(), 1):
            report.append(f"     {i}. {category}: {count:,}")
    
    # Анализ брендов
    report.append("\n6. АНАЛИЗ БРЕНДОВ:")
    top_brands = df['brand'].value_counts().head(5)
    report.append("   • Топ-5 брендов:")
    for i, (brand, count) in enumerate(top_brands.items(), 1):
        percentage = (count / len(df)) * 100
        report.append(f"     {i}. {brand}: {count:,} ({percentage:.1f}%)")
    
    # Пропущенные значения
    report.append("\n7. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    missing_data = df.isnull().sum()
    for column, missing_count in missing_data.items():
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            report.append(f"   • {column}: {missing_count:,} ({percentage:.2f}%)")
    
    # ГИПОТЕЗЫ ПО НУЛЕВЫМ ЦЕНАМ
    report.append("\n8. ГИПОТЕЗЫ ПО НУЛЕВЫМ ЦЕНАМ:")
    report.append("   • Бесплатные товары или образцы")
    report.append("   • Ошибки в данных или сбои при загрузке цен")
    report.append("   • Акционные товары (распродажи, промо-акции)")
    report.append("   • Тестовые или демонстрационные товары")
    
    report.append("\n" + "=" * 60)
    report.append("📈 УЛУЧШЕННЫЕ ВИЗУАЛИЗАЦИИ СОХРАНЕНЫ В ПАПКУ STAGING")
    report.append("=" * 60)
    
    # Сохранение отчета
    report_path = os.path.join(staging_dir, 'eda_report_improved.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Вывод в лог
    for line in report:
        logging.info(line)
    
    return report

def clean_staging():
    """Очистка и подготовка данных из RAW слоя для STAGING слоя"""
    config = load_config()
    src = config["paths"]["raw_file"]
    dst = config["paths"]["staging_file"]
    staging_dir = config["paths"]["staging_dir"]

    # Режим тестовой обработки
    fast_test = config.get("mode", {}).get("fast_test", False)
    chunksize = 1_000_000 if not fast_test else 100_000
    nrows = None if not fast_test else 500_000

    if not os.path.exists(src):
        logging.error("❌ Файл не найден: %s", src)
        raise FileNotFoundError("Нет исходного CSV")

    os.makedirs(staging_dir, exist_ok=True)
    logging.info("🧹 Очистка данных начата (режим: %s)", "FAST TEST" if fast_test else "FULL")

    clean_data = []
    processed_rows = 0

    try:
        for i, chunk in enumerate(pd.read_csv(src, chunksize=chunksize, nrows=nrows)):
            logging.info("📦 Обработка чанка #%d (%d строк)", i + 1, len(chunk))

            # БАЗОВАЯ ОЧИСТКА ДАННЫХ
            # Удаление строк с критически важными пропущенными значениями
            initial_count = len(chunk)
            chunk = chunk.dropna(subset=["event_time", "event_type", "user_id", "product_id"])
            logging.info("   Удалено строк с пропусками в ключевых полях: %d", initial_count - len(chunk))

            # ОЧИСТКА И СТАНДАРТИЗАЦИЯ ТЕКСТОВЫХ ПОЛЕЙ
            # Бренды: приведение к нижнему регистру и очистка
            chunk["brand"] = chunk["brand"].astype(str).str.lower().str.strip()
            chunk["brand"] = chunk["brand"].replace(['nan', 'none', 'null', ''], 'unknown')
            
            # Категории: приведение к нижнему регистру и очистка
            if 'category_code' in chunk.columns:
                chunk["category_code"] = chunk["category_code"].astype(str).str.lower().str.strip()
                chunk["category_code"] = chunk["category_code"].replace(['nan', 'none', 'null', ''], 'unknown')
            
            # ОЧИСТКА И ВАЛИДАЦИЯ ЦЕН
            # Замена отрицательных цен на 0
            chunk["price"] = chunk["price"].clip(lower=0)
            
            # Обработка аномально высоких цен (выбросов)
            price_q95 = chunk["price"].quantile(0.95)
            high_price_count = len(chunk[chunk["price"] > price_q95 * 10])  # Цены > 10*95% перцентиля
            if high_price_count > 0:
                logging.info("   Обнаружено аномально высоких цен: %d", high_price_count)
                # Можно заменить на перцентиль или оставить как есть в зависимости от бизнес-логики
            
            # ВАЛИДАЦИЯ ТИПОВ СОБЫТИЙ
            valid_event_types = ['view', 'cart', 'purchase']
            invalid_events = chunk[~chunk["event_type"].isin(valid_event_types)]
            if len(invalid_events) > 0:
                logging.warning("   Обнаружены невалидные типы событий: %s", invalid_events["event_type"].unique())
                chunk = chunk[chunk["event_type"].isin(valid_event_types)]

            # ОБРАБОТКА ДУБЛИКАТОВ
            initial_count = len(chunk)
            chunk = chunk.drop_duplicates(subset=["event_time", "user_id", "product_id", "event_type"])
            logging.info("   Удалено дубликатов: %d", initial_count - len(chunk))

            # ДОБАВЛЕНИЕ ВСПОМОГАТЕЛЬНЫХ ПОЛЕЙ ДЛЯ ВИМАРИН
            # Временные характеристики
            chunk["event_time"] = pd.to_datetime(chunk["event_time"], errors="coerce")
            chunk = chunk.dropna(subset=["event_time"])  # Удаление строк с невалидными датами
            
            chunk["date"] = chunk["event_time"].dt.date
            chunk["day_of_week"] = chunk["event_time"].dt.day_name()
            chunk["hour"] = chunk["event_time"].dt.hour
            chunk["month"] = chunk["event_time"].dt.month
            chunk["week"] = chunk["event_time"].dt.isocalendar().week
            
            # Флаги для анализа
            chunk["is_weekend"] = chunk["day_of_week"].isin(['Saturday', 'Sunday'])
            chunk["is_zero_price"] = chunk["price"] == 0
            chunk["is_high_price"] = chunk["price"] > chunk["price"].quantile(0.95)
            
            # Извлечение информации из категорий (если есть иерархия)
            if 'category_code' in chunk.columns:
                category_parts = chunk["category_code"].str.split('.', expand=True)
                for i in range(min(3, category_parts.shape[1])):
                    chunk[f"category_level_{i+1}"] = category_parts[i].fillna('unknown')

            clean_data.append(chunk)
            processed_rows += len(chunk)

        if not clean_data:
            logging.warning("⚠️ После очистки данные отсутствуют.")
            return

        # ОБЪЕДИНЕНИЕ ВСЕХ ЧАНКОВ
        df = pd.concat(clean_data, ignore_index=True)
        
        # ФИНАЛЬНАЯ ОЧИСТКА ПОСЛЕ ОБЪЕДИНЕНИЯ
        # Удаление тестовых/системных пользователей (если есть)
        if 'user_id' in df.columns:
            # Пример: удаление пользователей с подозрительно высокой активностью
            user_activity = df['user_id'].value_counts()
            suspicious_users = user_activity[user_activity > user_activity.quantile(0.999)].index
            if len(suspicious_users) > 0:
                logging.info("   Обнаружены пользователи с аномальной активностью: %d", len(suspicious_users))
                # df = df[~df['user_id'].isin(suspicious_users)]  # Раскомментировать при необходимости

        logging.info("📊 Статистика после очистки:")
        logging.info("   Всего строк: %d", len(df))
        logging.info("   Уникальных пользователей: %d", df['user_id'].nunique())
        logging.info("   Уникальных товаров: %d", df['product_id'].nunique())
        logging.info("   Уникальных брендов: %d", df['brand'].nunique())
        logging.info("   Распределение событий: %s", df['event_type'].value_counts().to_dict())
        logging.info("   Товаров с нулевой ценой: %d (%.2f%%)", 
                    len(df[df['price'] == 0]), 
                    len(df[df['price'] == 0]) / len(df) * 100)

        # СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ И ОТЧЕТОВ
        create_eda_plots(df, staging_dir)
        generate_eda_report(df, staging_dir)

        # СОХРАНЕНИЕ ОЧИЩЕННЫХ ДАННЫХ
        df.to_parquet(dst, index=False, compression='snappy')
        
        # ДОПОЛНИТЕЛЬНО: Сохранение основных метрик для мониторинга качества данных
        quality_metrics = {
            'total_rows': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_products': df['product_id'].nunique(),
            'unique_brands': df['brand'].nunique(),
            'zero_price_count': len(df[df['price'] == 0]),
            'data_quality_score': (len(df) / processed_rows) * 100 if processed_rows > 0 else 0
        }
        
        metrics_path = os.path.join(staging_dir, 'data_quality_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(quality_metrics, f, indent=2, ensure_ascii=False)
        
        logging.info("✅ Очистка завершена. Сохранено %d строк в %s", len(df), dst)
        logging.info("📊 Всего обработано строк: %d", processed_rows)
        logging.info("📈 Качество данных: %.2f%%", quality_metrics['data_quality_score'])

    except pd.errors.EmptyDataError:
        logging.error("❌ Файл пуст или повреждён: %s", src)
    except Exception as e:
        logging.exception("❌ Ошибка при обработке данных: %s", e)


if __name__ == "__main__":
    clean_staging()