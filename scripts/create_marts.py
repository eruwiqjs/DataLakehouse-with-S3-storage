import os
import pandas as pd
import numpy as np
import logging
from utils import load_config


def create_marts():
    config = load_config()
    src = config["paths"]["staging_file"]
    mart_dir = config["paths"]["mart_dir"]

    if not os.path.exists(src):
        logging.error("❌ Нет очищенных данных! Выполни clean_staging.py перед этим шагом.")
        raise FileNotFoundError("Нет очищенных данных")

    os.makedirs(mart_dir, exist_ok=True)
    df = pd.read_parquet(src)
    logging.info("📊 Формирование витрин из %d строк...", len(df))

    # 1. ВИТРИНА: Пользовательская активность
    logging.info("👥 Создание витрины пользовательской активности...")
    user_activity_mart = create_user_activity_mart(df)
    user_activity_mart.to_parquet(os.path.join(mart_dir, "user_activity_mart.parquet"), index=False)
    user_activity_mart.to_csv(os.path.join(mart_dir, "user_activity_mart.csv"), index=False)
    logging.info("✅ Витрина пользовательской активности сохранена (%d пользователей)", len(user_activity_mart))

    # 2. ВИТРИНА: Товары и цены
    logging.info("💰 Создание витрины товаров и цен...")
    products_mart = create_products_mart(df)
    products_mart.to_parquet(os.path.join(mart_dir, "products_mart.parquet"), index=False)
    products_mart.to_csv(os.path.join(mart_dir, "products_mart.csv"), index=False)
    logging.info("✅ Витрина товаров и цен сохранена (%d товаров)", len(products_mart))

    # 3. ВИТРИНА: Бренды и категории
    logging.info("🏷️ Создание витрины брендов и категорий...")
    brands_mart = create_brands_mart(df)
    brands_mart.to_parquet(os.path.join(mart_dir, "brands_mart.parquet"), index=False)
    brands_mart.to_csv(os.path.join(mart_dir, "brands_mart.csv"), index=False)
    logging.info("✅ Витрина брендов и категорий сохранена (%d брендов)", len(brands_mart))

    # 4. ВИТРИНА: Покупки и конверсия (воронка продаж)
    logging.info("🔄 Создание витрины воронки продаж...")
    funnel_data = create_funnel_mart(df)
    
    # Сохраняем каждую часть воронки отдельно
    for table_name, table_data in funnel_data.items():
        if hasattr(table_data, 'to_parquet'):  # Проверяем, что это DataFrame
            table_data.to_parquet(os.path.join(mart_dir, f"funnel_{table_name}.parquet"), index=False)
            table_data.to_csv(os.path.join(mart_dir, f"funnel_{table_name}.csv"), index=False)
            logging.info("   ✅ Сохранена таблица: funnel_%s", table_name)
    
    logging.info("✅ Витрина воронки продаж сохранена")

    # 5. ВИТРИНА: Временные паттерны
    logging.info("⏰ Создание витрины временных паттернов...")
    time_patterns_data = create_time_patterns_mart(df)
    
    # Сохраняем каждую часть временных паттернов отдельно
    for table_name, table_data in time_patterns_data.items():
        if hasattr(table_data, 'to_parquet'):  # Проверяем, что это DataFrame
            table_data.to_parquet(os.path.join(mart_dir, f"time_{table_name}.parquet"), index=False)
            table_data.to_csv(os.path.join(mart_dir, f"time_{table_name}.csv"), index=False)
            logging.info("   ✅ Сохранена таблица: time_%s", table_name)
    
    logging.info("✅ Витрина временных паттернов сохранена")

    # 6. ОСНОВНЫЕ ФАКТ-ТАБЛИЦЫ (сохраняем для совместимости)
    logging.info("📋 Сохранение основных факт-таблиц...")
    
    # Факт-таблица событий
    fact_events = df[[
        "event_time", "event_type", "user_id", "product_id", 
        "category_code", "brand", "price", "user_session",
        "date", "day_of_week", "hour", "month"
    ]]
    fact_events.to_parquet(os.path.join(mart_dir, "fact_events.parquet"), index=False)
    fact_events.to_csv(os.path.join(mart_dir, "fact_events.csv"), index=False)
    
    # Справочник пользователей
    dim_users = df.groupby("user_id").agg(
        first_activity=("event_time", "min"),
        last_activity=("event_time", "max"),
        total_events=("event_type", "count")
    ).reset_index()
    dim_users.to_csv(os.path.join(mart_dir, "dim_users.csv"), index=False)
    
    # Справочник товаров
    dim_products = df.groupby("product_id").agg({
        "brand": "first",
        "category_code": "first",
        "price": ["mean", "min", "max"]
    }).reset_index()
    dim_products.columns = ["product_id", "brand", "category_code", "avg_price", "min_price", "max_price"]
    dim_products.to_csv(os.path.join(mart_dir, "dim_products.csv"), index=False)

    logging.info("🎉 Все витрины успешно созданы в %s", mart_dir)


def create_user_activity_mart(df):
    """Витрина пользовательской активности"""
    user_activity = df.groupby('user_id').agg({
        'event_time': ['count', 'min', 'max'],
        'event_type': lambda x: x.value_counts().to_dict(),
        'product_id': 'nunique',
        'brand': 'nunique',
        'category_code': 'nunique',
        'price': ['sum', 'mean', 'max']
    }).reset_index()
    
    # Упрощение мультииндекса
    user_activity.columns = [
        'user_id', 'total_events', 'first_activity', 'last_activity',
        'event_type_distribution', 'unique_products', 'unique_brands',
        'unique_categories', 'total_spent', 'avg_price', 'max_price'
    ]
    
    # Разбивка распределения событий по типам
    for event_type in ['view', 'cart', 'purchase']:
        user_activity[f'{event_type}_count'] = user_activity['event_type_distribution'].apply(
            lambda x: x.get(event_type, 0) if isinstance(x, dict) else 0
        )
    
    # Дополнительные метрики
    user_activity['session_duration_days'] = (
        user_activity['last_activity'] - user_activity['first_activity']
    ).dt.total_seconds() / (24 * 3600)
    
    user_activity['purchase_conversion_rate'] = (
        user_activity['purchase_count'] / user_activity['view_count']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    user_activity['avg_events_per_day'] = user_activity['total_events'] / np.maximum(user_activity['session_duration_days'], 1)
    
    user_activity['user_segment'] = pd.cut(
        user_activity['total_events'],
        bins=[0, 5, 20, 100, np.inf],
        labels=['Низкая активность', 'Средняя активность', 'Высокая активность', 'Очень высокая активность']
    )
    
    return user_activity.drop('event_type_distribution', axis=1)


def create_products_mart(df):
    """Витрина товаров и цен"""
    products_mart = df.groupby('product_id').agg({
        'event_time': 'count',
        'event_type': lambda x: x.value_counts().to_dict(),
        'user_id': 'nunique',
        'brand': 'first',
        'category_code': 'first',
        'price': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    products_mart.columns = [
        'product_id', 'total_events', 'event_type_distribution', 'unique_users',
        'brand', 'category', 'avg_price', 'min_price', 'max_price', 'price_std'
    ]
    
    # Разбивка событий по типам
    for event_type in ['view', 'cart', 'purchase']:
        products_mart[f'{event_type}_count'] = products_mart['event_type_distribution'].apply(
            lambda x: x.get(event_type, 0) if isinstance(x, dict) else 0
        )
    
    # Метрики эффективности товара
    products_mart['conversion_rate'] = (
        products_mart['purchase_count'] / products_mart['view_count']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    products_mart['cart_to_purchase_rate'] = (
        products_mart['purchase_count'] / products_mart['cart_count']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    products_mart['popularity_score'] = (
        products_mart['view_count'] * 0.3 + 
        products_mart['cart_count'] * 0.5 + 
        products_mart['purchase_count'] * 0.2
    )
    
    # Сегментация товаров по цене
    products_mart['price_segment'] = pd.cut(
        products_mart['avg_price'],
        bins=[0, 50, 200, 500, np.inf],
        labels=['Бюджетный', 'Средний', 'Премиум', 'Люкс'],
        right=False
    )
    
    return products_mart.drop('event_type_distribution', axis=1)


def create_brands_mart(df):
    """Витрина брендов и категорий"""
    brands_mart = df.groupby('brand').agg({
        'event_time': 'count',
        'event_type': lambda x: x.value_counts().to_dict(),
        'user_id': 'nunique',
        'product_id': 'nunique',
        'category_code': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown',
        'price': ['mean', 'sum', 'std', 'median']
    }).reset_index()
    
    brands_mart.columns = [
        'brand', 'total_events', 'event_type_distribution', 'unique_users',
        'unique_products', 'top_category', 'avg_price', 'total_revenue', 'price_std', 'median_price'
    ]
    
    # Разбивка событий по типам
    for event_type in ['view', 'cart', 'purchase']:
        brands_mart[f'{event_type}_count'] = brands_mart['event_type_distribution'].apply(
            lambda x: x.get(event_type, 0) if isinstance(x, dict) else 0
        )
    
    # Метрики бренда
    brands_mart['conversion_rate'] = (
        brands_mart['purchase_count'] / brands_mart['view_count']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    brands_mart['avg_revenue_per_user'] = (
        brands_mart['total_revenue'] / brands_mart['unique_users']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    brands_mart['product_diversity'] = brands_mart['unique_products'] / brands_mart['total_events']
    
    # Рейтинг брендов
    brands_mart['brand_popularity_rank'] = brands_mart['total_events'].rank(ascending=False, method='dense')
    brands_mart['conversion_rank'] = brands_mart['conversion_rate'].rank(ascending=False, method='dense')
    
    return brands_mart.drop('event_type_distribution', axis=1)


def create_funnel_mart(df):
    """Витрина покупок и конверсии (воронка продаж)"""
    result = {}
    
    # Воронка по дням
    result['daily'] = df.groupby(['date', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Воронка по категориям
    result['category'] = df.groupby(['category_code', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Воронка по брендам
    result['brand'] = df.groupby(['brand', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Сводная статистика воронки
    total_views = len(df[df['event_type'] == 'view'])
    total_carts = len(df[df['event_type'] == 'cart'])
    total_purchases = len(df[df['event_type'] == 'purchase'])
    
    result['summary'] = pd.DataFrame({
        'metric': ['Просмотры', 'Добавления в корзину', 'Покупки', 
                  'Конверсия просмотр-корзина', 'Конверсия корзина-покупка', 'Общая конверсия'],
        'value': [
            total_views,
            total_carts, 
            total_purchases,
            total_carts / total_views if total_views > 0 else 0,
            total_purchases / total_carts if total_carts > 0 else 0,
            total_purchases / total_views if total_views > 0 else 0
        ]
    })
    
    return result


def create_time_patterns_mart(df):
    """Витрина временных паттернов"""
    result = {}
    
    # Активность по часам
    result['hourly'] = df.groupby(['hour', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Активность по дням недели
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_activity = df.groupby(['day_of_week', 'event_type']).size().unstack(fill_value=0).reset_index()
    weekday_activity['day_of_week'] = pd.Categorical(weekday_activity['day_of_week'], categories=weekday_order, ordered=True)
    result['weekday'] = weekday_activity.sort_values('day_of_week')
    
    # Активность по датам
    result['daily'] = df.groupby(['date', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Активность по месяцам
    result['monthly'] = df.groupby(['month', 'event_type']).size().unstack(fill_value=0).reset_index()
    
    # Эффективность по времени суток
    hour_conversion = df.groupby('hour').agg({
        'event_type': lambda x: (x == 'purchase').sum() / (x == 'view').sum() if (x == 'view').sum() > 0 else 0
    }).reset_index()
    hour_conversion.columns = ['hour', 'conversion_rate']
    result['hour_conversion'] = hour_conversion
    
    # Пиковые периоды
    peak_analysis = {
        'most_active_hour_overall': df['hour'].mode().iloc[0],
        'most_active_weekday': df['day_of_week'].mode().iloc[0],
        'weekend_activity_ratio': len(df[df['is_weekend']]) / len(df),
        'peak_purchase_hour': df[df['event_type'] == 'purchase']['hour'].mode().iloc[0] if len(df[df['event_type'] == 'purchase']) > 0 else None
    }
    result['peak_analysis'] = pd.DataFrame([peak_analysis])
    
    return result


if __name__ == "__main__":
    create_marts()