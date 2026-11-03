import os
import pandas as pd
import boto3
import glob
from sqlalchemy import create_engine, text
from trino.dbapi import connect
from botocore.client import Config

# ==========================================
# ⚙️ НАСТРОЙКИ
# ==========================================

FILE_RAW = 'data/raw/ecommerce_behavior.csv'
FILE_STAGING = 'data/staging/cleaned_data.parquet'
MART_DIR = 'data/mart'  # <--- ВАЖНО: Папка, где лежат твои готовые витрины

# Настройки MinIO
S3_CONF = {
    'endpoint_url': 'http://localhost:9000',
    'aws_access_key_id': 'admin',
    'aws_secret_access_key': 'password',
    'config': Config(signature_version='s3v4')
}
BUCKET_RAW = 'raw-layer'
BUCKET_WAREHOUSE = 'warehouse'

# Настройки Postgres
PG_CONN_STR = 'postgresql://postgres:password@localhost:5432/warehouse'

# Настройки Trino
TRINO_CONF = {
    'host': 'localhost',
    'port': 8080,
    'user': 'admin',
    'catalog': 'iceberg',
    'schema': 'db'
}

# ==========================================
# 🛠 ФУНКЦИИ
# ==========================================

def check_files():
    print("\n🔍 Проверка файлов...")
    if not os.path.exists(FILE_RAW):
        print(f"   ❌ ОШИБКА: Не найден исходный файл: {FILE_RAW}")
        return False
    # Проверяем папку с витринами
    if not os.path.exists(MART_DIR):
        print(f"   ⚠️ ПРЕДУПРЕЖДЕНИЕ: Папка с витринами не найдена: {MART_DIR}")
    else:
        files = os.listdir(MART_DIR)
        print(f"   📂 В папке {MART_DIR} найдено файлов: {len(files)}")
    return True

def ensure_buckets_exist():
    print(f"\n🌊 [Шаг 0] Проверка бакетов MinIO...")
    s3 = boto3.resource('s3', **S3_CONF)
    for b_name in [BUCKET_RAW, BUCKET_WAREHOUSE]:
        bucket = s3.Bucket(b_name)
        if not bucket.creation_date:
            try:
                s3.create_bucket(Bucket=b_name)
                print(f"   ✅ Бакет {b_name} создан.")
            except Exception as e:
                print(f"   ❌ Ошибка создания бакета {b_name}: {e}")

def step_1_upload_minio():
    print(f"\n🌊 [Шаг 1] Загрузка 'AS IS' в MinIO...")
    s3 = boto3.resource('s3', **S3_CONF)
    bucket = s3.Bucket(BUCKET_RAW)
    filename = os.path.basename(FILE_RAW)
    try:
        bucket.upload_file(FILE_RAW, filename)
        print(f"   ✅ Успешно! Файл доступен в MinIO: {BUCKET_RAW}/{filename}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки: {e}")

def step_2_upload_iceberg():
    print(f"\n❄️ [Шаг 2] Загрузка данных в Iceberg (Trino)...")
    if not os.path.exists(FILE_STAGING):
        return

    try:
        df = pd.read_parquet(FILE_STAGING)
        df.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in df.columns]
        
        conn = connect(**TRINO_CONF)
        cur = conn.cursor()

        # 1. Создаем таблицу (типы данных)
        cols = []
        for col, dtype in df.dtypes.items():
            if 'int' in str(dtype): t = 'BIGINT'
            elif 'float' in str(dtype): t = 'DOUBLE'
            else: t = 'VARCHAR' # Даты тоже станут VARCHAR, это нормально для staging
            cols.append(f"{col} {t}")
        
        table_name = "staging_data"
        s3_location = f"s3a://{BUCKET_WAREHOUSE}/db/{table_name}"
        
        create_query = f"""
        CREATE TABLE IF NOT EXISTS iceberg.db.{table_name} (
            {', '.join(cols)}
        ) WITH (location = '{s3_location}', format = 'PARQUET')
        """
        try: cur.execute(f"DROP TABLE IF EXISTS iceberg.db.{table_name}")
        except: pass

        cur.execute(create_query)
        
        print("   ⏳ Вставляем данные в Trino...")
        
        # 2. Подготовка данных для вставки
        batch_size = 2000
        data = df.to_dict(orient='records')
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            values = []
            for row in batch:
                row_vals = []
                for v in row.values():
                    # --- FIX START: ИСПРАВЛЕНИЕ ДЛЯ ДАТ ---
                    # Если это строка ИЛИ дата/время (имеет метод isoformat или это Timestamp)
                    if isinstance(v, str) or 'timestamp' in str(type(v)).lower() or hasattr(v, 'strftime'):
                        # Экранируем кавычки и оборачиваем в кавычки
                        safe_v = str(v).replace("'", "''")
                        row_vals.append(f"'{safe_v}'")
                    # --- FIX END ---
                    elif v is None or pd.isna(v):
                        row_vals.append("NULL")
                    else:
                        row_vals.append(str(v))
                values.append(f"({','.join(row_vals)})")
            
            if values:
                cur.execute(f"INSERT INTO iceberg.db.{table_name} VALUES {','.join(values)}")
        
        print(f"   ✅ Данные загружены в таблицу 'iceberg.db.{table_name}'")
        
    except Exception as e:
        print(f"   ❌ Ошибка Trino: {e}")
def step_3_create_mart_sql():
    print(f"\n📊 [Шаг 3] Создание витрины SQL-запросом в Trino...")
    try:
        conn = connect(**TRINO_CONF)
        cur = conn.cursor()
        mart_name = "brand_analytics"
        s3_location = f"s3a://{BUCKET_WAREHOUSE}/db/{mart_name}"
        cur.execute(f"DROP TABLE IF EXISTS iceberg.db.{mart_name}")
        
        # Этот шаг важен для сдачи задания (показывает, что Trino умеет считать)
        sql = f"""
        CREATE TABLE iceberg.db.{mart_name} 
        WITH (location = '{s3_location}', format = 'PARQUET') AS
        SELECT brand, count(*) as events_count, sum(price) as total_revenue
        FROM iceberg.db.staging_data
        WHERE brand IS NOT NULL GROUP BY brand
        """
        cur.execute(sql)
        print(f"   ✅ Витрина 'iceberg.db.{mart_name}' создана через SQL!")
    except Exception as e:
        print(f"   ❌ Ошибка создания витрины SQL: {e}")

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ---
def step_4_upload_postgres():
    print(f"\n🐘 [Шаг 4] Массовая загрузка ВСЕХ витрин в Postgres...")
    
    if not os.path.exists(MART_DIR):
        print(f"   ⚠️ Папка {MART_DIR} не найдена. Нечего грузить.")
        return

    # Ищем все CSV и Parquet файлы в папке data/mart
    files = glob.glob(os.path.join(MART_DIR, "*.csv")) + glob.glob(os.path.join(MART_DIR, "*.parquet"))
    
    if not files:
        print("   ⚠️ В папке data/mart пусто!")
        return

    engine = create_engine(PG_CONN_STR)

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            table_name = os.path.splitext(filename)[0] # Имя файла без расширения = имя таблицы
            
            # Чистим имя таблицы (Postgres не любит пробелы)
            table_name = table_name.lower().replace(' ', '_').replace('-', '_')

            print(f"   ⏳ Обработка файла: {filename} -> таблица '{table_name}'")
            
            # Читаем в зависимости от формата
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)

            # Заливаем в Postgres
            with engine.begin() as connection:
                df.to_sql(table_name, con=connection, if_exists='replace', index=False)
            
            print(f"      ✅ Успешно загружено ({len(df)} строк)")
            
        except Exception as e:
            print(f"      ❌ Ошибка при загрузке {filename}: {e}")

if __name__ == "__main__":
    if check_files():
        ensure_buckets_exist()
        step_1_upload_minio()
        step_2_upload_iceberg()
        step_3_create_mart_sql()
        step_4_upload_postgres() # Теперь загрузит ВСЁ
        print("\n🏁 Работа завершена!")