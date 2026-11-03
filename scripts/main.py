import logging
from utils import load_config, setup_logger
from load_raw import load_raw
from clean_staging import clean_staging
from create_marts import create_marts
# Импортируем новую функцию
from upload_data import upload_to_minio_and_postgres

if __name__ == "__main__":
    config = load_config()
    setup_logger(config)

    logging.info("🚀 Запуск полного пайплайна E-Commerce Data\n")

    try:
        # 1. Скачивание (если нет файла)
        load_raw()
        # 2. Очистка
        clean_staging()
        # 3. Витрины
        create_marts()
        
        # 4. Загрузка в Инфраструктуру
        logging.info("\n--- ЭТАП 4: ЗАГРУЗКА В LAKEHOUSE ---")
        upload_to_minio_and_postgres()
        
        logging.info("\n🎉 Пайплайн завершён успешно! Данные в Minio и Postgres.")
        
    except Exception as e:
        logging.exception("❌ Ошибка при выполнении пайплайна: %s", str(e))