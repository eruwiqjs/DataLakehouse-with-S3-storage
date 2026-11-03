import os
import shutil
import kagglehub
import logging
from utils import load_config


def load_raw():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    kaggle_id = config["dataset"]["kaggle_id"]
    dest = config["paths"]["raw_file"]

    # Проверяем, есть ли уже файл
    if os.path.exists(dest):
        file_size = round(os.path.getsize(dest) / (1024 * 1024 * 1024), 2)
        logging.info(f"⚙️ Найден локальный датасет: {dest} ({file_size} ГБ). Пропускаем загрузку.")
        return

    try:
        logging.info(f"📥 Начало загрузки датасета с Kaggle: {kaggle_id}")
        path = kagglehub.dataset_download(kaggle_id)

        # Ищем CSV в загруженном датасете
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("❌ CSV-файлы не найдены в датасете Kaggle")

        os.makedirs(raw_dir, exist_ok=True)
        src = os.path.join(path, files[0])
        shutil.copy(src, dest)
        logging.info(f"✅ Датасет успешно помещён в {dest}")

    except Exception as e:
        logging.error(f"❌ Ошибка при загрузке датасета: {e}")
        logging.info("⚠️ Используй локальную копию, если она доступна.")


if __name__ == "__main__":
    load_raw()