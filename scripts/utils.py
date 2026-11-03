import logging
import yaml
from datetime import datetime


def load_config(path="scripts/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(config):
    log_file = config["logging"]["log_file"]
    log_level = config["logging"]["log_level"].upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("🧭 Логирование запущено: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
