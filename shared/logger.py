"""
shared/logger.py
Zentrales Logging-Modul für konsistentes Debugging und Error-Tracking über alle Microservices.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path("data/audit_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_service_logger(service_name: str) -> logging.Logger:
    """Erzeugt einen strukturierten Logger für den jeweiligen Microservice."""
    logger = logging.getLogger(service_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s | [%(levelname)s] | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Konsolen-Ausgabe für VS Code Terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Datei-Ausgabe für detailliertes Debugging
        file_handler = logging.FileHandler(LOG_DIR / f"{service_name}.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger