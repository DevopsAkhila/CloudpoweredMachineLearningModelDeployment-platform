import json
import logging
import os
from datetime import datetime
from .config import Config

os.makedirs(Config.LOG_DIR, exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(Config.LOG_DIR, "mlserve.log"))
    handler.setFormatter(JsonFormatter())
    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.addHandler(console)
