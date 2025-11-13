import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
    DB_URL = os.getenv("DB_URL", "sqlite:///mlserve.db")
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
    AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
    AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin123")
    MAX_REQ_BODY_MB = int(os.getenv("MAX_REQ_BODY_MB", "5"))
    MAX_JSON_KEYS = int(os.getenv("MAX_JSON_KEYS", "200"))
