# Compatibility for SQLAlchemy 1.x and 2.x
try:
    from sqlalchemy.orm import declarative_base, relationship, sessionmaker
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.sql import func
from .config import Config

Base = declarative_base()
engine = create_engine(Config.DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    version = Column(String(32), nullable=False)
    framework = Column(String(32), nullable=False)  # sklearn | torch | onnx
    path = Column(Text, nullable=False)
    input_schema = Column(Text, nullable=True)  # JSON string
    state = Column(String(32), default="uploaded")  # uploaded | validated | deployed
    active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    model = relationship("Model", back_populates="versions")
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)

def init_db():
    Base.metadata.create_all(engine)
