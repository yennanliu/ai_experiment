"""SQLAlchemy models for ML profiling metrics."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class ProfileResult(Base):
    """Stores profiling results for ML models."""

    __tablename__ = "profile_results"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50), default="1.0.0")
    hardware_target = Column(String(100), default="cpu")

    # Performance metrics
    total_time_ms = Column(Float, nullable=False)
    cpu_time_ms = Column(Float)
    cuda_time_ms = Column(Float)
    memory_allocated_mb = Column(Float)
    memory_reserved_mb = Column(Float)

    # Model info
    total_params = Column(Integer)
    total_flops = Column(Float)
    input_shape = Column(String(255))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    def __repr__(self):
        return f"<ProfileResult(model={self.model_name}, time={self.total_time_ms:.2f}ms)>"


def get_engine(database_url: str):
    """Create database engine."""
    return create_engine(database_url)


def get_session(engine):
    """Create database session."""
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine):
    """Initialize database tables."""
    Base.metadata.create_all(engine)
