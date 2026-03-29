"""Database operations for ML profiler."""

import os
from typing import Optional

from sqlalchemy import desc

from .models import ProfileResult, get_engine, get_session, init_db
from .profiler import ProfileMetrics


def get_database_url() -> str:
    """Get database URL from environment or use SQLite default."""
    return os.environ.get(
        "DATABASE_URL",
        "sqlite:///mlprofiler.db",
    )


class ProfileDB:
    """Database interface for profile results."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url()
        self.engine = get_engine(self.database_url)
        init_db(self.engine)

    def save(self, metrics: ProfileMetrics, notes: str = "") -> ProfileResult:
        """Save profiling metrics to database."""
        session = get_session(self.engine)
        result = ProfileResult(
            model_name=metrics.model_name,
            model_version=metrics.model_version,
            hardware_target=metrics.hardware_target,
            total_time_ms=metrics.total_time_ms,
            cpu_time_ms=metrics.cpu_time_ms,
            cuda_time_ms=metrics.cuda_time_ms,
            memory_allocated_mb=metrics.memory_allocated_mb,
            memory_reserved_mb=metrics.memory_reserved_mb,
            total_params=metrics.total_params,
            total_flops=metrics.total_flops,
            input_shape=metrics.input_shape,
            notes=notes,
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        session.close()
        return result

    def get_history(self, model_name: str, limit: int = 50) -> list[ProfileResult]:
        """Get profiling history for a model."""
        session = get_session(self.engine)
        results = (
            session.query(ProfileResult)
            .filter(ProfileResult.model_name == model_name)
            .order_by(desc(ProfileResult.created_at))
            .limit(limit)
            .all()
        )
        session.close()
        return results

    def get_all(self, limit: int = 100) -> list[ProfileResult]:
        """Get all profiling results."""
        session = get_session(self.engine)
        results = (
            session.query(ProfileResult)
            .order_by(desc(ProfileResult.created_at))
            .limit(limit)
            .all()
        )
        session.close()
        return results

    def get_models(self) -> list[str]:
        """Get list of unique model names."""
        session = get_session(self.engine)
        results = session.query(ProfileResult.model_name).distinct().all()
        session.close()
        return [r[0] for r in results]

    def compare_versions(self, model_name: str) -> list[ProfileResult]:
        """Get results grouped by version for comparison."""
        session = get_session(self.engine)
        results = (
            session.query(ProfileResult)
            .filter(ProfileResult.model_name == model_name)
            .order_by(ProfileResult.model_version, desc(ProfileResult.created_at))
            .all()
        )
        session.close()
        return results
