import os
from typing import Optional, List
from pydantic import BaseModel


def _env(key: str, default=None):
    val = os.getenv(key)
    return val if val is not None else default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(key: str, default: int = 0) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


class Settings(BaseModel):
    HOST: str = _env("HOST", "127.0.0.1")
    PORT: int = _env_int("PORT", 8000)
    DEBUG: bool = _env_bool("DEBUG", False)

    OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)

    DATABASE_URL: str = _env("DATABASE_URL", "postgresql://localhost:5432/career_regret_ai")

    MODEL_PATH: str = _env("MODEL_PATH", "./models")
    ML_LEARNING_RATE: float = _env_float("ML_LEARNING_RATE", 0.001)
    ENSEMBLE_DL_WEIGHT: float = _env_float("ENSEMBLE_DL_WEIGHT", 0.7)
    ENSEMBLE_ML_WEIGHT: float = _env_float("ENSEMBLE_ML_WEIGHT", 0.3)
    DECAY_FACTOR: float = _env_float("DECAY_FACTOR", 0.95)
    TEMPORAL_DECAY: float = _env_float("TEMPORAL_DECAY", 0.99)
    MONTE_CARLO_SIMULATIONS: int = _env_int("MONTE_CARLO_SIMULATIONS", 1000)
    FEEDBACK_BATCH_SIZE: int = _env_int("FEEDBACK_BATCH_SIZE", 10)
    FEEDBACK_WEIGHT: float = _env_float("FEEDBACK_WEIGHT", 0.3)
    ENABLE_AB_TESTING: bool = _env_bool("ENABLE_AB_TESTING", True)

    CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    JWT_SECRET_KEY: Optional[str] = _env("JWT_SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = _env_int("REFRESH_TOKEN_EXPIRE_DAYS", 7)

    RATE_LIMIT_RPM: int = _env_int("RATE_LIMIT_RPM", 30)
    RATE_LIMIT_RPH: int = _env_int("RATE_LIMIT_RPH", 500)
    RATE_LIMIT_BURST: int = _env_int("RATE_LIMIT_BURST", 5)

    CORS_ORIGINS: Optional[str] = _env("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000")

    IP_BLACKLIST: Optional[str] = _env("IP_BLACKLIST")
    IP_WHITELIST: str = _env("IP_WHITELIST", "127.0.0.1,::1")
    TRUSTED_PROXIES: str = _env("TRUSTED_PROXIES", "127.0.0.1")

    MAX_LOGIN_ATTEMPTS: int = _env_int("MAX_LOGIN_ATTEMPTS", 5)
    LOCKOUT_DURATION_MINUTES: int = _env_int("LOCKOUT_DURATION_MINUTES", 15)

    MAX_REQUEST_SIZE_MB: int = _env_int("MAX_REQUEST_SIZE_MB", 1)

    ENABLE_API_KEYS: bool = _env_bool("ENABLE_API_KEYS", False)

    ADMIN_PASSWORD: Optional[str] = _env("ADMIN_PASSWORD")

    GITHUB_CLIENT_ID: Optional[str] = _env("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: Optional[str] = _env("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI: str = _env("GITHUB_REDIRECT_URI", "http://localhost:8000/api/auth/github/callback")

    CACHE_MAX_SIZE: int = _env_int("CACHE_MAX_SIZE", 500)
    CACHE_TTL: int = _env_int("CACHE_TTL", 300)

    class Config:
        extra = "ignore"

settings = Settings()

def validate_production_settings():
    warnings = []
    errors = []

    if not settings.DEBUG:
        if not settings.JWT_SECRET_KEY:
            errors.append("JWT_SECRET_KEY must be set in production")
        elif len(settings.JWT_SECRET_KEY) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")

        if not settings.CORS_ORIGINS:
            warnings.append("CORS_ORIGINS not set - API may be inaccessible from browsers")

        if not settings.ADMIN_PASSWORD:
            warnings.append("ADMIN_PASSWORD not set - using auto-generated password")

    return errors, warnings

_errors, _warnings = validate_production_settings()
for w in _warnings:
    print(f"WARNING: {w}")
for e in _errors:
    print(f"ERROR: {e}")
