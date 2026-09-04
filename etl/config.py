"""Configuración compartida y creación de conexiones a bases de datos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import URL, Engine, create_engine

load_dotenv()


class ConfigurationError(RuntimeError):
    """Indica que falta una variable necesaria para ejecutar el proyecto."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Datos mínimos para construir una URL de SQLAlchemy de forma segura."""

    driver: str
    host: str
    port: int
    database: str
    username: str
    password: str

    @classmethod
    def from_env(cls, prefix: str, driver: str, default_port: int) -> DatabaseSettings:
        values = {
            "host": os.getenv(f"{prefix}_HOST"),
            "database": os.getenv(f"{prefix}_DATABASE"),
            "username": os.getenv(f"{prefix}_USERNAME"),
            "password": os.getenv(f"{prefix}_PASSWORD"),
        }
        missing = [f"{prefix}_{key.upper()}" for key, value in values.items() if not value]
        if missing:
            raise ConfigurationError(
                "Faltan variables de entorno requeridas: " + ", ".join(missing)
            )

        raw_port = os.getenv(f"{prefix}_PORT", str(default_port))
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ConfigurationError(
                f"{prefix}_PORT debe ser un número entero; se recibió {raw_port!r}"
            ) from exc

        return cls(driver=driver, port=port, **values)  # type: ignore[arg-type]

    def sqlalchemy_url(self) -> URL:
        """Construye la URL sin concatenar ni escapar credenciales manualmente."""

        return URL.create(
            drivername=self.driver,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


@lru_cache(maxsize=1)
def get_crm_settings() -> DatabaseSettings:
    return DatabaseSettings.from_env("CRM", "mysql+pymysql", 3306)


@lru_cache(maxsize=1)
def get_pg_settings() -> DatabaseSettings:
    return DatabaseSettings.from_env("PG", "postgresql+psycopg2", 5432)


def get_crm_engine() -> Engine:
    """Crea un engine para el CRM MySQL."""

    return create_engine(get_crm_settings().sqlalchemy_url(), pool_pre_ping=True)


def get_pg_engine() -> Engine:
    """Crea un engine para PostgreSQL analítico."""

    return create_engine(get_pg_settings().sqlalchemy_url(), pool_pre_ping=True)
