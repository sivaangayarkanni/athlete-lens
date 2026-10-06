from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

if _is_sqlite and ":memory:" not in settings.database_url:
    # sqlite:////tmp/x.db -> /tmp/x.db ; sqlite:///./x.db -> ./x.db
    db_path = settings.database_url.split("sqlite:///", 1)[-1]
    if db_path:
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True, pool_pre_ping=True)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after v1.1. create_all() never alters existing tables, so add them in place.
_ADDED_COLUMNS = {
    "athletes": {"injury_history": "TEXT DEFAULT ''", "growth_cm": "FLOAT", "nordic_program": "INTEGER DEFAULT 0",
                 "adductor_program": "INTEGER DEFAULT 0"},
    "sessions": {"session_type": "VARCHAR(16) DEFAULT 'training'", "asymmetry_pct": "FLOAT"},
    "predictions": {"region_risks": "TEXT", "top_region": "VARCHAR(24)"},
}


def migrate(eng=None) -> list[str]:
    """Idempotent, additive migration for older databases. Returns the columns added."""
    eng = eng or engine
    added = []
    insp = inspect(eng)
    tables = set(insp.get_table_names())
    with eng.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if table not in tables:
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    added.append(f"{table}.{name}")
    return added
