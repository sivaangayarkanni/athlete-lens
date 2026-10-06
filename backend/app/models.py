from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Athlete(Base):
    __tablename__ = "athletes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    sport: Mapped[str] = mapped_column(String(60), index=True)
    role: Mapped[str] = mapped_column(String(60), default="Athlete")
    sex: Mapped[str] = mapped_column(String(16), default="F")
    age: Mapped[int] = mapped_column(Integer)
    height_cm: Mapped[float] = mapped_column(Float)
    weight_kg: Mapped[float] = mapped_column(Float)
    years_training: Mapped[float] = mapped_column(Float, default=2.0)
    city: Mapped[str] = mapped_column(String(80), default="Coimbatore")
    academy: Mapped[str] = mapped_column(String(120), default="District Sports Academy")
    previous_injuries: Mapped[int] = mapped_column(Integer, default=0)
    injury_history: Mapped[str] = mapped_column(Text, default="")          # comma-separated body regions
    growth_cm: Mapped[float | None] = mapped_column(Float, nullable=True)   # height gained in the last 6 months
    nordic_program: Mapped[int] = mapped_column(Integer, default=0)
    adductor_program: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    injuries: Mapped[list["InjuryRecord"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan", order_by="InjuryRecord.onset_date"
    )
    sessions: Mapped[list["TrainingSession"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan", order_by="TrainingSession.session_date"
    )
    predictions: Mapped[list["PredictionLog"]] = relationship(cascade="all, delete-orphan")


class TrainingSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    session_date: Mapped[date] = mapped_column(Date, index=True)
    duration_min: Mapped[float] = mapped_column(Float)
    distance_km: Mapped[float] = mapped_column(Float, default=0)
    sprint_100m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    vertical_jump_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    resting_hr: Mapped[float] = mapped_column(Float, default=68)
    session_hr_avg: Mapped[float] = mapped_column(Float, default=142)
    rpe: Mapped[float] = mapped_column(Float, default=6)
    sleep_hours: Mapped[float] = mapped_column(Float, default=7)
    wellness: Mapped[float] = mapped_column(Float, default=7)
    sessions_last_7: Mapped[int] = mapped_column(Integer, default=4)
    rest_days_last_7: Mapped[int] = mapped_column(Integer, default=2)
    session_type: Mapped[str] = mapped_column(String(16), default="training")   # training | match | recovery
    asymmetry_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # single-leg hop/jump asymmetry
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    athlete: Mapped[Athlete] = relationship(back_populates="sessions")
    prediction: Mapped["PredictionLog | None"] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    performance_index: Mapped[float] = mapped_column(Float)
    readiness_score: Mapped[float] = mapped_column(Float)
    injury_risk: Mapped[str] = mapped_column(String(24))
    injury_probability: Mapped[float] = mapped_column(Float)
    overtraining: Mapped[int] = mapped_column(Integer, default=0)
    region_risks: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON {region: probability}
    top_region: Mapped[str | None] = mapped_column(String(24), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    session: Mapped[TrainingSession | None] = relationship(back_populates="prediction")


class InjuryRecord(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id", ondelete="CASCADE"), index=True)
    region: Mapped[str] = mapped_column(String(24))
    injury_type: Mapped[str] = mapped_column(String(24))
    onset_date: Mapped[date] = mapped_column(Date, index=True)
    return_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    athlete: Mapped[Athlete] = relationship(back_populates="injuries")
