from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


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
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sessions: Mapped[list["TrainingSession"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )


class TrainingSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    session_date: Mapped[date] = mapped_column(Date, index=True)
    duration_min: Mapped[float] = mapped_column(Float)
    distance_km: Mapped[float] = mapped_column(Float, default=0)
    sprint_100m_s: Mapped[float] = mapped_column(Float, default=14.5)
    vertical_jump_cm: Mapped[float] = mapped_column(Float, default=38)
    resting_hr: Mapped[float] = mapped_column(Float, default=68)
    session_hr_avg: Mapped[float] = mapped_column(Float, default=142)
    rpe: Mapped[float] = mapped_column(Float, default=6)
    sleep_hours: Mapped[float] = mapped_column(Float, default=7)
    wellness: Mapped[float] = mapped_column(Float, default=7)
    sessions_last_7: Mapped[int] = mapped_column(Integer, default=4)
    rest_days_last_7: Mapped[int] = mapped_column(Integer, default=2)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    athlete: Mapped[Athlete] = relationship(back_populates="sessions")


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    performance_index: Mapped[float] = mapped_column(Float)
    readiness_score: Mapped[float] = mapped_column(Float)
    injury_risk: Mapped[str] = mapped_column(String(24))
    injury_probability: Mapped[float] = mapped_column(Float)
    overtraining: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
