from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class AthleteCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    sport: str
    role: str = "Athlete"
    sex: Literal["F", "M", "Other"] = "F"
    age: int = Field(ge=10, le=60)
    height_cm: float = Field(ge=120, le=230)
    weight_kg: float = Field(ge=30, le=160)
    years_training: float = Field(ge=0, le=30, default=2)
    city: str = "Coimbatore"
    academy: str = "District Sports Academy"
    previous_injuries: int = Field(ge=0, le=20, default=0)
    notes: str = ""


class AthleteOut(AthleteCreate):
    id: int
    created_at: datetime
    latest_readiness: float | None = None
    latest_risk: str | None = None
    session_count: int = 0

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    athlete_id: int
    session_date: date
    duration_min: float = Field(ge=10, le=300)
    distance_km: float = Field(ge=0, le=80, default=0)
    sprint_100m_s: float = Field(ge=9.5, le=25, default=14.5)
    vertical_jump_cm: float = Field(ge=15, le=90, default=38)
    resting_hr: float = Field(ge=38, le=110, default=68)
    session_hr_avg: float = Field(ge=80, le=210, default=142)
    rpe: float = Field(ge=1, le=10, default=6)
    sleep_hours: float = Field(ge=3, le=12, default=7)
    wellness: float = Field(ge=1, le=10, default=7)
    sessions_last_7: int = Field(ge=0, le=14, default=4)
    rest_days_last_7: int = Field(ge=0, le=7, default=2)
    notes: str = ""


class SessionOut(SessionCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalyzePayload(BaseModel):
    athlete_id: int | None = None
    age: int = 18
    sex: Literal["F", "M", "Other"] = "F"
    sport: str = "Athletics"
    years_training: float = 3
    height_cm: float = 165
    weight_kg: float = 55
    previous_injuries: int = 0
    duration_min: float = 75
    distance_km: float = 6
    sprint_100m_s: float = 13.8
    vertical_jump_cm: float = 42
    resting_hr: float = 64
    session_hr_avg: float = 148
    rpe: float = 7
    sleep_hours: float = 7.2
    wellness: float = 7
    sessions_last_7: int = 5
    rest_days_last_7: int = 2


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: Literal["now", "this_week", "monitor"]
    tag: str


class AnalysisResult(BaseModel):
    performance_index: float
    readiness_score: float
    injury_risk: Literal["low", "moderate", "high"]
    injury_probability: float
    overtraining: bool
    load_score: float
    recovery_score: float
    speed_score: float
    power_score: float
    explanations: list[str]
    recommendations: list[Recommendation]
    feature_importance: list[dict]


class DashboardStats(BaseModel):
    athlete_count: int
    session_count: int
    high_risk: int
    avg_readiness: float
    sports: list[dict]
    recent: list[dict]
