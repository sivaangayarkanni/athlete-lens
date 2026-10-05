from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ml_engine import SPORTS

Sport = Literal["Athletics", "Kabaddi", "Kho-Kho", "Football", "Hockey", "Volleyball", "Badminton", "Wrestling"]
Sex = Literal["F", "M", "Other"]
assert list(Sport.__args__) == SPORTS  # keep schema + model in sync


def _strip(v):
    return v.strip() if isinstance(v, str) else v


class AthleteBase(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    sport: Sport
    role: str = Field(default="Athlete", max_length=60)
    sex: Sex = "F"
    age: int = Field(ge=10, le=60)
    height_cm: float = Field(ge=120, le=230)
    weight_kg: float = Field(ge=30, le=160)
    years_training: float = Field(ge=0, le=30, default=2)
    city: str = Field(default="Coimbatore", max_length=80)
    academy: str = Field(default="District Sports Academy", max_length=120)
    previous_injuries: int = Field(ge=0, le=20, default=0)
    notes: str = Field(default="", max_length=1000)

    _strip_strings = field_validator("name", "role", "city", "academy", "notes", mode="before")(_strip)


class AthleteCreate(AthleteBase):
    pass


class AthleteUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    sport: Sport | None = None
    role: str | None = Field(default=None, max_length=60)
    sex: Sex | None = None
    age: int | None = Field(default=None, ge=10, le=60)
    height_cm: float | None = Field(default=None, ge=120, le=230)
    weight_kg: float | None = Field(default=None, ge=30, le=160)
    years_training: float | None = Field(default=None, ge=0, le=30)
    city: str | None = Field(default=None, max_length=80)
    academy: str | None = Field(default=None, max_length=120)
    previous_injuries: int | None = Field(default=None, ge=0, le=20)
    notes: str | None = Field(default=None, max_length=1000)


class AthleteOut(AthleteBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    latest_readiness: float | None = None
    latest_performance: float | None = None
    latest_risk: str | None = None
    last_session: date | None = None
    session_count: int = 0


class SessionMetrics(BaseModel):
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


class SessionCreate(SessionMetrics):
    athlete_id: int
    session_date: date
    notes: str = Field(default="", max_length=1000)

    @field_validator("session_date")
    @classmethod
    def not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError("session_date cannot be in the future")
        return v


class SessionOut(SessionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

    @field_validator("session_date")
    @classmethod
    def not_in_future(cls, v: date) -> date:  # stored rows are never re-validated
        return v


class AnalyzePayload(BaseModel):
    """Guest / what-if analysis. If athlete_id is given, profile fields default to that athlete."""
    athlete_id: int | None = None
    age: int = Field(default=18, ge=10, le=60)
    sex: Sex = "F"
    sport: Sport = "Athletics"
    years_training: float = Field(default=3, ge=0, le=30)
    height_cm: float = Field(default=165, ge=120, le=230)
    weight_kg: float = Field(default=55, ge=30, le=160)
    previous_injuries: int = Field(default=0, ge=0, le=20)
    duration_min: float = Field(default=75, ge=10, le=300)
    distance_km: float = Field(default=6, ge=0, le=80)
    sprint_100m_s: float = Field(default=13.8, ge=9.5, le=25)
    vertical_jump_cm: float = Field(default=42, ge=15, le=90)
    resting_hr: float = Field(default=64, ge=38, le=110)
    session_hr_avg: float = Field(default=148, ge=80, le=210)
    rpe: float = Field(default=7, ge=1, le=10)
    sleep_hours: float = Field(default=7.2, ge=3, le=12)
    wellness: float = Field(default=7, ge=1, le=10)
    sessions_last_7: int = Field(default=5, ge=0, le=14)
    rest_days_last_7: int = Field(default=2, ge=0, le=7)


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: Literal["now", "this_week", "monitor"]
    tag: str


class PlanDay(BaseModel):
    day: int
    focus: str
    intensity: Literal["rest", "low", "moderate", "high"]
    minutes: int
    detail: str


class AnalysisResult(BaseModel):
    performance_index: float
    readiness_score: float
    readiness_band: Literal["green", "amber", "red"]
    injury_risk: Literal["low", "moderate", "high"]
    injury_probability: float
    overtraining: bool
    load_score: float
    recovery_score: float
    speed_score: float
    power_score: float
    explanations: list[str]
    recommendations: list[Recommendation]
    plan_72h: list[PlanDay]
    feature_importance: list[dict]


class SessionLogged(BaseModel):
    session: SessionOut
    analysis: AnalysisResult


class TrendPoint(BaseModel):
    date: date
    session_id: int
    srpe_load: float
    acute_7d: float
    chronic_28d: float
    acwr: float | None
    readiness_score: float | None
    performance_index: float | None
    injury_probability: float | None
    injury_risk: str | None
    sprint_100m_s: float
    vertical_jump_cm: float
    sleep_hours: float
    wellness: float


class AthleteDetail(BaseModel):
    athlete: AthleteOut
    sessions: list[SessionOut]
    trend: list[TrendPoint]
    latest: AnalysisResult | None


class DashboardStats(BaseModel):
    athlete_count: int
    session_count: int
    high_risk: int
    moderate_risk: int
    avg_readiness: float
    sports: list[dict]
    risk_bands: dict
    recent: list[dict]
    watchlist: list[dict]


class UploadResult(BaseModel):
    rows: int
    athletes_created: int
    sessions_created: int
    errors: list[dict]
