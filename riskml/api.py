"""REST API for the trained model.

    uvicorn riskml.api:app --reload
    open http://localhost:8000/docs   (interactive Swagger UI)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FACULTIES

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "models" / "model.joblib"))

YesNo = Literal["yes", "no"]


class Student(BaseModel):
    age: int = Field(ge=15, le=70)
    faculty: Literal[tuple(FACULTIES)]  # type: ignore[valid-type]
    tuition: Literal["grant", "contract"]
    scholarship: YesNo
    works_part_time: YesNo
    dormitory: YesNo
    commute_minutes: float = Field(ge=0, le=300)
    admission_score: float = Field(ge=0, le=189)
    attendance_rate: float = Field(ge=0, le=1, description="share of classes attended, 0..1")
    study_hours_week: float = Field(ge=0, le=100)
    gpa_sem1: float = Field(ge=0, le=5)
    gpa_sem2: float = Field(ge=0, le=5)
    failed_courses: int = Field(ge=0, le=50)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 20, "faculty": "computer_science", "tuition": "contract",
                "scholarship": "no", "works_part_time": "yes", "dormitory": "no", "commute_minutes": 70,
                "admission_score": 128.5, "attendance_rate": 0.61, "study_hours_week": 4,
                "gpa_sem1": 3.1, "gpa_sem2": 2.7, "failed_courses": 3,
            }
        }
    }


class Prediction(BaseModel):
    dropout_probability: float
    at_risk: bool
    risk_level: Literal["low", "medium", "high"]
    recommendations: list[str]


class BatchRequest(BaseModel):
    students: list[Student] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(503, "model is not trained yet: run `python -m riskml.train`")
    return joblib.load(MODEL_PATH)


def recommendations(s: Student) -> list[str]:
    """Simple, explainable advice for an academic advisor (rule-based on top of the model)."""
    tips = []
    if s.attendance_rate < 0.75:
        tips.append("Attendance is below 75%: contact the student and the group curator.")
    if s.failed_courses >= 2:
        tips.append("Several failed courses: plan retakes and offer tutoring.")
    if s.gpa_sem2 < s.gpa_sem1 - 0.4:
        tips.append("GPA dropped sharply in the last semester: schedule a meeting with an advisor.")
    if s.works_part_time == "yes" and s.study_hours_week < 6:
        tips.append("Works part-time and studies little: discuss a flexible schedule.")
    if s.tuition == "contract" and s.gpa_sem2 < 3.2:
        tips.append("Contract student with low grades: inform about payment plans and grant options.")
    return tips or ["No specific risk factors detected."]


def predict_frame(students: list[Student]) -> list[Prediction]:
    bundle = model_bundle()
    df = pd.DataFrame([s.model_dump() for s in students])[bundle["features"]]
    probs = bundle["pipeline"].predict_proba(df)[:, 1]
    thr = bundle["threshold"]
    out = []
    for s, p in zip(students, probs):
        level = "high" if p >= max(thr, 0.6) else "medium" if p >= thr * 0.6 else "low"
        out.append(Prediction(dropout_probability=round(float(p), 4), at_risk=bool(p >= thr),
                              risk_level=level, recommendations=recommendations(s)))
    return out


app = FastAPI(
    title="Student Dropout Risk API",
    version="1.0.0",
    description="Predicts the probability that a student drops out, so advisors can help early.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict:
    b = model_bundle()
    return {"model": b["model_name"], "threshold": round(b["threshold"], 4), "test_metrics": b["metrics"]}


@app.post("/predict", response_model=Prediction)
def predict(student: Student) -> Prediction:
    return predict_frame([student])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest) -> list[Prediction]:
    return predict_frame(req.students)
