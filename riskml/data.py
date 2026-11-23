"""Dataset generation and loading.

Real student records are personal data, so this project ships a **synthetic** dataset
generator. The generating process is written by hand from common findings in
educational research (attendance, prior grades and failed courses are the strongest
predictors of dropout), with noise and interactions so the problem is not trivial.

Any real CSV with the same columns can be used instead: `python -m riskml.train --data my.csv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = [
    "age",
    "admission_score",       # 0..189 (DTM-style entrance exam score)
    "attendance_rate",       # 0..1
    "gpa_sem1",              # 0..5
    "gpa_sem2",              # 0..5
    "failed_courses",
    "study_hours_week",
    "commute_minutes",
]
CATEGORICAL = ["tuition", "scholarship", "works_part_time", "faculty", "dormitory"]
TARGET = "dropout"

FACULTIES = ["computer_science", "economics", "engineering", "medicine", "philology"]


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["male", "female"], n)
    age = np.clip(rng.normal(19.5, 1.6, n).round(), 17, 30).astype(int)
    faculty = rng.choice(FACULTIES, n, p=[0.25, 0.2, 0.25, 0.15, 0.15])
    tuition = rng.choice(["grant", "contract"], n, p=[0.3, 0.7])
    admission = np.where(tuition == "grant", rng.normal(165, 12, n), rng.normal(135, 18, n)).clip(56, 189).round(1)
    works = rng.choice(["yes", "no"], n, p=[0.35, 0.65])
    dorm = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    commute = np.where(dorm == "yes", rng.normal(10, 5, n), rng.gamma(3, 15, n)).clip(2, 150).round()

    # latent "engagement" drives attendance, study time and grades
    engagement = (
        rng.normal(0, 1, n)
        + 0.025 * (admission - 140)
        - 0.6 * (works == "yes")
        - 0.006 * commute
    )
    attendance = (0.82 + 0.08 * engagement + rng.normal(0, 0.07, n)).clip(0.2, 1).round(3)
    study = (10 + 3.5 * engagement + rng.normal(0, 3, n)).clip(0, 40).round(1)
    gpa1 = (3.6 + 0.45 * engagement + rng.normal(0, 0.45, n)).clip(1.5, 5).round(2)
    gpa2 = (gpa1 + 0.25 * (attendance - 0.8) * 4 + rng.normal(0, 0.35, n)).clip(1.0, 5).round(2)
    failed = rng.poisson(np.exp(0.8 - 1.1 * (gpa2 - 3.2)).clip(0.01, 6))
    scholarship = np.where((gpa1 > 4.3) & (rng.random(n) < 0.7), "yes", "no")

    logit = (
        -4.4
        - 3.0 * (attendance - 0.8) * 5 / 2
        - 1.1 * (gpa2 - 3.5)
        + 0.55 * failed
        + 0.5 * (works == "yes")
        + 0.4 * ((tuition == "contract") & (gpa2 < 3.2))   # paying students struggling academically
        - 0.6 * (scholarship == "yes")
        + 0.004 * commute
        + 0.12 * (age - 19.5)
        + rng.normal(0, 0.6, n)
    )
    dropout = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "student_id": [f"S{100000 + i}" for i in range(n)],
        "gender": gender,
        "age": age,
        "faculty": faculty,
        "tuition": tuition,
        "scholarship": scholarship,
        "works_part_time": works,
        "dormitory": dorm,
        "commute_minutes": commute,
        "admission_score": admission,
        "attendance_rate": attendance,
        "study_hours_week": study,
        "gpa_sem1": gpa1,
        "gpa_sem2": gpa2,
        "failed_courses": failed,
        TARGET: dropout,
    })


def load(path: str | None = None, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path) if path else generate(n, seed)
    missing = set(NUMERIC + CATEGORICAL + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/students.csv", index=False)
    print(out.head())
    print(f"dropout rate: {out[TARGET].mean():.1%}")
