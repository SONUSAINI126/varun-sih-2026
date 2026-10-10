"""Run: uvicorn varun.api:app --reload"""
import numpy as np
from fastapi import FastAPI
from .pipeline import run

app = FastAPI(title="VARUN API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/hazard/demo")
def demo(threshold_mm: float = 100.0):
    r = run(np.random.default_rng(1).gamma(2, 25, (20, 30)), threshold_mm)
    return {"members_kept": r["members_kept"], "peak_prob": r["peak_prob"]}
