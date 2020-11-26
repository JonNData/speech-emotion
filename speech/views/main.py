from speech import app, templates
from fastapi import Request
import numpy as np


@app.get("/")
# Specifying a the var type here is akin to pydantic code
async def root(request: Request):
    return {"message": "I feel the need for FastAPI"}

@app.put("/predict")
async def predict():
    return {"message": "I feel the need for FastAPI"}

@app.put("/graph")
async def graph():
    return {"message": "I feel the need for FastAPI"}