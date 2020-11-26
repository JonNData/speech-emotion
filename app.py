from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "I feel the need for FastAPI"}

@app.put("/predict")
async def predict():
    return {"message": "I feel the need for FastAPI"}

@app.put("/graph")
async def graph():
    return {"message": "I feel the need for FastAPI"}