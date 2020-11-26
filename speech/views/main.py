from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from speech import templates
from joblib import load

app = FastAPI()

# Tell the app where the static files are
app.mount("/static", StaticFiles(directory=".\\speech\\static"), name="static")
templates = Jinja2Templates(directory=".\\speech\\templates")


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