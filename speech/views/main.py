from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from speech import templates
from joblib import load
from speech.functions import *

model = load('speech\\mlpipeline.joblib')
app = FastAPI()

# Tell the app where the static files are
app.mount("/static", StaticFiles(directory=".\\speech\\static"), name="static")
templates = Jinja2Templates(directory=".\\speech\\templates")


@app.get("/")
# Specifying a the var type here is akin to pydantic code
async def root(request: Request):
    return {"message": "I feel the need for FastAPI"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile=File(...)):
    audio_sample = extract_audio_features(file, mfcc=True, chroma=True, mel=True)
    audio_ready = np.array(audio_sample).reshape(1,-1)
    prediction = model.predict(audio_ready)
    return {"prediction": str(prediction)}

@app.post("/graph")
async def graph():
    return {"message": "I feel the need for FastAPI"}