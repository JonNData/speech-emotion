from tutorial import app, templates
from fastapi import Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse

import numpy as np
from joblib import load
from functions import extract_audio_features
import soundfile
import librosa
import librosa.display

import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from config import ALLOWED_EXTENSIONS

model = load('.\\tutorial\\mlpipeline.joblib')


@app.get("/")
# Specifying a the var type here is akin to pydantic code
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile=File(...)):

    if file.filename.split('.')[1] in ALLOWED_EXTENSIONS:
        audio_file = await file.read()
        file.file.close()

        # Save it to work with
        with open('tutorial/static/user_assets/current.wav', mode='wb') as f:
            f.write(audio_file)
        
        # run the prediction on saved file
        audio_sample = extract_audio_features('tutorial/static/user_assets/current.wav', mfcc=True, chroma=True, mel=True)
        audio_ready = np.array(audio_sample).reshape(1,-1)
        prediction = model.predict(audio_ready)
        
        # save prediction
        with open('tutorial/static/user_assets/prediction.txt', "wb") as f:
            f.write(prediction[0])

        # Graph it
    #def vis_spec(audio_file):
        y, sr = librosa.load(path='tutorial/static/user_assets/current.wav', sr=None)
        fig = Figure()
        plt.figure(figsize=(12, 8))

        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        plt.subplot(4, 2, 1)

        librosa.display.specshow(D, y_axis="linear")
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.xlabel('Time')
        
        plt.savefig('tutorial/static/user_assets/spectrogram.png')

    #def vis_harm(audio_file):
        y, sr = librosa.load(path='tutorial/static/user_assets/current.wav', sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)
        fig = Figure()
        plt.subplot(3, 1, 3)
        librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
        librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
        plt.title('Harmonic + Percussive')
        plt.tight_layout()

        plt.savefig('tutorial/static/user_assets/harmonic.png')

        return {
                 "prediction": str(prediction[0])
                }

@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    # read in prediction
    with open('tutorial/static/user_assets/prediction.txt', "rb") as f:
        saved_pred = f.read()
    return templates.TemplateResponse("results.html",
                 {  "request": request,
                 "prediction": saved_pred.decode("utf-8")
                })
        