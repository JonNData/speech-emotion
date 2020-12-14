from tutorial import app, templates
from fastapi import Request, File, UploadFile
from fastapi.responses import HTMLResponse

import numpy as np
from joblib import load
from functions import extract_audio_features
import soundfile
import librosa
import librosa.display

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from config import ALLOWED_EXTENSIONS

model = load('.\\tutorial\\mlpipeline.joblib')


@app.get("/")
# Specifying a the var type here is akin to pydantic code
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile=File(...)):

    if file.filename.split('.')[1] in ALLOWED_EXTENSIONS:
        audio_file = await file.read()
        file.file.close()

        # Save it to work with
        with open('current.wav', mode='wb') as f:
            f.write(audio_file)
        
        # run the prediction on saved file
        audio_sample = extract_audio_features('current.wav', mfcc=True, chroma=True, mel=True)
        audio_ready = np.array(audio_sample).reshape(1,-1)
        prediction = model.predict(audio_ready)
        return {"prediction": str(prediction[0])}


@app.post("/graph", response_class=HTMLResponse)
async def graph():
    ## Move this to a different module later
    def vis_spec(audio_file):
        y, sr = librosa.load(path=audio_file, sr=None)
        plt.figure(figsize=(12, 8))

        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        plt.subplot(4, 2, 1)

        librosa.display.specshow(D, y_axis="linear")
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.xlabel('Time')
        plt.show

    def vis_harm(audio_file):
        y, sr = librosa.load(path=audio_file, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)
        plt.subplot(3, 1, 3)
        librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
        librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
        plt.title('Harmonic + Percussive')
        plt.tight_layout()

    return {"spectrogram": vis_spec(file),
            "Harmonics": vis_harm(file)}