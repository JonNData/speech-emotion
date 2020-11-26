from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Tell the app where the static files are
app.mount("/static", StaticFiles(directory="./speech-app/static"), name="static")
templates = Jinja2Templates(directory="./speech-app/templates")

# Import the main view