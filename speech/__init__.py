from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Tell the app where the static files are
app.mount("/static", StaticFiles(directory="./speech/static"), name="static")
templates = Jinja2Templates(directory="./speech/templates")

# Import the main view
from speech.views import main