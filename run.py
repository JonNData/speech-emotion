import uvicorn
from config import PORT

if __name__ == "__main__":
    uvicorn.run("tutorial:app", host='0.0.0.0', port=int(PORT), reload=True, debug=True, workers=1)
