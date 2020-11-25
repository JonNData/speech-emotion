from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "I feel the need for FastAPI"}