from fastapi import Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from tutorial import app, templates
import time
from datetime import datetime

# This script will handle delaying simultaneous pings, log users

# no @ and _ for private functions
def _run_task(name: str, id=None):
    time.sleep(3)
    with open('tasks_out.txt', mode='a') as file: # 'a' mode lets us add to it
        now = datetime.now()
        content = f"{name} [{id}]: {now}\n"
        file.write(content)

@app.post("/task/run/{name}/{id}")
async def task_run(name: str, id: int, background_tasks: BackgroundTasks):
    """
    Takes input task and writes it into a file
    """
    background_tasks.add_task(_run_task, name, id)
    return {"message": f"Task {name} ID {ID} is being run\n"}