"""Module containing code to run a FastAPI Server."""
from fastapi import FastAPI, Request, HTTPException
import pandas as pd

data: dict = {}
ready_status: dict = {}
app = FastAPI()


@app.get("/")
def read_root():
    return {"This is Root": "True"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in data:
        return {"Error": "Item not found"}
    return data[item_id]

@app.put("/items/{item_id}")
async def create_item(item_id: int, request: Request):
    global data
    json_data = await request.json()
    data[item_id] = json_data
    ready_status[item_id] = False # not ready right after creation
    return {"Response Saved Successfully": True}

@app.get("/items")
def get_items():
    return data

@app.get("/itemsdf/{item_id}")
def read_item_df(item_id: int):
    if item_id not in data:
        return {"Error": "Item not found"}
    dataframe = pd.read_json(data[item_id])
    return dataframe

# VERY IMPORTANT

# Code to implement a polling mechanism between
# containers, so as to prevent dependencies from
# firing before their required data pieces are ready

@app.put("/set_ready/{item_id}")
async def set_ready(item_id: int):

    ready_status[item_id] = True
    print(f"Set readiness for item_id={item_id}, current status: {ready_status}")
    return {"status": "Item is now ready"}

@app.get("/ready/{item_id}")
def check_ready(item_id: int):
    print(f"Checking readiness for item_id={item_id}, current status: {ready_status}")
    if item_id in ready_status and ready_status[item_id]:
        return {"status": "ready"}
    raise HTTPException(status_code=404, detail="Not ready")
