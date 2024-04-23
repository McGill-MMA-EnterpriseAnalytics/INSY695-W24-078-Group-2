from fastapi import FastAPI, Request
import pandas as pd

data = {}
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

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
