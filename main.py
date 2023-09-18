from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import pickle
import utils

load_dotenv(".env")

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Plato":
            from plato import Plato

            return Plato
        return super().find_class(module, name)



@app.on_event("startup")
def preload_model():
    model_path = os.getenv("MODEL_PATH")

    app.model = {}
    app.model["b16"] = CustomUnpickler(
        open(os.path.join(model_path, os.getenv("MODEL_16")), "rb")
    ).load()

    app.model["b32"] = CustomUnpickler(
        open(os.path.join(model_path, os.getenv("MODEL_32")), "rb")
    ).load()



class Query(BaseModel):
    text: str
    top: int = 100
    model: str = "b16"
    with_weith: bool = True
    text_gamma: float = (1.0,)
    skip: int = (25,)
    gamma: float = (0.9,)
    decay: float = (0.3,)
    window_size: int = (3,)

class KeyFrame(BaseModel):
    video: str
    keyframe: str

@app.post("/weighted-query")
async def query(query: Query):
    queries = None
    queries = utils.extract_query(query.text)
    result = app.model.predicts(
        queries,
        top=query.top,
        text_gamma=1.0,
        skip=25,
        gamma=0.9,
        decay=0.3,
        window_size=3,
    )

    for index in range(len(result)):
        del result[index]["clip_embedding"]
        del result[index]["filepath"]

    return {"data": result.tolist()}


@app.post("/query")
async def query(query: Query):
    model = app.model[query.model]
    result = model.predict(query.text, query.top)

    for index in range(len(result)):
        del result[index]["clip_embedding"]
        del result[index]["filepath"]

    return {"data": result.tolist()}

@app.post("/keyframe")
async def keyframe(keyframe: KeyFrame):
    result = utils.get_key_frame(app.model, keyframe.video, keyframe.keyframe)

    return {"mappedKeyFrame": result[0], "youtubeLink": result[1] }
