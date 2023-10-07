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
    print(
        "model path:",
        os.path.join(model_path, os.getenv("MODEL_16")),
        "rb",
    )
    app.model["b16"] = CustomUnpickler(
        open(
            os.path.join(model_path, os.getenv("MODEL_16")),
            "rb",
        )
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
    keyframe: int


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

    return {"data": result}


@app.post("/keyframe")
async def keyframe(keyframe: KeyFrame):
    result = utils.get_key_frame(
        app.model["b16"], keyframe.video, keyframe.keyframe
    )

    return {
        "mappedKeyFrame": result[0],
        "youtubeLink": result[1],
    }


class ASRQuery(BaseModel):
    text: str
    top: int = 40

@app.post("/asr_fuzzy")
async def asrquery(asrquery: ASRQuery):
    results = utils.get_result_fuzzy_search(query=asrquery.text, top=asrquery.top)

    return [
        {
            "mappedText": result[1],
            "similarity": result[2],
            "video": result[0],
            "keyframeStart": int(float(result[3]) * 25),
            "youtubeLink": utils.get_utube_link(app.model["b16"], result[0], result[3]),
        }
        for result in results
    ]

@app.post("/asr_fulltext")
async def asrquery(asrquery: ASRQuery):
    results = utils.fulltext_search(query=asrquery.text, top=asrquery.top)

    return [
        {
            "mappedText": result[1],
            "video": result[0],
            "keyframeStart": int(float(result[2]) * 25),
            "youtubeLink": utils.get_utube_link(app.model["b16"], result[0], result[2]),
        }
        for result in results
    ]

class Respond(BaseModel):
    text: str

@app.post("/print_log")
async def printLog(respond: Respond):
    utils.print_log(respond.text, "log.txt")