from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from asonic import Client
from asonic.enums import Channel

import os
import json
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
async def preload_model():
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

    c = Client(
        host="127.0.0.1",
        port=1491,
        password="admin",
        max_connections=100,
    )
    await c.channel(Channel.SEARCH)
    app.client = c

    files = []
    asr_folder = "./data_processing/raw/transcript/"
    app.text_data = {}

    for file in os.listdir(asr_folder):
        if file.endswith(".json"):
            file_path = os.path.join(asr_folder, file)
            vid_id = file.split(".")[0]
            with open(
                file_path, "r", encoding="utf-8"
            ) as f:
                data = json.load(f)["segments"]
                app.text_data[vid_id] = {}

                for segment in data:
                    start = int(
                        float(segment["start"]) * 25
                    )
                    app.text_data[vid_id][start] = segment[
                        "text"
                    ]


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


@app.post("/asr")
async def asrquery(asrquery: ASRQuery):
    results = await app.client.query(
        collection="asr",
        bucket="test",
        terms=asrquery.text,
        limit=asrquery.top,
    )

    res = []
    # print(results)
    for r in results:
        vid_id, frame_start, frame_end = r.decode("utf-8").split(
            "-"
        )

        frame_start = int(float(frame_start) * 25)
        frame_end = int(float(frame_end) * 25)
        frame_mid = ((frame_start + frame_end) >> 1)
        frame_mid -= frame_mid % 25

        res.append(
            {
                "text": app.text_data[vid_id][frame_start],
                "frame_id": frame_mid,
                "video_id": vid_id,
            },
        )

    return res


class Respond(BaseModel):
    text: str


@app.post("/print_log")
async def printLog(respond: Respond):
    utils.print_log(respond.text, "log.txt")
