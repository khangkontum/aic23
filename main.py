from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from plato import predict, get_key_frame
import pickle

app = FastAPI()

origins = [
    "http://localhost:3000"
]

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
    app.model = CustomUnpickler(
        open("./plato.pkl", "rb")
    ).load()


class Query(BaseModel):
    text: str
    method: str = "dot"
    top: int = 200

class KeyFrame(BaseModel):
    video: str
    keyframe: str


@app.post("/query")
async def query(query: Query):
    result = predict(
        app.model, query.text, query.method, query.top
    )

    for index in range(len(result)):
        del result[index]["clip_embedding"]
        del result[index]["filepath"]

    # response = jsonify({"data": result.tolist()})
    # response.status_code = 200
    return {"data": result.tolist()}


@app.post("/keyframe")
async def keyframe(keyframe: KeyFrame):
    result = get_key_frame(app.model, keyframe.video, keyframe.keyframe)

    return {"mappedKeyFrame": result[0], "youtubeLink": result[1] }