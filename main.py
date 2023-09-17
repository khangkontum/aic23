from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pickle
import utils

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
    app.model = CustomUnpickler(
        open("./plato.pkl", "rb")
    ).load()


class Query(BaseModel):
    text: str
    top: int = 100
    with_weith: bool = True
    text_gamma: float = (1.0,)
    skip: int = (25,)
    gamma: float = (0.9,)
    decay: float = (0.3,)
    window_size: int = (3,)


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
    result = app.model.predict(query.text, query.top)

    for index in range(len(result)):
        del result[index]["clip_embedding"]
        del result[index]["filepath"]

    # response = jsonify({"data": result.tolist()})
    # response.status_code = 200
    return {"data": result.tolist()}
