from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.plato import predict
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
            from app.plato import Plato

            return Plato
        return super().find_class(module, name)


@app.on_event("startup")
def preload_model():
    app.model = CustomUnpickler(
        open("../plato.pkl", "rb")
    ).load()


class Query(BaseModel):
    text: str
    method: str = "dot"
    top: int = 200


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
