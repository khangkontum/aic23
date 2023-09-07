from fastapi import FastAPI
from pydantic import BaseModel

from plato import predict
import pickle

app = FastAPI()


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Plato":
            from plato import Plato

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
