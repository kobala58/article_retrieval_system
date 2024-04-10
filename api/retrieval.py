from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI

import emb_utils


class QueryPayload(BaseModel):
    query: str
    articles_count: int = 5


model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    tmp = emb_utils.load_embeddings_and_model()
    print(tmp.keys())
    model["embeddings"] = tmp["embeddings"]
    model["text_chunks"] = tmp["text_chunks"]
    model["transformer"] = tmp["transformer"]

    yield
    # Clean up the ML models and release the resources
    model.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/query/")
def read_root(query_req: QueryPayload):
    req_dict = query_req.dict()
    data = emb_utils.top_results_and_scores(
        query=req_dict["query"],
        embeddings=model["embeddings"],
        pages_and_chunks=model["text_chunks"],
        model=model["transformer"],
        n_resources_to_return=req_dict["articles_count"]
    )
    return data
