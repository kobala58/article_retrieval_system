import torch
from sentence_transformers import util, SentenceTransformer
import os
import pandas as pd
import numpy as np


def load_embeddings(device) -> [torch.tensor, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # detect if there is GPU installed
    filename = "./data/data_embedded_output.csv"
    embedding_df = pd.read_csv(filename)
    embedding_df["embedding"] = (embedding_df["embedding"]
                                 .apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")))  # convert it back to np.array
    embedding_dict = embedding_df.to_dict(orient='records')
    only_embeddings = torch.tensor(np.array(embedding_df["embedding"].tolist()),
                                   dtype=torch.float32).to(device)
    return [only_embeddings, embedding_dict]


def load_transformer(device) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)


def load_embeddings_and_model() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # detect if there is GPU installed
    emb, embedding_dict = load_embeddings(device)
    transformer = load_transformer(device)
    return {"embeddings": emb, "transformer": transformer, "text_chunks": embedding_dict}


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int = 5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query

    query_embedding = model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]



    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices


def top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 model: SentenceTransformer,
                                 n_resources_to_return: int = 5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.
    """

    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  model=model,
                                                  n_resources_to_return=n_resources_to_return)

    # Loop through zipped together scores and indicies and create payload to return
    reponse = []
    for score, index in zip(scores, indices):
        tmp = {
            "score": f'{score:.4f}',
            "atricle_title": pages_and_chunks[index]["Title"],
            "text": pages_and_chunks[index]["sentence_chunk"]
        }
        reponse.append(tmp)

    return reponse
