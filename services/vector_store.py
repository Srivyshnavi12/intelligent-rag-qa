import faiss
import numpy as np
import os

INDEX_PATH = "database/vector_index.faiss"

def load_or_create_index(embedding_dim):

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(embedding_dim)

    return index


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


def add_embeddings(index, embeddings):

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    save_index(index)

    return index


def search_index(index, query_embedding, k=5):

    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return distances, indices