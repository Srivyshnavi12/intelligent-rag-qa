import faiss
import numpy as np

def create_faiss_index(embeddings):

    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  # inner product = cosine similarity
    index.add(embeddings)

    return index


def search_index(index, query_embedding, k=5):

    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return distances, indices