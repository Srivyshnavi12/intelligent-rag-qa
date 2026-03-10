from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts, prefix="passage"):

    formatted_texts = [f"{prefix}: {text}" for text in texts]

    embeddings = model.encode(formatted_texts)

    return embeddings