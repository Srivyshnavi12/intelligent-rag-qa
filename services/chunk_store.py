import json
import os

CHUNK_PATH = "database/chunks.json"


def load_chunks():

    if not os.path.exists(CHUNK_PATH):
        return []

    with open(CHUNK_PATH, "r") as f:
        return json.load(f)


def save_chunks(data):

    with open(CHUNK_PATH, "w") as f:
        json.dump(data, f, indent=4)


def add_chunks(document_name, chunks):

    data = load_chunks()

    start_id = len(data)

    for i, chunk in enumerate(chunks):

        data.append({
            "id": start_id + i,
            "document": document_name,
            "text": chunk
        })

    save_chunks(data)