import json
import os

META_PATH = "database/metadata.json"


def load_metadata():

    if not os.path.exists(META_PATH):
        return {"documents": []}

    with open(META_PATH, "r") as f:
        return json.load(f)


def save_metadata(data):

    with open(META_PATH, "w") as f:
        json.dump(data, f, indent=4)


def add_document(name, chunks):

    data = load_metadata()

    data["documents"].append({
        "name": name,
        "chunks": chunks
    })

    save_metadata(data)


def remove_document(name):

    data = load_metadata()

    data["documents"] = [
        d for d in data["documents"] if d["name"] != name
    ]

    save_metadata(data)