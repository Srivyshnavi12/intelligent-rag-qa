from services.reranker import rerank

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np

from services.ingest.loader import load_document
from services.chunking import split_text
from services.embeddings import create_embeddings
from services.generator import generate_answer

from services.vector_store import load_or_create_index, add_embeddings, search_index
from services.metadata import add_document, load_metadata
from services.chunk_store import add_chunks, load_chunks
from services.web_loader import load_website

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Upload Documents
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():

    files = request.files.getlist("documents")

    for file in files:

        if file.filename == "":
            continue

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        text = load_document(path)

        chunks = split_text(text)

        # store chunks
        add_chunks(file.filename, chunks)

        # embeddings
        embeddings = create_embeddings(chunks)

        dim = len(embeddings[0])

        index = load_or_create_index(dim)

        add_embeddings(index, embeddings)

        add_document(file.filename, len(chunks))

        print(f"{file.filename} stored successfully")

    return redirect(url_for("index"))


# -----------------------------
# Add Website
# -----------------------------
@app.route("/add_website", methods=["POST"])
def add_website():

    url = request.form.get("url")

    text = load_website(url)

    chunks = split_text(text)

    add_chunks(url, chunks)

    embeddings = create_embeddings(chunks)

    dim = len(embeddings[0])

    index = load_or_create_index(dim)

    add_embeddings(index, embeddings)

    add_document(url, len(chunks))

    return jsonify({"message": "Website added"})


# -----------------------------
# Ask Question
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():

    query = request.form.get("query")

    if not query:
        return jsonify({"answer": "Please enter a question."})

    query_embedding = np.array(create_embeddings([query], prefix="query")).astype("float32")

    index = load_or_create_index(len(query_embedding[0]))

    distances, results = search_index(index, query_embedding, k=8)

    chunk_db = load_chunks()

    retrieved_chunks = []
    seen = set()
    sources = set()

    for idx in results[0]:

        if idx != -1 and idx < len(chunk_db):

            doc = chunk_db[idx]["document"]
            text = chunk_db[idx]["text"]

            if text not in seen:

                seen.add(text)

                sources.add(doc)

                retrieved_chunks.append(f"Document: {doc}\n{text}")

    ranked_chunks = rerank(query, retrieved_chunks)

    context_chunks = ranked_chunks[:3]

    context = "\n\n".join(context_chunks)

    print("\n===== RETRIEVED CONTEXT =====\n")
    print(context[:500])

    answer = generate_answer(context, query)

    return render_template(
    "index.html",
    answer=answer,
    sources=list(sources)
)

# -----------------------------
# List Documents
# -----------------------------
@app.route("/documents")
def documents():
    return jsonify(load_metadata())


if __name__ == "__main__":
    app.run(debug=True)