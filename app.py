from services.generator import generate_answer

from services.vector_store import search_index
import numpy as np

from services.chunking import split_text
from services.embeddings import create_embeddings
from services.vector_store import create_faiss_index

from flask import Flask, render_template, request
import os
from pypdf import PdfReader

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["document"]

    if file.filename == "":
        return "No file selected"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text from PDF
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    # Split text into chunks
    chunks = split_text(text)
    print(f"Total Chunks Created: {len(chunks)}")

    # Create embeddings
    embeddings = create_embeddings(chunks)
    print("Embeddings created.")

    # Store in FAISS
    index = create_faiss_index(embeddings)
    print("FAISS index created.")

    # ---- TEST QUERY ----
    query = "What is this document about?"

    query_embedding = create_embeddings([query])
    query_embedding = np.array(query_embedding)

    results = search_index(index, query_embedding, k=3)

    # Combine retrieved chunks into context
    retrieved_text = ""
    for idx in results[0]:
        retrieved_text += chunks[idx] + "\n"

    # Generate final answer
    answer = generate_answer(retrieved_text, query)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)

    return answer


if __name__ == "__main__":
    app.run(debug=True)