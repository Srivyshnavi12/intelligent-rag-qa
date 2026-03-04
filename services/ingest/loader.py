import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx


def load_document(source):

    # website link
    if source.startswith("http"):
        return load_webpage(source)

    ext = os.path.splitext(source)[1].lower()

    if ext == ".txt":
        return load_txt(source)

    elif ext == ".pdf":
        return load_pdf(source)

    elif ext == ".docx":
        return load_docx(source)

    else:
        raise ValueError("Unsupported file type")


# -------- loaders --------

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def load_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ")