import re

def split_text(text, chunk_size=500, overlap=100):
    """
    Semantic-style chunking using sentences.
    """

    # split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # build chunk gradually
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())

            # overlap keeps context continuity
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks