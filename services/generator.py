from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def generate_answer(context, question):

    prompt = f"""
You are an assistant answering questions about uploaded documents.

Use ONLY the information in the context below.

If the answer is not in the context, say:
"I could not find the answer in the uploaded documents."

If the user asks what the document is about, summarize it briefly.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=False
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)