from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer once
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def generate_answer(context, question):
    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer