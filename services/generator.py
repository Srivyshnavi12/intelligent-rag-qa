from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer once
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def generate_answer(context, question):

    # limit context size (VERY IMPORTANT)
    context = context[:2000]

    prompt = f"""
Answer the question using ONLY the information in the context.

If the context contains the answer, explain clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,     # makes answers less random
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer