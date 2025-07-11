def build_prompt(question: str, context_chunks):
    """
    Compose prompt with question and retrieved chunks for LLM.
    """
    context_text = "\n\n".join([chunk for _, chunk in context_chunks])
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return prompt
