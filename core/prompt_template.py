from langchain_core.prompts import PromptTemplate


def template_1():
    rag_prompt = PromptTemplate.from_template(
        template="""You are a helpful assistant.
    Use only the provided context to answer the question.
    If the answer is not present in the context, say: "I don't know based on the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    )

    return rag_prompt


def template_2():
    prompt_template = """
    According to the contents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}
    Answer the given question: {{query}}
    Answer:
    """
    return prompt_template
