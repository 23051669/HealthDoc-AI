import os

from langchain_groq import ChatGroq


def get_llm():
    """
    Create the Groq language model.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY was not found."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.1-8b-instant"
    )

    llm = ChatGroq(
        model=model_name,
        temperature=0.1,
        api_key=api_key
    )

    return llm
