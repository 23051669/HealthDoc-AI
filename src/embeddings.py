from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Load the embedding model used to convert
    document text into vectors.
    """

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return model
