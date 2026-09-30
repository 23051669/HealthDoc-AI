from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.embeddings import get_embeddings

# Name of our Chroma collection
COLLECTION_NAME = "healthdoc_documents"


def create_vector_store(chunks):
    """
    Store document chunks in ChromaDB.
    """

    documents = []

    for number, chunk in enumerate(chunks):

        document = Document(
            page_content=chunk,
            metadata={
                "section": number + 1
            }
        )

        documents.append(document)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vector_store


def ask_question(
    llm,
    vector_store,
    question,
    number_of_results=4
):
    """
    Search the uploaded document and use the
    relevant sections to answer the question.
    """

    # Find the most relevant sections
    results = vector_store.similarity_search(
        question,
        k=number_of_results
    )

    if not results:

        return (
            "I could not find relevant information "
            "in the uploaded document.",
            []
        )

    # Combine retrieved sections
    context = ""

    for number, document in enumerate(
        results,
        start=1
    ):

        context += (
            f"\n\n--- Document Section {number} ---\n"
        )

        context += document.page_content

    # Prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are HealthDoc AI.

Answer the user's question using the
document sections provided below.

Important rules:

- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not recommend treatment.
- Do not invent information.
- Do not invent test values.
- Use simple language.
- Only use information supported by
  the document.

If the answer is not present in the
document, say:

"I could not find that information
in the uploaded document."

For medical decisions, recommend
consulting a qualified healthcare
professional.

DOCUMENT:

{context}
"""
            ),

            (
                "human",
                "{question}"
            )
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    # Return the answer and the sections
    # used to generate it
    sources = [
        document.page_content
        for document in results
    ]

    return response.content, sources


def make_summary(llm, text):
    """
    Generate a simple summary of the
    uploaded medical document.
    """

    # Avoid sending an extremely large document
    text = text[:12000]

    prompt = f"""
You are HealthDoc AI.

Summarize the following medical document
in simple and clear language.

Follow these rules:

- Use only information present in the document.
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not recommend treatment.
- Do not invent information.
- Do not change reported test values.
- Explain important technical terms simply.
- Keep the summary organized.

At the end, mention that the summary is
for educational purposes and is not a
medical diagnosis.

DOCUMENT:

{text}
"""

    response = llm.invoke(prompt)

    return response.content
