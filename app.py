import os

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_text_from_pdf, split_text
from src.rag import create_vector_store, ask_question, make_summary
from src.llm import get_llm


# Load the API key from .env
load_dotenv()

# Basic Streamlit settings
st.set_page_config(
    page_title="HealthDoc AI",
    page_icon="🩺",
    layout="wide"
)


# Page heading


st.title("🩺 HealthDoc AI")
st.write("### Medical Document Understanding Assistant")

st.write(
    "Upload a medical PDF and use AI to understand the "
    "information inside the document."
)

st.warning(
    "This project is for educational purposes only. "
    "It does not provide medical diagnosis or treatment advice."
)


# Sidebar


with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        HealthDoc AI uses Generative AI and RAG to
        understand information from uploaded medical
        documents.
        """
    )

    st.write("### Main Features")

    st.write("📄 Upload a PDF")
    st.write("📝 Generate a summary")
    st.write("💬 Ask questions")
    st.write("📚 Explain medical terms")
    st.write("🔍 Search the uploaded document")


# Check API key


if not os.getenv("GROQ_API_KEY"):

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )

    st.stop()


# Upload PDF


uploaded_file = st.file_uploader(
    "Upload your medical PDF",
    type=["pdf"]
)


if uploaded_file is None:

    st.info(
        "Upload a PDF file to start using HealthDoc AI."
    )

    st.stop()


# Read the uploaded PDF


try:

    pdf_data = uploaded_file.getvalue()

    with st.spinner("Reading the PDF..."):

        text = extract_text_from_pdf(pdf_data)

    if not text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        st.stop()

    st.success("PDF loaded successfully!")

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Characters extracted:** {len(text)}"
    )


except Exception as error:

    st.error(
        f"Could not read the PDF: {error}"
    )

    st.stop()


# Split document into smaller pieces


try:

    with st.spinner("Preparing the document..."):

        chunks = split_text(text)

    st.write(
        f"**Number of text sections:** {len(chunks)}"
    )

except Exception as error:

    st.error(
        f"Could not prepare the document: {error}"
    )

    st.stop()


# Create vector database

try:

    with st.spinner("Creating document knowledge base..."):

        vector_store = create_vector_store(chunks)

    st.success(
        "Document is ready for questions!"
    )

except Exception as error:

    st.error(
        f"Could not create the document database: {error}"
    )

    st.stop()

# Generate summary

st.divider()

st.header("📝 Document Summary")

if st.button(
    "Generate Summary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Generating summary..."
        ):

            llm = get_llm()

            summary = make_summary(
                llm,
                text
            )

        st.write(summary)

    except Exception as error:

        st.error(
            f"Could not generate summary: {error}"
        )


# Ask questions

st.divider()

st.header(" Ask Questions About the Document")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What tests are mentioned in the report?"
)


if st.button(
    "Ask Question",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        try:

            with st.spinner(
                "Finding the answer..."
            ):

                llm = get_llm()

                answer, sources = ask_question(
                    llm,
                    vector_store,
                    question
                )

            st.subheader(" Answer")

            st.write(answer)

            with st.expander(
                "View document sections used"
            ):

                for number, source in enumerate(
                    sources,
                    start=1
                ):

                    st.write(
                        f"**Section {number}**"
                    )

                    st.write(source)


        except Exception as error:

            st.error(
                f"Could not answer the question: {error}"
            )


# Explain a medical term

st.divider()

st.header("📚 Explain a Medical Term")

term = st.text_input(
    "Enter a medical term",
    placeholder="Example: Hemoglobin"
)


if st.button(
    "Explain Term",
    use_container_width=True
):

    if not term.strip():

        st.warning(
            "Please enter a medical term."
        )

    else:

        try:

            with st.spinner(
                "Preparing a simple explanation..."
            ):

                llm = get_llm()

                prompt = f"""
Explain the medical term "{term}"
in simple educational language.

Keep the explanation easy for a student
or general user to understand.

Do not diagnose any disease.
Do not prescribe medicines.
Do not recommend treatment.
Do not interpret the user's personal
medical condition.
"""

                response = llm.invoke(prompt)

            st.subheader(
                f"📖 {term}"
            )

            st.write(response.content)

        except Exception as error:

            st.error(
                f"Could not explain the term: {error}"
            )
