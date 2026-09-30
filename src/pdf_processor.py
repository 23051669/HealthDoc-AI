import io
import fitz

def extract_text_from_pdf(pdf_data):
    """
    Read the uploaded PDF and return its text.
    """

    pdf = fitz.open(
        stream=io.BytesIO(pdf_data),
        filetype="pdf"
    )

    text = ""

    for page in pdf:

        page_text = page.get_text()

        if page_text:
            text += page_text + "\n"

    pdf.close()

    return text.strip()

def split_text(
    text,
    chunk_size=1000,
    overlap=150
):
    """
    Break a large document into smaller sections.
    """

    if not text:
        return []

    if overlap >= chunk_size:

        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        section = text[start:end].strip()

        if section:
            chunks.append(section)

        if end >= len(text):
            break

        start = end - overlap

    return chunks
