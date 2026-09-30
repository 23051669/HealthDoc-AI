# HealthDoc AI

## Medical Document Understanding Assistant

HealthDoc AI is a Generative AI-based project that helps users
understand the information given in medical PDF documents.

Users can upload a medical PDF and use the application to get
a simple summary, ask questions about the document, and understand
difficult medical terms in easier language.

## Main Features

- Upload a medical PDF
- Extract text from the PDF
- Divide the document into smaller sections
- Convert the text into embeddings
- Store the embeddings in ChromaDB
- Find relevant information from the document
- Ask questions using RAG
- Generate a simple summary of the document
- Explain medical terms in easy language
- Follow basic safety rules while generating responses

## Technologies Used

- Python – Main programming language
- Streamlit – Used to build the web interface
- LangChain – Used to connect the different AI components
- Groq – Used for the language model
- ChromaDB – Used to store and search document embeddings
- Sentence Transformers – Used to create text embeddings
- PyMuPDF – Used to extract text from PDF files

## How the Project Works

The project follows a simple process:

PDF Upload
    ↓
Extract Text
    ↓
Split Text into Sections
    ↓
Create Embeddings
    ↓
Store in ChromaDB
    ↓
Find Relevant Sections
    ↓
Send Information to Groq LLM
    ↓
Generate Answer

When a user asks a question, the system first searches the
uploaded document for relevant information. The retrieved
information is then given to the AI model, which generates
an answer based on the document.

This approach is called Retrieval-Augmented Generation (RAG).

## Installation

### 1. Create the project folder

Open the terminal and run:

```bash
mkdir HealthDoc-AI
cd HealthDoc-AI
