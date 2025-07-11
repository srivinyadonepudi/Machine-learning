import tempfile
from typing import List, Tuple
from PyPDF2 import PdfReader
import docx
import re

MAX_CHUNK_SIZE = 500  # chars
CHUNK_OVERLAP = 50    # chars

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, max_len=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_len, text_len)
        chunk = text[start:end]
        chunks.append(clean_text(chunk))
        start += max_len - overlap
    return chunks

def load_pdf(file) -> str:
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def load_docx(file) -> str:
    doc = docx.Document(file)
    full_text = "\n".join([p.text for p in doc.paragraphs])
    return full_text

def load_txt(file) -> str:
    return file.read().decode('utf-8')

def load_and_chunk_document(file) -> List[Tuple[str, str]]:
    """
    Returns List of (doc_id, chunk_text)
    doc_id can be filename or unique id
    """
    filename = file.name if hasattr(file, "name") else "uploaded_doc"
    ext = filename.split(".")[-1].lower()

    if ext == "pdf":
        text = load_pdf(file)
    elif ext == "docx":
        text = load_docx(file)
    elif ext == "txt":
        text = load_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = chunk_text(text)
    return [(filename, chunk) for chunk in chunks]
