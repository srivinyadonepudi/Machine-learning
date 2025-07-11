from PyPDF2 import PdfReader
import docx
import re

MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, max_len=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(clean_text(text[start:end]))
        start += max_len - overlap
    return chunks

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def load_txt(file):
    return file.read().decode('utf-8')

def load_and_chunk_document(file):
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
