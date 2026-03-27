from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import db_setup
import fitz  # pymupdf
from langchain_core.documents import Document

MAX_KEYWORDS = 10

def load_pdf(file_path: str) -> list:
    try:
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    except Exception as e:
        print(f"PyPDFLoader failed ({e}), falling back to PyMuPDF.")
        docs = []
        pdf = fitz.open(str(file_path))
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            docs.append(Document(
                page_content=text,
                metadata={"source": str(file_path), "page": page_num}
            ))
        pdf.close()
        return docs

def trim_keywords(meta: dict) -> dict:
    kw = meta.get("keywords")
    if isinstance(kw, str):
        parts = [k.strip() for k in kw.split(", ") if k.strip()]
        meta["keywords"] = ", ".join(parts[:MAX_KEYWORDS])
    return meta

def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def sanitize_metadata(meta: dict) -> dict:
    return {
        k: sanitize_text(v) if isinstance(v, str) else v
        for k, v in meta.items()
    }

files_directory = Path("./data")
vector_store = db_setup.get_db()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=600,
    length_function=len,
    is_separator_regex=False,
)

for file_path in files_directory.rglob("*.pdf"):
    file = file_path.name
    print(f"Document {file} processing started.")
    docs = load_pdf(file_path)
    print(f"Document {file} loaded.")
    texts = text_splitter.split_documents(docs)
    print(f"Document {file} splitted.")
    ids = []
    documents = []
    chunk_id = 1
    for text in texts:
        text.page_content = sanitize_text(text.page_content)
        text.metadata["source"] = str(file_path)
        text.metadata = sanitize_metadata(text.metadata)
        text.metadata = trim_keywords(text.metadata)
        ids.append(f"{file}-page{text.metadata['page'] + 1}-chunk{chunk_id}")
        documents.append(text)
        chunk_id += 1
    print(f"Document {file} chunks processed.")
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Document {file} chunks saved.")
    
