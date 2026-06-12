from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    docs: list[dict] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append({"page_content": text, "metadata": {"page": idx}})
    return docs


def split_docs(raw_docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked: list[dict] = []
    for doc in raw_docs:
        chunks = splitter.split_text(doc["page_content"])
        for chunk in chunks:
            chunked.append({"page_content": chunk, "metadata": doc["metadata"]})
    return chunked


def main() -> int:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "한글맞춤법 표준어규정 해설.pdf"
    db_path = Path(os.environ.get("CHROMA_DB_PATH", "./chroma_db"))
    if not db_path.is_absolute():
        db_path = (root / db_path).resolve()
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "korean_pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_docs = load_pdf_pages(pdf_path)
    if not raw_docs:
        raise RuntimeError("No extractable text found in PDF")

    chunked_docs = split_docs(raw_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # Rebuild collection content to ensure deterministic eval.
    ids = vectorstore.get().get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)

    texts = [d["page_content"] for d in chunked_docs]
    metadatas = [d["metadata"] for d in chunked_docs]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    final_count = len(vectorstore.get().get("ids", []))
    print(f"indexed_chunks={final_count}")
    print(f"db_path={db_path}")
    print(f"collection={collection_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
