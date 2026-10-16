"""Indexa PDFs locales en Chroma con embeddings locales (sentence-transformers)."""

from __future__ import annotations

import glob
import hashlib
import os
import sys
from typing import Iterable

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

PDF_DIR = "./pdfs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdfs"
EMBED_MODEL = "all-MiniLM-L3-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
INDEXED_FILE = os.path.join(CHROMA_DIR, ".indexed")


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_indexed() -> dict[str, str]:
    if not os.path.exists(INDEXED_FILE):
        return {}
    indexed: dict[str, str] = {}
    with open(INDEXED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            path, digest = line.split("\t", 1)
            indexed[path] = digest
    return indexed


def save_indexed(indexed: dict[str, str]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INDEXED_FILE, "w", encoding="utf-8") as f:
        for path, digest in indexed.items():
            f.write(f"{path}\t{digest}\n")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Devuelve lista de (numero_pagina, texto) para cada pagina del PDF."""
    pages: list[tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Trocea texto respetando parrafos, con solapamiento."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = (current + " " + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # solapamiento: toma los últimos `overlap` chars del chunk anterior
            current = current[-overlap:] + " " + para if current else para
            current = current.strip()
            if len(current) > size:
                # parrafo muy largo: lo trunca
                current = current[:size]
    if current:
        chunks.append(current)
    return chunks


def main() -> int:
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"No se encontraron PDFs en {PDF_DIR}/")
        return 0

    indexed = load_indexed()
    to_index = []
    for pdf in pdfs:
        digest = file_sha1(pdf)
        if indexed.get(pdf) == digest:
            print(f"Ya indexado (sin cambios): {pdf}")
            continue
        to_index.append((pdf, digest))

    if not to_index:
        print("Todos los PDFs ya estan indexados.")
        return 0

    print(f"Cargando modelo de embeddings: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Conectando a Chroma en {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []

    for pdf_path, digest in to_index:
        print(f"Procesando: {pdf_path}")
        pages = extract_pages(pdf_path)
        for page_no, page_text in pages:
            chunks = chunk_text(page_text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{os.path.basename(pdf_path)}::p{page_no}::c{idx}"
                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append({
                    "source": os.path.basename(pdf_path),
                    "page": page_no,
                    "chunk": idx,
                })

    if not texts:
        print("No hay texto nuevo que indexar.")
        return 0

    print(f"Generando embeddings para {len(texts)} chunks...")
    emb = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=False)
    embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in emb]

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    for pdf_path, digest in to_index:
        indexed[pdf_path] = digest
    save_indexed(indexed)

    print(f"Indexados {len(texts)} chunks de {len(to_index)} PDF(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
