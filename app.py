"""Buscador de PDFs con agente local + LLM remoto (OpenAI GPT-4o-mini)."""

from __future__ import annotations

import os
import sys
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Permitir importar ingest desde la misma carpeta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest  # noqa: E402

from openai import OpenAI  # noqa: E402
import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

load_dotenv()

PDF_DIR = "./pdfs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdfs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5


def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def embed_query(text: str) -> list[float]:
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([text], convert_to_numpy=False)[0]
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)


def search_chroma(query_emb: list[float], top_k: int = TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    if not results["ids"] or not results["ids"][0]:
        return hits
    for id_, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "id": id_,
            "text": doc,
            "source": meta.get("source", "desconocido"),
            "page": meta.get("page", "?"),
            "distance": dist,
        })
    return hits


def build_context(hits: list[dict]) -> str:
    if not hits:
        return "No se encontro contexto relevante en los PDFs indexados."
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Fuente {i}] {h['source']} — pagina {h['page']}\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def ask_llm(client: OpenAI, question: str, context: str) -> str:
    system = (
        "Eres un asistente que responde SOLO usando el contexto proporcionado. "
        "Responde en español. Si la respuesta no esta en el contexto, di que no "
        "sabes. Cita siempre la fuente y pagina entre parentesis como "
        "(archivo.pdf — pagina X). No inventes informacion."
    )
    user = (
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n\n"
        "Responde citando fuentes y paginas."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "Sin respuesta."


def main() -> None:
    st.set_page_config(page_title="Buscador de PDFs", page_icon="📄")
    st.title("📄 Buscador de PDFs con agente")
    st.caption("Todo local excepto el modelo de lenguaje (OpenAI GPT-4o-mini).")

    with st.sidebar:
        st.header("⚙️ Configuracion")
        api_key = st.text_input(
            "OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Se necesita solo para el LLM (remoto). Los embeddings y Chroma son locales.",
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key

        st.divider()
        st.subheader("1. Subir PDFs")
        uploaded = st.file_uploader(
            "Selecciona uno o varios PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded:
            os.makedirs(PDF_DIR, exist_ok=True)
            for u in uploaded:
                path = os.path.join(PDF_DIR, u.name)
                with open(path, "wb") as f:
                    f.write(u.getbuffer())
                st.write(f"Guardado: {u.name}")
            st.success(f"{len(uploaded)} PDF(s) guardado(s) en ./{PDF_DIR}/")

        st.divider()
        st.subheader("2. Indexar PDFs")
        if st.button("Indexar ahora", type="primary"):
            with st.spinner("Indexando PDFs..."):
                rc = ingest.main()
            if rc == 0:
                st.success("Indexacion completa.")
            else:
                st.error("Error en la indexacion.")

        st.divider()
        st.subheader("Estado")
        chroma_exists = os.path.exists(CHROMA_DIR)
        st.write(f"Chroma DB: {'existe' if chroma_exists else 'no creada aun'}")
        pdf_count = len(
            [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        ) if os.path.exists(PDF_DIR) else 0
        st.write(f"PDFs en carpeta: {pdf_count}")

    st.divider()
    st.subheader("3. Hacer una pregunta")
    question = st.text_input(
        "¿Sobre qué tema quieres buscar en tus PDFs?",
        placeholder="Ej: ¿qué dice el contrato sobre rescisión?",
    )

    if st.button("Buscar", disabled=not question):
        if not get_openai_client():
            st.error("Necesitas configurar tu OPENAI API Key en la barra lateral.")
            return

        if not os.path.exists(CHROMA_DIR):
            st.error("Primero indexa los PDFs (paso 2 en la barra lateral).")
            return

        with st.spinner("Buscando en Chroma y consultando al modelo..."):
            query_emb = embed_query(question)
            hits = search_chroma(query_emb, top_k=TOP_K)
            context = build_context(hits)
            client = get_openai_client()
            answer = ask_llm(client, question, context)

        st.markdown("### Respuesta")
        st.write(answer)

        if hits:
            st.markdown("### Fuentes")
            for i, h in enumerate(hits, start=1):
                st.markdown(
                    f"{i}. **{h['source']}** — pagina {h['page']} "
                    f"(relevancia: {1 - h['distance']:.2f})"
                )
                with st.expander(f"Ver fragmento {i}"):
                    st.write(h["text"])


if __name__ == "__main__":
    main()
