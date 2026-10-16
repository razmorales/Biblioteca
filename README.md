# Biblioteca — Buscador de PDFs con agente

Busca por significado dentro de tus PDFs usando un agente. Todo corre **localmente**
excepto el modelo de lenguaje (OpenAI GPT-4o-mini), que es la única parte remota.

## Stack

| Capa | Tecnología | Dónde corre |
|------|-----------|-------------|
| Interfaz | Streamlit | Local |
| Extracción de texto | PyMuPDF (`fitz`) | Local |
| Troceo (chunking) | Python propio (~800 chars, solapamiento 100) | Local |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Local |
| Base vectorial | Chroma (Persistente, en disco) | Local |
| LLM | OpenAI GPT-4o-mini | Remoto (solo para responder) |

## Instalación

```bash
# 1. Clona el repo
git clone git@github.com:razmorales/Biblioteca.git
cd Biblioteca

# 2. Crea entorno e instala dependencias
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configura tu clave de OpenAI (obligatoria para el LLM)
echo "OPENAI_API_KEY=sk-tu-clave-aqui" > .env
# o exportala: export OPENAI_API_KEY=sk-...
```

## Uso

### A) Agrega tus PDFs
Copia los archivos PDF a la carpeta `./pdfs/` o súbelos desde la barra lateral de la app.

### B) Indexa
```bash
streamlit run app.py
```
En la barra lateral, haz clic en **"Indexar ahora"**. Esto extrae texto, lo trocea,
genera embeddings locales y los guarda en `./chroma_db/`.

### C) Busca
Escribe tu pregunta y presiona **Buscar**. La app consulta Chroma, trae los fragmentos
más similares y les pide a GPT-4o-mini que responda citando archivo y página.

## Estructura

```
Biblioteca/
├── app.py                 # Interfaz Streamlit
├── ingest.py              # Indexación de PDFs en Chroma
├── requirements.txt
├── .env                   # OPENAI_API_KEY (no se sube a git)
├── .gitignore
├── pdfs/                  # Tus PDFs (ignorado en git)
├── chroma_db/             # Base vectorial local (ignorado en git)
├── docs/agents/           # setup-matt-pocock-skills output
└── CLAUDE.md              # Bloque ## Agent skills
```

## Notas

- Los PDFs y la base Chroma no se suben a GitHub (están en `.gitignore`).
- Reindexa solo cuando añades o modificas PDFs.
- Si no hay `OPENAI_API_KEY`, la app te pide que la pegues en la barra lateral.
