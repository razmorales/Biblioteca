# Plan de verificación — Buscador de PDFs (repo `razmorales/Biblioteca`)

Este plan permite verificar que la app funciona en tu máquina local. Todo corre localmente excepto el LLM (opcional).

## Prerrequisitos
- Python 3.12+ instalado (`python3 --version`)
- Git configurado y acceso SSH a GitHub (`ssh -T git@github.com`)
- Una clave de OpenAI (opcional; sin ella la app funciona en modo búsqueda local)

---

## Paso 1: Entrar al repo y actualizar

```bash
cd ~/Biblioteca
git pull origin main
```

Resultado esperado: `Updating ...` / `Fast-forward` o `Already up to date`.

---

## Paso 2: Crear entorno virtual e instalar dependencias

**Importante:** usa `python3` (no `python`) y `.venv/bin/pip` directamente (no uses `source` en este shell).

```bash
cd ~/Biblioteca
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Resultado esperado: termina con `Successfully installed ...`. Puede tardar 2-5 minutos (descarga `sentence-transformers` y su modelo `all-MiniLM-L6-v2`).

Si falla con `externally-managed-environment`, ignora: estamos usando el venv, no el sistema.

---

## Paso 3: Copiar PDFs de prueba

```bash
mkdir -p pdfs
cp /ruta/a/tu/archivo.pdf pdfs/
ls -la pdfs
```

Reemplaza `/ruta/a/tu/archivo.pdf` por la ruta de un PDF tuyo que tenga texto digital (no escaneado).

---

## Paso 4: Ejecutar la app

```bash
cd ~/Biblioteca
.venv/bin/streamlit run app.py
```

Resultado esperado:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Abre `http://localhost:8501` en tu navegador.

---

## Paso 5: Probar end-to-end

En la barra lateral de la app:

1. **Paso 1 (Subir PDFs):** si ya copiaste PDFs a `./pdfs/`, salta este paso. Si no, súbelos desde el file uploader.
2. **Paso 2 (Indexar ahora):** clic en el botón. Espera `Indexación completa.`
3. **Paso 3 (Hacer una pregunta):** escribe algo relacionado con tus PDFs y clic en **Buscar**.

**Resultado esperado (modo local, sin clave de OpenAI):**
- La app muestra los fragmentos más similares con `archivo.pdf — pagina X` y una puntuación de relevancia.
- Cada fragmento tiene un expander para ver el texto completo.

**Resultado esperado (con clave de OpenAI en `.env` o barra lateral):**
- Además de lo anterior, se genera una respuesta en español citando fuentes y páginas.

---

## Paso 6: Reportar

Después de ejecutar, responde con:
1. ¿Paso 2 terminó sin errores?
2. ¿Paso 5 abrió la app en `http://localhost:8501`?
3. ¿Paso 6 mostró fragmentos con archivo y página?

Si algún paso falla, copia y pega el mensaje de error completo.

---

## Archivos del repo

| Archivo | Función |
|---------|---------|
| `app.py` | Interfaz Streamlit (subir PDFs, indexar, buscar) |
| `ingest.py` | Extrae texto (PyMuPDF), trocea, embeddings locales, guarda en Chroma |
| `requirements.txt` | Dependencias Python |
| `.gitignore` | Ignora `pdfs/`, `chroma_db/`, `.env`, `__pycache__` |
| `README.md` | Instrucciones de uso |
| `PLAN_VERIFICACION.md` | Este plan |
| `docs/agents/` | Output de `setup-matt-pocock-skills` |
| `CLAUDE.md` | Bloque `## Agent skills` |

---

## Stack

PyMuPDF → chunks (~800 chars) → sentence-transformers (`all-MiniLM-L6-v2`) → Chroma (local) → búsqueda semántica → (opcional) OpenAI GPT-4o-mini → respuesta con citas → Streamlit.
