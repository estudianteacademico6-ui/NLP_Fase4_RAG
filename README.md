# Orientador académico documental – NLP Fase 4

Prototipo RAG académico desarrollado con Streamlit, Sentence-Transformers, SQLite y el modelo abierto Qwen2.5-0.5B-Instruct.

## Componentes

- `app.py`: interfaz conversacional Streamlit.
- `rag_engine.py`: lectura de documentos, fragmentación, embeddings, recuperación, memoria y generación.
- `knowledge/`: documentos `.txt` públicos que alimentan la base de conocimiento.
- `memory.db`: se crea automáticamente; almacena fragmentos/embeddings y resúmenes de sesión. No se debe subir con datos reales.
- `notebooks/Fase4_NLP_RAG.ipynb`: desarrollo reproducible y protocolo de pruebas.
- `evaluacion_10_consultas.csv`: plantilla para registrar resultados reales.

## Requisitos

Python 3.10 o superior. La primera ejecución requiere conexión a Internet para descargar los modelos. Qwen se ejecuta localmente en CPU por defecto y puede tardar o requerir memoria considerable.

### macOS / Linux

```bash
cd NLP_Fase4_actualizado
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Configuración opcional

```bash
export LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export EMBED_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
export TOP_K="3"
export MIN_SCORE="0.20"
export MAX_TURNS="4"
```

Para agregar conocimiento, coloca documentos `.txt` públicos y autorizados dentro de `knowledge/`. El motor detecta cambios y actualiza el índice SQLite automáticamente.

## Ejecución de las pruebas

Abre `notebooks/Fase4_NLP_RAG.ipynb`, ejecuta las celdas en orden y conserva las salidas reales. La matriz `evaluacion_10_consultas.csv` se completa con las respuestas observadas; no rellenar resultados antes de ejecutar.

## Despliegue

El despliegue público no está incluido ni se declara como completado. Antes de publicar, revisa el consumo de memoria y el tiempo de carga del modelo en el proveedor gratuito. El modelo local puede superar los límites de una instancia gratuita. No subas `memory.db`, `.venv`, datos personales ni archivos privados.

## Limitaciones conocidas

- La calidad depende del contenido y cobertura de `knowledge/`.
- La búsqueda vectorial usa embeddings persistidos en SQLite y similitud coseno calculada con NumPy; es adecuada para un prototipo pequeño, no para grandes corpus.
- El modelo pequeño puede producir respuestas incompletas o lentas.
- La memoria SQLite conserva resúmenes de sesión, no perfiles personales.
- Se requiere evaluación empírica para estimar precisión, utilidad y latencia.
