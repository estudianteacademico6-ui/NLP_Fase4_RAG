"""Interfaz Streamlit del Orientador Académico RAG."""
import os, uuid, logging
import streamlit as st
from rag_engine import (ROOT, DOCS, MAX_TURNS, clean, get_embedder, retrieve,
                        generate_answer, get_session_summary, save_session_summary,
                        list_documents, index_documents)

st.set_page_config(page_title="Orientador académico documental", page_icon="📚", layout="centered")

@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def cached_embedder():
    return get_embedder()

@st.cache_resource(show_spinner="Cargando modelo de lenguaje Qwen…")
def cached_llm():
    from transformers import AutoTokenizer, pipeline
    from rag_engine import LLM_MODEL
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    generator = pipeline("text-generation", model=LLM_MODEL, tokenizer=tokenizer, device=-1)
    return generator, tokenizer

st.title("📚 Orientador académico documental")
st.caption("Prototipo educativo: las respuestas se generan con apoyo de los documentos cargados. Verifica siempre las fuentes originales.")
st.info("Consulta los archivos públicos de la carpeta knowledge. No introduzcas datos personales ni información sensible.")

if "messages" not in st.session_state: st.session_state.messages = []
if "sid" not in st.session_state: st.session_state.sid = uuid.uuid4().hex

files = list_documents()
st.caption(f"Documentos disponibles: {len(files)}")
with st.expander("Consultar documentos cargados"):
    if files:
        for name in files: st.write(f"📄 {name}")
    else:
        st.warning("No hay archivos TXT en knowledge. Agrega documentos públicos y vuelve a consultar.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Fuentes recuperadas: " + ", ".join(msg["sources"]))

question = st.chat_input("Escribe tu pregunta académica aquí…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando evidencia y preparando la respuesta…"):
                embedder = cached_embedder()
                retrieved = retrieve(question, embedder=embedder)
                summary = get_session_summary(st.session_state.sid)
                if retrieved:
                    generator, tokenizer = cached_llm()
                    response = generate_answer(question, retrieved, st.session_state.messages[:-1], summary,
                                                generator=generator, tokenizer=tokenizer)
                else:
                    response = generate_answer(question, [], st.session_state.messages[:-1], summary)
            st.write(response)
            source_names = sorted({r["source"] for r in retrieved})
            if source_names: st.caption("Fuentes recuperadas: " + ", ".join(source_names))
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": source_names})
            if len(st.session_state.messages) > 2 * MAX_TURNS:
                old = st.session_state.messages[:-2 * MAX_TURNS]
                old_text = " ".join(f"{m['role']}: {clean(m['content'], 160)}" for m in old)
                previous = get_session_summary(st.session_state.sid)
                save_session_summary(st.session_state.sid, (previous + " " + old_text)[-1000:])
                st.session_state.messages = st.session_state.messages[-2 * MAX_TURNS:]
            logging.info("event=query length=%d retrieved=%d", len(question), len(retrieved))
        except Exception:
            logging.exception("Error procesando consulta")
            response = "No fue posible completar la consulta. Revisa la terminal; si el error corresponde a memoria, prueba un equipo con más RAM o un modelo más pequeño."
            st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})

if st.button("Borrar conversación"):
    st.session_state.messages = []
    st.session_state.sid = uuid.uuid4().hex
    st.rerun()

st.divider()
st.caption("Orientador académico RAG | Recuperación documental, embeddings multilingües, generación local y memoria de sesión SQLite.")
st.caption("Las respuestas dependen de la información contenida en los documentos cargados y pueden contener errores.")
