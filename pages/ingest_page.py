import streamlit as st
from markdown import markdown
from genai_services import summarize_text, chunk_text
from chroma_services import ingest_documents
import tempfile
import os

st.title("Document Ingestion & Summarization")
uploaded_file = st.file_uploader(
    "Upload a document (txt, pdf, or any text-based file supported by markitdown)",
    type=[
        "txt", "pdf", "md", "html", "docx"
    ]
)
if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert to text using markitdown
    converter = MarkItDown()
    doc_text = converter.convert(tmp_path).text_content
    st.subheader("Document Preview:")
    st.text_area("Extracted Text", doc_text, height=200)

    # Summarize
    with st.spinner("Summarizing document..."):
        summary = summarize_text(doc_text)
    st.subheader("Summary:")
    st.write(summary)
    # Upload button        if st.button("Upload & Ingest to Chroma DB"):
    # Chunk and ingest
    with st.spinner("Ingesting document..."):
        chunks = chunk_text(doc_text)
        ingest_documents(chunks)
    if st.button("Chatbot"):
        st.switch_page("pages/chatbot_page.py")
