import streamlit as st
from markdown import markdown
from genai_services import summarize_text, chunk_text
from chroma_services import ingest_documents
import tempfile
import os

# Import MarkItDown with error handling
try:
    from markitdown import MarkItDown
except ImportError:
    st.error("MarkItDown not installed. Please install it with: pip install markitdown")
    st.stop()

st.title("Document Ingestion & Summarization")

uploaded_file = st.file_uploader(
    "Upload a document (txt, pdf, or any text-based file supported by markitdown)",
    type=[
        "txt", "pdf", "md", "html", "docx"
    ]
)

if uploaded_file:
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Convert to text using markitdown
        converter = MarkItDown()
        doc_text = converter.convert(tmp_path).text_content
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        st.subheader("Document Preview:")
        st.text_area("Extracted Text", doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text, height=200)

        # Summarize
        with st.spinner("Summarizing document..."):
            summary = summarize_text(doc_text)
        
        st.subheader("Summary:")
        st.write(summary)
        
        # Upload button
        if st.button("Upload & Ingest to Chroma DB"):
            # Chunk and ingest
            with st.spinner("Ingesting document..."):
                chunks = chunk_text(doc_text)
                num_chunks = ingest_documents(chunks)
                
                if num_chunks > 0:
                    st.success(f"Successfully ingested {num_chunks} chunks into ChromaDB!")
                else:
                    st.error("Failed to ingest document.")
        
        # Navigation to chatbot
        if st.button("Go to Chatbot"):
            st.switch_page("pages/chatbot_page.py")
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        # Clean up temp file if it exists
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass