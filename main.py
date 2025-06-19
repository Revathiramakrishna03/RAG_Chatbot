import streamlit as st

st.set_page_config(page_title="RAG QnA & Summarization Chatbot", layout="wide")
ingest_page = st.Page("pages/ingest_page.py", title="Ingest")
chatbot_page = st.Page("pages/chatbot_page.py", title="Chatbot")

pg = st.navigation([
    ingest_page,
    chatbot_page
])
pg.run()
