import streamlit as st
from genai_services import answer_with_context
from chroma_services import query_documents

st.title("RAG QnA Chatbot")
st.write("Ask questions about your ingested document!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("Your question:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Query Chroma for context
    with st.spinner("Searching for relevant information..."):
        context_chunks = query_documents(user_query, n_results=3)

    # Generate response
    with st.spinner("Generating answer..."):
        if context_chunks:
            answer = answer_with_context(user_query, context_chunks)
        else:
            answer = "I couldn't find relevant information in the ingested documents to answer your question. Please make sure you have uploaded and ingested documents first."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # Show context in an expander
        if context_chunks:
            with st.expander("Show retrieved context"):
                for i, chunk in enumerate(context_chunks):
                    st.write(f"**Context {i+1}:**")
                    st.write(chunk)
                    st.write("---")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Navigation back to ingest page
if st.button("Go to Document Ingestion"):
    st.switch_page("pages/ingest_page.py")