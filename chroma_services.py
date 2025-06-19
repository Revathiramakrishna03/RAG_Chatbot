import chromadb
import streamlit as st
import os

# Set environment variable to allow ChromaDB reset (fixes runtime error)
os.environ['ALLOW_RESET'] = 'TRUE'

try:
    # Initialize ChromaDB client with error handling
    chroma_client = chromadb.Client()
    
    # Use a default collection name if not specified
    collection_name = "rag_documents"  # Default name
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )
except Exception as e:
    st.error(f"Error initializing ChromaDB: {str(e)}")
    st.stop()


def ingest_documents(docs):
    """Ingest documents into ChromaDB using 'all-MiniLM-L6-v2' Sentence Transformer

    Args:
        docs: list of strings (document chunks)
    """
    try:
        # Ids for the docs
        ids = [f"chunk_{i}" for i in range(len(docs))]

        # Ingest chunks into the collection
        collection.add(documents=docs, ids=ids)

        return len(docs)
    except Exception as e:
        st.error(f"Error ingesting documents: {str(e)}")
        return 0


def query_documents(query_text, n_results=3):
    """Query the collection for relevant documents

    Args:
        query_text: string to search for
        n_results: number of results to return

    Returns:
        List of relevant document chunks
    """
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        if 'documents' in results and results['documents']:
            return results['documents'][0]
        else:
            return []
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return []