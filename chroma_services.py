import chromadb
import streamlit as st
import os
import tempfile
import shutil

# Set environment variables for ChromaDB in Streamlit Cloud
os.environ['ALLOW_RESET'] = 'TRUE'
os.environ['ANONYMIZED_TELEMETRY'] = 'FALSE'

# Global variables to store client and collection
chroma_client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB with proper configuration for Streamlit Cloud"""
    global chroma_client, collection
    
    if chroma_client is None:
        try:
            # Create a temporary directory for ChromaDB data
            if 'chroma_db_path' not in st.session_state:
                st.session_state.chroma_db_path = tempfile.mkdtemp()
            
            # Initialize ChromaDB client with ephemeral settings for cloud deployment
            chroma_client = chromadb.EphemeralClient()
            
            # Use a default collection name
            collection_name = "rag_documents"
            
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Specify distance metric
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            return False
    
    return True


def ingest_documents(docs):
    """Ingest documents into ChromaDB using 'all-MiniLM-L6-v2' Sentence Transformer

    Args:
        docs: list of strings (document chunks)
    """
    if not initialize_chromadb():
        return 0
        
    try:
        # Clear existing documents first (for demo purposes)
        try:
            collection.delete()
        except:
            pass
        
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
    if not initialize_chromadb():
        return []
        
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        if 'documents' in results and results['documents']:
            return results['documents'][0]
        else:
            return []
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return []