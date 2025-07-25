import os
import tiktoken
import streamlit as st
from typing import List
from openai import OpenAI

# Check for API key first
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Please set up your OPENAI_API_KEY in Streamlit secrets")
    st.stop()

# Set OpenAI client with Gemini API configuration
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Default model name if not specified
MODEL_NAME = "gemini-1.5-flash"

def call_llm(messages: List[dict]) -> str:
    """Helper function to call Gemini API"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."


def summarize_text(text: str) -> str:
    """
    Generate a summary of the text using LLM

    Args:
        text: Text to summarize

    Returns:
        Summary of the text
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes documents accurately and concisely."
        },
        {
            "role": "user",
            "content": f"Please summarize the following text concisely while capturing the key points:\n\n{text}"
        }
    ]

    return call_llm(messages)


def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 10) -> List[str]:
    """
    Split text into overlapping chunks of specified size

    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    if not text:
        return []

    try:
        # Use tiktoken to count tokens
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)

        # Create chunks with overlap
        chunks = []
        i = 0
        while i < len(tokens):
            # Get chunk of size chunk_size
            chunk_end = min(i + chunk_size, len(tokens))
            chunks.append(enc.decode(tokens[i:chunk_end]))
            # Move with overlap
            i = chunk_end - chunk_overlap if chunk_end < len(tokens) else chunk_end

        return chunks
    except Exception as e:
        # Fallback to simple text splitting if tiktoken fails
        words = text.split()
        chunks = []
        chunk_size_words = chunk_size * 4  # Rough approximation
        overlap_words = chunk_overlap * 4
        
        i = 0
        while i < len(words):
            chunk_end = min(i + chunk_size_words, len(words))
            chunks.append(" ".join(words[i:chunk_end]))
            i = chunk_end - overlap_words if chunk_end < len(words) else chunk_end
        
        return chunks


def answer_with_context(question: str, contexts: List[str]) -> str:
    """
    Generate a response to a query using context from RAG

    Args:
        question: User's question
        contexts: List of relevant document chunks from ChromaDB

    Returns:
        LLM response to the question
    """
    # Combine context into a single string with limited length
    combined_context = "\n\n---\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. If you don't know the answer based on the context, say so."
        },
        {
            "role": "user",
            "content": f"Context information:\n\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]

    return call_llm(messages)
