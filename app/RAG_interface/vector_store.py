"""
Vector store utilities for the RAG interface
"""

import os
from typing import List, Optional
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def get_embedding_function():
    """
    Initialize and return the embedding function.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Successfully initialized embedding function")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
        raise

def setup_vector_store(documents: List[Document], embedding_function) -> Optional[Chroma]:
    """
    Set up a vector store with the given documents.
    
    Args:
        documents: List of documents to store
        embedding_function: Function to generate embeddings
        
    Returns:
        Chroma vector store instance or None if setup fails
    """
    if not documents:
        logger.warning("No documents provided for vector store setup")
        return None

    try:
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory="vector_store"
        )
        
        logger.info(f"Successfully created vector store with {len(documents)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to set up vector store: {e}", exc_info=True)
        return None 