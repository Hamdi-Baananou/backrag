"""
PDF processing utilities for the RAG interface
"""

import os
import tempfile
from typing import List, Optional
from fastapi import UploadFile
from loguru import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

async def process_uploaded_pdfs(files: List[UploadFile], temp_dir: str) -> Optional[List]:
    """
    Process uploaded PDF files and return a list of document chunks.
    
    Args:
        files: List of uploaded PDF files
        temp_dir: Temporary directory to store files
        
    Returns:
        List of document chunks or None if processing fails
    """
    if not files:
        logger.warning("No files provided for processing")
        return None

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            
            # Load and split PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Split documents into chunks
            splits = text_splitter.split_documents(docs)
            
            # Add metadata
            for split in splits:
                split.metadata.update({
                    "source": file.filename,
                    "page": split.metadata.get("page", "N/A"),
                    "start_index": split.metadata.get("start_index", None)
                })
            
            all_docs.extend(splits)
            logger.info(f"Processed {len(splits)} chunks from {file.filename}")

        if not all_docs:
            logger.warning("No document chunks were created from the uploaded files")
            return None

        logger.success(f"Successfully processed {len(all_docs)} total chunks from {len(files)} files")
        return all_docs

    except Exception as e:
        logger.error(f"Error processing PDF files: {e}", exc_info=True)
        return None 