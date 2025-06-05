"""
Configuration settings for the RAG interface
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

# LLM Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mixtral-8x7b-32768")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))

# Vector Store Configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Web Scraping Configuration
SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "20000"))  # milliseconds
CACHE_MODE = os.getenv("CACHE_MODE", "BYPASS")  # Options: BYPASS, USE, UPDATE 