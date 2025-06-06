from setuptools import setup, find_namespace_packages

setup(
    name="pdf-extraction-api",
    version="1.0.0",
    package_dir={"": "app"},
    packages=find_namespace_packages(include=["app", "app.*"]),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "pydantic>=1.8.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=0.19.0",
        "loguru>=0.5.3",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "langchain-groq>=0.0.1",
        "chromadb>=0.4.0",
        "pysqlite3==0.5.2",
        "groq==0.4.2",
        "beautifulsoup4>=4.9.3",
        "crawl4ai>=0.1.0",
        "sentence-transformers>=2.2.2",
        "PyPDF2>=3.0.0"
    ],
    python_requires=">=3.9",
) 