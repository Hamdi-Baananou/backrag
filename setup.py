from setuptools import setup, find_packages

setup(
    name="pdf-extraction-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
    ],
) 