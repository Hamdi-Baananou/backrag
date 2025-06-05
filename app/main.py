from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from loguru import logger
import tempfile
import os
import uvicorn

from services.extractor import ExtractionService
from models.schemas import ExtractionRequest, ExtractionResponse

app = FastAPI(
    title="PDF Extraction API",
    description="API for extracting information from PDFs and web data using LLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get extraction service
def get_extraction_service():
    service = ExtractionService()
    return service

@app.get("/")
async def root():
    return {"message": "PDF Extraction API is running"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_information(
    files: List[UploadFile] = File(...),
    request: ExtractionRequest = None,
    extraction_service: ExtractionService = Depends(get_extraction_service)
):
    """
    Extract information from uploaded PDF files and optional part number.
    
    Args:
        files: List of PDF files to process
        request: Optional request body containing part number
        extraction_service: Extraction service instance
    
    Returns:
        ExtractionResponse containing results and metrics
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
            
        part_number = request.part_number if request else None
        results = await extraction_service.process_files(files, part_number)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 