from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import os
from app.models.schemas import ExtractionResponse
from app.services.extractor import ExtractionService

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
    part_number: Optional[str] = None,
    extraction_service: ExtractionService = Depends(get_extraction_service)
):
    """
    Extract information from uploaded PDF files and optional part number.
    
    Args:
        files: List of PDF files to process
        part_number: Optional part number for web data lookup
        extraction_service: Extraction service instance
    
    Returns:
        ExtractionResponse containing results and metrics
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
            
        results = await extraction_service.process_files(files, part_number)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True) 