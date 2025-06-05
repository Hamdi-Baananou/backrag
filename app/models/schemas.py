from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExtractionRequest(BaseModel):
    """Request model for extraction endpoint"""
    part_number: Optional[str] = Field(None, description="Optional part number for web data lookup")

class ExtractionResult(BaseModel):
    """Model for a single extraction result"""
    prompt_name: str = Field(..., description="Name of the extraction prompt")
    extracted_value: str = Field(..., description="Extracted value")
    source: str = Field(..., description="Source of the extraction (PDF or Web)")
    raw_output: Optional[str] = Field(None, description="Raw LLM output")
    parse_error: Optional[str] = Field(None, description="Any parsing errors")
    latency: float = Field(..., description="Processing time in seconds")

class ExtractionResponse(BaseModel):
    """Response model for extraction endpoint"""
    results: List[ExtractionResult] = Field(..., description="List of extraction results")
    metrics: Dict[str, Any] = Field(..., description="Summary metrics of the extraction")
    processed_files: List[str] = Field(..., description="List of processed file names")
