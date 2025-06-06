from typing import List, Dict, Optional
from fastapi import UploadFile
from loguru import logger
import json
import time
import os
import tempfile

from ..RAG_interface.llm_interface import initialize_llm, create_pdf_extraction_chain, create_web_extraction_chain
from ..RAG_interface.pdf_processor import process_uploaded_pdfs
from ..RAG_interface.vector_store import setup_vector_store, get_embedding_function
from ..RAG_interface.extraction_prompts import prompts_to_run
from ..RAG_interface.extraction_prompts_web import prompts_to_run as web_prompts
from ..models.schemas import ExtractionResult, ExtractionResponse

class ExtractionService:
    def __init__(self):
        """Initialize the extraction service with LLM and embeddings"""
        self.llm = None
        self.embedding_function = None
        self.pdf_chain = None
        self.web_chain = None
        self.initialize()

    def initialize(self):
        """Initialize LLM and embedding models"""
        try:
            self.embedding_function = get_embedding_function()
            self.llm = initialize_llm()
            logger.success("Successfully initialized LLM and embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise

    async def process_files(
        self,
        files: List[UploadFile],
        part_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded files and extract information
        
        Args:
            files: List of uploaded PDF files
            part_number: Optional part number for web data lookup
            
        Returns:
            Dictionary containing extraction results and metrics
        """
        start_time = time.time()
        results = []
        processed_files = []

        try:
            # Create temporary directory for PDF processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temp directory
                for file in files:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        f.write(await file.read())
                    processed_files.append(file.filename)

                # Process PDFs
                pdf_docs = await process_uploaded_pdfs(files, temp_dir)
                
                if pdf_docs:
                    # Setup vector store
                    retriever = setup_vector_store(pdf_docs, self.embedding_function)
                    
                    # Create extraction chains
                    self.pdf_chain = create_pdf_extraction_chain(retriever, self.llm)
                    self.web_chain = create_web_extraction_chain(self.llm)

                    # Process all attributes
                    for prompt_name, prompt_info in prompts_to_run.items():
                        try:
                            # Process with PDF chain
                            pdf_result = await self.pdf_chain.arun(
                                prompt_info["pdf_prompt"]
                            )
                            
                            # Process with web chain if part number provided
                            web_result = None
                            if part_number and prompt_name in web_prompts:
                                web_result = await self.web_chain.arun(
                                    web_prompts[prompt_name]["web_prompt"]
                                )

                            # Add results
                            results.append({
                                "prompt_name": prompt_name,
                                "extracted_value": pdf_result or web_result or "NOT FOUND",
                                "source": "WEB" if web_result else "PDF",
                                "raw_output": web_result or pdf_result,
                                "latency": time.time() - start_time
                            })

                        except Exception as e:
                            logger.error(f"Error processing prompt {prompt_name}: {e}")
                            results.append({
                                "prompt_name": prompt_name,
                                "extracted_value": "ERROR",
                                "source": "ERROR",
                                "parse_error": str(e),
                                "latency": time.time() - start_time
                            })

        except Exception as e:
            logger.error(f"Error in file processing: {e}")
            raise

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        return {
            "results": results,
            "metrics": metrics,
            "processed_files": processed_files
        }

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate extraction metrics"""
        total_fields = len(results)
        success_count = sum(1 for r in results if r["extracted_value"] != "NOT FOUND" and r["extracted_value"] != "ERROR")
        error_count = sum(1 for r in results if r["extracted_value"] == "ERROR")
        not_found_count = sum(1 for r in results if r["extracted_value"] == "NOT FOUND")
        
        total_latency = sum(r["latency"] for r in results)
        avg_latency = total_latency / total_fields if total_fields > 0 else 0

        return {
            "total_fields": total_fields,
            "success_count": success_count,
            "error_count": error_count,
            "not_found_count": not_found_count,
            "success_rate": (success_count / total_fields * 100) if total_fields > 0 else 0,
            "error_rate": (error_count / total_fields * 100) if total_fields > 0 else 0,
            "not_found_rate": (not_found_count / total_fields * 100) if total_fields > 0 else 0,
            "average_latency": avg_latency
        } 