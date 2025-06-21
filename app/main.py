from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Dict
import logging
from langchain.chains import SequentialChain, TransformChain
from .model import CancerClassifier, CancerExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Text Processing API",
    description="API for cancer-related text classification and information extraction",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: Union[str, List[str]]

class ProcessingResult(BaseModel):
    text: str
    classification: Union[str, dict]
    extraction: Union[str, dict]
    error: Optional[str] = None

class BatchResponse(BaseModel):
    results: List[ProcessingResult]

# Initialize models
try:
    logger.info("Loading classification model...")
    classification_pipeline = CancerClassifier("user1729/BiomedBERT-cancer-bert-classifier-v1.0")
    
    logger.info("Loading extraction model...")
    extraction_pipeline = CancerExtractor()
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise RuntimeError("Could not initialize models")

def batch_classification_transform(inputs: Dict) -> Dict:
    """Process batch of texts through classification model"""
    try:
        texts = inputs["input_texts"]
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to batch of one
            
        results = []
        for text in texts:
            try:
                result = classification_pipeline.predict(text)
                results.append(str(result))
            except Exception as e:
                logger.warning(f"Classification failed for text: {text[:50]}... Error: {str(e)}")
                results.append({"error": str(e)})
                
        return {"classification_results": results}
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise

def batch_extraction_transform(inputs: Dict) -> Dict:
    """Process batch of texts through extraction model"""
    try:
        texts = inputs["input_texts"]
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to batch of one
            
        results = []
        for text in texts:
            try:
                result = extraction_pipeline.predict(text)
                results.append(str(result))
            except Exception as e:
                logger.warning(f"Extraction failed for text: {text[:50]}... Error: {str(e)}")
                results.append({"error": str(e)})
                
        return {"extraction_results": results}
    except Exception as e:
        logger.error(f"Batch extraction failed: {str(e)}")
        raise

# Create processing chains
classification_chain = TransformChain(
    input_variables=["input_texts"],
    output_variables=["classification_results"],
    transform=batch_classification_transform
)

extraction_chain = TransformChain(
    input_variables=["input_texts"],
    output_variables=["extraction_results"],
    transform=batch_extraction_transform
)

# Create sequential chain
processing_chain = SequentialChain(
    chains=[classification_chain, extraction_chain],
    input_variables=["input_texts"],
    output_variables=["classification_results", "extraction_results"],
    verbose=True
)

@app.post("/process", response_model=BatchResponse)
async def process_texts(input: TextInput):
    """
    Process cancer-related texts through classification and extraction pipeline
    
    Args:
        input: TextInput object containing either a single string or list of strings
    
    Returns:
        BatchResponse with processing results for each input text
    """
    try:
        texts = [input.text] if isinstance(input.text, str) else input.text
        
        # Validate input
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise HTTPException(status_code=400, detail="Input must be string or list of strings")
        
        # Process through LangChain pipeline
        chain_result = processing_chain({"input_texts": texts})
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            classification = chain_result["classification_results"][i]
            extraction = chain_result["extraction_results"][i]
            
            error = None
            if isinstance(classification, dict) and "error" in classification:
                error = classification["error"]
            elif isinstance(extraction, dict) and "error" in extraction:
                error = extraction["error"]
            
            results.append(ProcessingResult(
                text=text,
                classification=classification,
                extraction=extraction,
                error=error
            ))
        
        return BatchResponse(results=results)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test with a simple cancer-related phrase
        test_text = "breast cancer diagnosis"
        classification_pipeline.predict(test_text)
        extraction_pipeline.predict(test_text)
        return {"status": "healthy", "models": ["classification", "extraction"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
