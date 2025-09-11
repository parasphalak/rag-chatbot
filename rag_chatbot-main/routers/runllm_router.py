# routers/hackrx.py

import os
import tempfile
import requests
import secrets
import logging
import json
from typing import List, AsyncGenerator, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from fastapi.security import APIKeyHeader
import config

import time

# Import your services
from services.initialise_llm import initialize_llm_and_embeddings, create_rag_chain, astream_rag_response
from services.initialise_vectordb import initialize_pinecone, get_vector_store, setup_knowledge_base

# This tells FastAPI to look for a header named "Authorization"
api_key_header_scheme = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(auth_header: str = Depends(api_key_header_scheme)):
    """Dependency that extracts and validates the API key."""
    if auth_header is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header is missing.")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme. Must be 'Bearer'.")
    
    api_key = auth_header.removeprefix("Bearer ")
    if not secrets.compare_digest(api_key, config.VALID_API_KEY):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key.")
    return api_key

# Initialize router
router = APIRouter()

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
class QuestionAnswer(BaseModel):
    question: str
    answer: str
    
class DocumentResponse(BaseModel):
    status: str
    message: str
    results: List[QuestionAnswer]
    total_questions: int

# Global variables for initialized services
llm, embeddings_model, pinecone_index, vector_store, rag_chain = None, None, None, None, None

def initialize_services_sync():
    """Synchronous version of service initialization to be called on startup."""
    global llm, embeddings_model, pinecone_index, vector_store, rag_chain
    try:
        logging.info("Initializing services on startup...")
        llm, embeddings_model = initialize_llm_and_embeddings()
        pinecone_index = initialize_pinecone()
        vector_store = get_vector_store(pinecone_index, embeddings_model)
        setup_knowledge_base(pinecone_index, embeddings_model)
        rag_chain = create_rag_chain(llm, vector_store)
        logging.info("🎉 All services initialized successfully on startup!")
    except Exception as e:
        logging.critical(f"❌ CRITICAL: Failed to initialize services on startup: {str(e)}")
        raise e



# --- NON-STREAMING LOGIC (OPTIMIZED) ---
async def process_document_and_get_answers(request: DocumentRequest) -> dict:
    """
    Processes the document asynchronously, collects all answers in a single batch,
    and returns a single dictionary.
    """
    pdf_filename = None
    final_result = {
        "answers": []
    }

    try:
        # --- 1. Document Setup and Processing ---
        logging.info("Downloading document...")
        pdf_response = requests.get(str(request.documents), timeout=30)
        pdf_response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_response.content)
            pdf_filename = tmp_file.name

        logging.info("Processing document...")
        loader = PyPDFLoader(pdf_filename)
        docs = loader.load()
        if not docs:
            raise ValueError("Could not extract any content from the provided PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Document content was empty or could not be split into chunks.")

        logging.info(f"Creating vector store for {len(chunks)} chunks...")
        vector_store = await FAISS.afrom_documents(chunks, embeddings_model)
        doc_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4,"fetch_k": 6, "lambda_mult": 0.8  })

        # --- 2. Batch Question Answering ---
        logging.info(f"Preparing to answer {len(request.questions)} questions in a batch...")

        # Prepare a list of inputs for the batch call
        batch_inputs = [
            {"question": q, "query_doc_retriever": doc_retriever}
            for q in request.questions
        ]

        # Invoke the chain with .batch() for parallel processing
        batch_results = rag_chain.batch(batch_inputs)

        # Structure the final result
        final_result["answers"] = [
             batch_results[i]
            for i in range(len(batch_results))
        ]
        logging.info("✅ All questions answered successfully.")


    except Exception as e:
        logging.error(f"A critical error occurred during document processing: {e}")
        return {"error": f"A critical error occurred: {str(e)}"}
    finally:
        # --- 3. Cleanup ---
        if pdf_filename and os.path.exists(pdf_filename):
            os.remove(pdf_filename)

    # --- 4. Return Final Result ---
    return final_result
# --- How to Use It in Your Endpoint ---

router = APIRouter()

@router.post("/hackrx/run")
async def process_document_sync(request: DocumentRequest, api_key: str = Depends(get_api_key)):
    """
    Processes a PDF and returns a single JSON object with all answers.
    """
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Services are not initialized.")
    
    # Await the function and get the final dictionary
    result_dict = await process_document_and_get_answers(request)

    # Check if the result is an error object
    if "error" in result_dict:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result_dict["error"])

    # FastAPI will automatically convert the dictionary to a JSON response
    return result_dict

@router.post("/hackrx/run-old", response_model=DocumentResponse)
async def process_document(request: DocumentRequest, api_key: str = Depends(get_api_key)):
    """Processes a PDF document and answers questions using RAG (Blocking)."""
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Services are not initialized.")
    
    pdf_filename = None
    results = []
    try:
        if not request.questions:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Questions list cannot be empty")
        
        logging.info(f"Downloading PDF from: {request.documents}")
        pdf_response = requests.get(str(request.documents), timeout=30)
        pdf_response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_response.content)
            pdf_filename = tmp_file.name
        
        logging.info(f"Processing PDF: {pdf_filename}...")
        query_loader = PyPDFLoader(pdf_filename)
        query_docs = query_loader.load()
        if not query_docs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not extract content from PDF")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        query_chunks = splitter.split_documents(query_docs)
        if not query_chunks:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create chunks from PDF")
        
        faiss_vector_store = FAISS.from_documents(query_chunks, embeddings_model)
        query_doc_retriever = faiss_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'fetch_k': 5, "lambda_mult": 0.8})
        
        logging.info("✅ Document processed. Now answering questions...")
        
        for question in request.questions:
            logging.info(f"🔎 Processing question: {question}")
            try:
                answer = rag_chain.invoke({"question": question, "query_doc_retriever": query_doc_retriever})
                print(f"Answer for '{question}': {answer}")
                results.append(QuestionAnswer(question=question, answer=str(answer)))
                logging.info(f"✅ Answer generated for: {question}")
            except Exception as e:
                logging.error(f"❌ Error processing question '{question}': {str(e)}")
                results.append(QuestionAnswer(question=question, answer=f"Error processing question: {str(e)}"))
        
        return DocumentResponse(
            status="success",
            message=f"Successfully processed {len(results)} questions",
            results=results,
            total_questions=len(request.questions)
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error downloading PDF: {str(e)}")
    except Exception as e:
        logging.error(f"❌ Error in process_document: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")
    finally:
        if pdf_filename and os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                logging.info(f"✅ Cleaned up temporary file: {pdf_filename}")
            except Exception as e:
                logging.error(f"⚠️ Could not remove temporary file {pdf_filename}: {str(e)}")

@router.get("/status")
async def get_status():
    """Check if the document processing service is ready."""
    # ... (Your status code remains the same) ...
    pass