from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import runllm_router 
from routers.runllm_router import initialize_services_sync
import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for processing PDF documents and answering questions using RAG with Google Gemini and Pinecone",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(runllm_router.router, prefix="/api/v1", tags=["documents"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logging.info("🚀 Starting Document Q&A API...")
    try:
        initialize_services_sync()
        logging.info("✅ All services initialized successfully on startup")
    except Exception as e:
        logging.error(f"❌ Failed to initialize services on startup: {str(e)}")
        logging.info("💡 You can manually initialize services using POST /api/v1/initialize-services")

@app.get("/")
async def root():
    return {
        "message": "Document Q&A API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "status_endpoint": "/api/v1/status"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
