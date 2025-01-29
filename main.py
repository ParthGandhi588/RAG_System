from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List
from langflow.schema.message import Message
from chat import RetrievalChatSystem
import uvicorn
from loguru import logger
import os
from dotenv import load_dotenv

# Import the necessary classes from load_data_flow.py
from load_data_flow import DataIngestionConfig, DataIngestionPipeline

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Retrieval Chat API",
    description="API for chatting with documents stored in ChromaDB",
    version="1.0.0"
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    sender_name: Optional[str] = "User"

class ChatResponse(BaseModel):
    message: str
    sender: str
    sender_name: str


# Global chat system instance
chat_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize the chat system on startup"""
    global chat_system
    try:
        chat_system = RetrievalChatSystem(
            collection_name="Random_Data",  # Replace with your collection name
            persist_directory=os.getenv("PERSISTANT_DIRECTORY"),  # Replace with your ChromaDB path
            api_key = os.getenv("GROQ_API_KEY"),  # Groq API key
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Chat system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat system: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {"status": "running", "message": "Welcome to Retrieval Chat API"}



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not chat_system:
        raise HTTPException(status_code=503, detail="Chat system not initialized")
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    if not chat_system:
        raise HTTPException(status_code=500, detail="Chat system not initialized")
    
    try:
        # Create input message
        input_message = Message(
            text=request.message,
            sender="user",
            sender_name=request.sender_name
        )
        
        # Get response from chat system
        response = chat_system.process_chat_input(input_message)
        
        return ChatResponse(
            message=response.text,
            sender=response.sender,
            sender_name=response.sender_name
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload a PDF file and process it into ChromaDB"""
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Configure the data ingestion pipeline
        config = DataIngestionConfig(
            file_path=file_path,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_device="cpu",
            persist_directory=os.getenv("PERSISTANT_DIRECTORY")
        )
        
        # Process the file using the data ingestion pipeline
        pipeline = DataIngestionPipeline(config)
        vector_store = pipeline.process()
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return {"status": "success", "message": "PDF processed and stored in ChromaDB"}
    except Exception as e:
        logger.error(f"Error processing PDF file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)