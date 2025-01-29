import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from loguru import logger

from data_ingestion.file_upload import FileComponent
from data_ingestion.split_text import SplitTextComponent

class DataIngestionConfig(BaseModel):
    # File Configuration
    file_path: str
    silent_errors: bool = False
    use_multithreading: bool = False
    concurrency_multithreading: int = 4
    
    # Text Splitting Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n"
    
    # HuggingFace Configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encode_device: str = "cpu"  # or "cuda" for GPU
    cache_folder: Optional[str] = None
    
    # ChromaDB Configuration
    collection_name: str = "Random_Data" # Can change collection name from here
    persist_directory: Optional[str] = None
    chroma_server_host: Optional[str] = None
    chroma_server_http_port: Optional[int] = None
    allow_duplicates: bool = False

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def _validate_embeddings(self):
        """Validate HuggingFace embeddings by attempting a simple embedding"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                cache_folder=self.config.cache_folder,
                encode_kwargs={"device": self.config.encode_device}
            )
            
            # Test with a simple string
            test_embedding = embeddings.embed_query("test")
            if not test_embedding:
                raise ValueError("Failed to generate test embedding")
            logger.info("HuggingFace embeddings validated successfully")
        except Exception as e:
            logger.error(f"HuggingFace embeddings validation failed: {str(e)}")
            raise ValueError(f"Error initializing embeddings: {str(e)}")

    def _batch_documents(self, documents: List[any], batch_size: int = 10):
        """Split documents into smaller batches"""
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]

    def process(self):
        """Run the complete data ingestion pipeline"""
        try:
            # Validate embeddings first
            self._validate_embeddings()

            # 1. Load files
            logger.info("Starting file loading process")
            file_loader = FileComponent(
                path=self.config.file_path,
                silent_errors=self.config.silent_errors,
                use_multithreading=self.config.use_multithreading,
                concurrency_multithreading=self.config.concurrency_multithreading
            )
            loaded_data = file_loader.load_file()
            logger.info(f"Successfully loaded files: {len(loaded_data) if isinstance(loaded_data, list) else 1} documents")

            # 2. Split text
            logger.info("Starting text splitting process")
            text_splitter = SplitTextComponent(
                data_inputs=[loaded_data] if not isinstance(loaded_data, list) else loaded_data,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.separator
            )
            split_data = text_splitter.split_text()
            logger.info(f"Split text into {len(split_data)} chunks")

            # 3. Initialize embeddings
            logger.info("Initializing embeddings model")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                cache_folder=self.config.cache_folder,
                encode_kwargs={"device": self.config.encode_device}
            )

            # 4. Initialize and populate ChromaDB
            logger.info("Initializing ChromaDB")
            chroma_settings = None
            if self.config.chroma_server_host:
                chroma_settings = Settings(
                    chroma_server_host=self.config.chroma_server_host,
                    chroma_server_http_port=self.config.chroma_server_http_port
                )

            persist_directory = str(Path(self.config.persist_directory).resolve()) if self.config.persist_directory else None

            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=self.config.collection_name,
                client_settings=chroma_settings
            )

            # Convert split data to documents
            documents = [data.to_lc_document() for data in split_data]
            
            # Process documents in batches
            logger.info("Adding documents to ChromaDB in batches")
            for batch in self._batch_documents(documents):
                try:
                    vector_store.add_documents(batch)
                    logger.info(f"Successfully added batch of {len(batch)} documents")
                except Exception as e:
                    logger.error(f"Error adding batch to ChromaDB: {str(e)}")
                    raise

            logger.info("Successfully completed data ingestion pipeline")
            return vector_store

        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {str(e)}")
            raise