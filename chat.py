from typing import List, Optional
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from loguru import logger
from langflow.schema.message import Message
from langflow.schema import Data

class RetrievalChatSystem:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str = None,
        # model_name: str = "mistral-small-latest", # Mistral's version of Mixtral
        model_name: str = "DeepSeek-R1-Distill-Llama-70B", # Groq's version of DeepSeek
        # model_name: str = "mixtral-8x7b-32768",  # Groq's version of Mixtral
        chroma_settings: Optional[Settings] = None
    ):
        # Initialize embeddings with the same model used for creating the collection
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"device": "cpu"}
        )
        
        # Connect to existing ChromaDB collection
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
            client_settings=chroma_settings
        )
        
        # # Initialize Mistral LLM
        # self.llm = ChatMistralAI(
        #     model_name=model_name,
        #     api_key=api_key,
        #     temperature=0.4
        # )

        # # Initialize Groq LLM
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.7
        )
        
        logger.info(f"Connected to existing collection '{collection_name}' at {persist_directory}")
        
    def search(
        self,
        query: str,
        search_type: str = "Similarity", ## default value
        n_results: int = 5 ## default value
    ) -> List[Data]:
        """Search the existing vector store"""
        try:
            if search_type == "MMR":
                docs = self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=n_results
                )
            else:
                docs = self.vectorstore.similarity_search(
                    query,
                    k=n_results
                )
            
            return [Data(text=doc.page_content) for doc in docs]
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def process_chat_input(self, input_message: Message) -> Message:
        """Process chat input and return response with retrieved context"""
        # Search for relevant documents from existing collection
        retrieved_docs = self.search(
            input_message.text,
            n_results=5
        )
        
        if not retrieved_docs:
            # Handle case where no relevant documents are found
            return Message(
                text="I couldn't find any relevant information in the database to answer your question. Could you please rephrase or ask something else?",
                sender="AI",
                sender_name="LOM-AI Assistant",
                properties={
                    "background_color": "#f0f0f0",
                    "icon": "bot"
                }
            )
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        
        # Create prompt with context
        prompt = f"""Based on the following context, please provide a relevant response in Higlish language 
        example:"Maine usse party mein milne ka plan banaya hai." and do not add anything new out of the conetext. 
        If the question in not related to the context, just respond with 'I couldn't find any relevant information in the database to answer your question. 
        Could you please rephrase or ask something else?'.:

Context:
{context}

Question:
{input_message.text}

Response:"""

        # Generate response using LLM
        response = self.llm.invoke(prompt)
        
        # Create output message
        output_message = Message(
            text=response.content,
            sender="AI",
            sender_name="LOM-AI Assistant",
            properties={
                "background_color": "#f0f0f0",
                "icon": "bot"
            }
        )
        
        return output_message