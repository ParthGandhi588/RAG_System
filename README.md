# Retrieval Chat System with Document Processing

A robust document processing and retrieval chat system that allows users to upload documents, process them into embeddings, and interact with the content through a chat interface. The system uses advanced **DeepSeek model** to provide contextually relevant responses based on the uploaded documents.

## 🚀 Features

- Document upload and processing support (PDF and text files)
- Efficient text chunking and embedding generation
- Vector storage using ChromaDB
- Real-time chat interface with context-aware responses
- Support for Hinglish language responses
- REST API endpoints for easy integration
- Parallel processing for large document sets
- Robust error handling and logging

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications powered by language models
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **Groq**: High-performance language model integration
- **HuggingFace Transformers**: For text embeddings (sentence-transformers)
- **Pydantic**: Data validation using Python type annotations
- **Loguru**: Advanced Python logging
- **Uvicorn**: Lightning-fast ASGI server

## 📋 Prerequisites

- Python 3.8+ (Python 3.10 recommended)
- Groq API key
- Sufficient storage space for the vector database
- RAM: 8GB minimum (16GB recommended)

## ⚙️ Installation

1. Clone the repository:
```cmd
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```cmd
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```cmd
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```cmd
GROQ_API_KEY=your_groq_api_key_here
PERSISTANT_DIRECTORY=path to your persistant directory
```

## 🚦 Running the Application

1. First, ensure your environment variables are set up correctly in the `.env` file.

2. Start the FastAPI server:
```cmd
python main.py
```

The server will start running at `http://localhost:8000`

## 🔄 API Endpoints

- `GET /`: Root endpoint to verify API status
- `GET /health`: Health check endpoint
- `POST /chat`: Process chat messages and get responses
- `POST /upload`: Upload and process documents

## 📁 Project Structure

```
├── main.py                 # FastAPI application entry point
├── chat.py                 # Chat system implementation
├── load_data_flow.py       # Data ingestion pipeline
├── data_ingestion/
│   ├── file_upload.py      # File handling component
│   └── split_text.py       # Text splitting component
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables
```

## 💡 Use Cases

This system is particularly useful for:
- Creating chatbots that can answer questions based on specific documents
- Building document-based Q&A systems
- Implementing context-aware customer support systems
- Processing and analyzing large document collections
- Creating multilingual chat interfaces (supports Hinglish)

## ⚠️ Important Notes

1. Make sure to set up proper environment variables before running the application
2. The system requires sufficient storage space for the vector database
3. For large documents, consider adjusting the chunk size and overlap in the configuration
4. Monitor the ChromaDB storage directory size as it grows with document additions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details

## 📧 Contact

For any queries or support, please create an issue in the repository.