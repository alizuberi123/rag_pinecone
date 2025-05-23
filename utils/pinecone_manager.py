import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

class PineconeManager:
    """
    Manages Pinecone vector store operations for the RAG system.
    """
    
    def __init__(self, api_key: str = None, environment: str = None):
        """
        Initialize Pinecone manager.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENV env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENV")
        
        if not self.api_key or not self.environment:
            raise ValueError("Pinecone API key and environment must be provided")
            
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def create_index(self, index_name: str, dimension: int = 384) -> None:
        """
        Create a new Pinecone index.
        
        Args:
            index_name: Name of the index to create
            dimension: Dimension of the vectors (default: 384 for all-MiniLM-L6-v2)
        """
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
    def get_vector_store(self, index_name: str) -> PineconeVectorStore:
        """
        Get a Pinecone vector store instance.
        
        Args:
            index_name: Name of the Pinecone index
            
        Returns:
            PineconeVectorStore instance
        """
        return PineconeVectorStore(
            index_name=index_name,
            embedding=self.embedding_model,
            text_key="text"
        )
        
    def add_documents(self, index_name: str, documents: List[Document]) -> None:
        """
        Add documents to the Pinecone index.
        
        Args:
            index_name: Name of the Pinecone index
            documents: List of documents to add
        """
        vector_store = self.get_vector_store(index_name)
        vector_store.add_documents(documents)
        
    def similarity_search(self, index_name: str, query: str, k: int = 10, 
                         filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform similarity search on the Pinecone index.
        
        Args:
            index_name: Name of the Pinecone index
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        vector_store = self.get_vector_store(index_name)
        return vector_store.similarity_search(query, k=k, filter=filter)
        
    def delete_index(self, index_name: str) -> None:
        """
        Delete a Pinecone index.
        
        Args:
            index_name: Name of the index to delete
        """
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
            
    def list_indexes(self) -> List[str]:
        """
        List all Pinecone indexes.
        
        Returns:
            List of index names
        """
        return list(self.pc.list_indexes().names()) 