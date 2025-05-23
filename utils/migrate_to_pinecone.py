import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.pinecone_manager import PineconeManager

def migrate_chroma_to_pinecone(
    chroma_db_path: str,
    pinecone_index_name: str,
    pinecone_api_key: str = None,
    pinecone_env: str = None,
    batch_size: int = 100
) -> None:
    """
    Migrate data from ChromaDB to Pinecone.
    
    Args:
        chroma_db_path: Path to the ChromaDB directory
        pinecone_index_name: Name of the Pinecone index to create/use
        pinecone_api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        pinecone_env: Pinecone environment (defaults to PINECONE_ENV env var)
        batch_size: Number of documents to process in each batch
    """
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load ChromaDB
    print(f"Loading ChromaDB from {chroma_db_path}...")
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embedding_model
    )
    
    # Get all documents from ChromaDB
    print("Retrieving documents from ChromaDB...")
    chroma_docs = chroma_db.get()
    
    if not chroma_docs or 'documents' not in chroma_docs:
        print("No documents found in ChromaDB.")
        return
        
    # Initialize Pinecone
    print("Initializing Pinecone...")
    pinecone_manager = PineconeManager(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )
    
    # Create Pinecone index if it doesn't exist (384 dims, dotproduct, AWS us-east-1)
    print(f"Creating Pinecone index '{pinecone_index_name}'...")
    pinecone_manager.create_index(pinecone_index_name, dimension=384)
    
    # Convert ChromaDB documents to Langchain documents
    print("Converting documents...")
    documents = []
    for text, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas']):
        documents.append({
            'page_content': text,
            'metadata': metadata
        })
    
    # Upload documents to Pinecone in batches
    print(f"Uploading {len(documents)} documents to Pinecone...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}...")
        pinecone_manager.add_documents(pinecone_index_name, batch)
    
    print("Migration completed successfully!")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate data from ChromaDB to Pinecone")
    parser.add_argument("--chroma-db-path", required=True, help="Path to ChromaDB directory")
    parser.add_argument("--pinecone-index", required=True, help="Name of Pinecone index")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key")
    parser.add_argument("--pinecone-env", help="Pinecone environment")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for document upload")
    
    args = parser.parse_args()
    
    migrate_chroma_to_pinecone(
        chroma_db_path=args.chroma_db_path,
        pinecone_index_name=args.pinecone_index,
        pinecone_api_key=args.pinecone_api_key,
        pinecone_env=args.pinecone_env,
        batch_size=args.batch_size
    ) 