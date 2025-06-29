from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import logging
from typing import List, Optional, Dict, Any
import hashlib
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGService:
    def __init__(self, index_name: str = "mybiodata", documents_directory: str = "documents/"):
        """Initialize RAG service with configuration"""
        self.index_name = index_name
        self.documents_directory = documents_directory
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None
        self.pc = None
        self.index = None
        
        # Initialize components
        self._setup_embeddings()
        self._setup_llm()
        self._setup_pinecone()
        self._setup_qa_chain()
    
    def _setup_embeddings(self):
        """Initialize HuggingFace embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _setup_llm(self):
        """Initialize Groq LLM"""
        try:
            api_key_groq = os.getenv("GROQ_API_KEY")
            if not api_key_groq:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.llm = ChatGroq(
                api_key=api_key_groq,
                model="allam-2-7b",  # Updated to more reliable model
                temperature=0.7,
                max_tokens=1024
            )
            logger.info("Groq LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(self.index_name)
            
            # Initialize vector store
            self.vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=api_key
            )
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _setup_qa_chain(self):
        """Initialize QA chain"""
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
                ),
                return_source_documents=True
            )
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {e}")
            raise
    
    def read_documents(self, directory: Optional[str] = None) -> List[Any]:
        """Read PDF documents from directory"""
        doc_directory = directory or self.documents_directory
        
        if not os.path.exists(doc_directory):
            raise FileNotFoundError(f"Documents directory not found: {doc_directory}")
        
        try:
            file_loader = PyPDFDirectoryLoader(doc_directory)
            documents = file_loader.load()
            logger.info(f"Loaded {len(documents)} documents from {doc_directory}")
            return documents
        except Exception as e:
            logger.error(f"Failed to read documents: {e}")
            raise
    
    def chunk_documents(self, documents: List[Any], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Any]:
        """Split documents into chunks"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            raise
    
    def generate_document_hash(self, documents: List[Any]) -> str:
        """Generate hash for document content to check if indexing is needed"""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_index_populated(self) -> bool:
        """Check if Pinecone index has data"""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count > 0
        except Exception as e:
            logger.error(f"Failed to check index stats: {e}")
            return False
    
    def index_documents(self, force_reindex: bool = False) -> bool:
        """Index documents to Pinecone"""
        try:
            # Check if reindexing is needed
            if not force_reindex and self.is_index_populated():
                logger.info("Index already populated, skipping indexing")
                return True
            
            # Read and chunk documents
            documents = self.read_documents()
            if not documents:
                logger.warning("No documents found to index")
                return False
            
            chunked_docs = self.chunk_documents(documents)
            
            # Prepare texts and embeddings
            texts = [doc.page_content for doc in chunked_docs]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            # Generate embeddings in batches to avoid memory issues
            batch_size = 50
            all_vectors = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_vectors = self.embeddings.embed_documents(batch_texts)
                all_vectors.extend(batch_vectors)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Prepare data for upsert
            data = []
            for i, (vector, chunk) in enumerate(zip(all_vectors, chunked_docs)):
                metadata = {
                    "text": texts[i],
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_id": i
                }
                data.append((str(i), vector, metadata))
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                self.index.upsert(vectors=batch_data)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
            
            # Verify indexing
            stats = self.index.describe_index_stats()
            logger.info(f"Indexing complete. Total vectors: {stats.total_vector_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def retrieve_answers(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """Retrieve answers using RAG"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            # Ensure index is populated
            if not self.is_index_populated():
                logger.info("Index is empty, attempting to index documents...")
                self.index_documents()
            
            # Get response from QA chain
            response = self.qa_chain.invoke({"query": query})
            
            result = {
                "answer": response.get("result", ""),
                "query": query
            }
            
            if include_sources and "source_documents" in response:
                sources = []
                for doc in response["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0)
                    })
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve answer for query '{query}': {e}")
            raise RuntimeError(f"Failed to retrieve answer: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Perform similarity search without QA"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            response = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            results = []
            for match in response.matches:
                results.append({
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "page": match.metadata.get("page", 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed similarity search: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            
            # Test Pinecone connection
            stats = self.index.describe_index_stats()
            
            # Test LLM (simple test)
            test_response = self.llm.invoke("Say 'OK'")
            
            return {
                "status": "healthy",
                "embeddings": "working",
                "pinecone_vectors": stats.total_vector_count,
                "llm": "working" if "OK" in str(test_response) else "warning"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global instance (initialize on first use)
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

# Convenience function for backward compatibility
def retrieve_answers(query: str) -> str:
    """Simple function to get answers (backward compatibility)"""
    service = get_rag_service()
    result = service.retrieve_answers(query, include_sources=False)
    return result["answer"]

# For testing
'''if __name__ == "__main__":
    try:
        rag = RAGService()
        
        # Index documents if needed
        rag.index_documents()
        
        # Test query
        query = "Name of rafin wife"
        result = rag.retrieve_answers(query)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        
        if 'sources' in result:
            print("\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['source']} (Page {source['page']})")
                print(f"   {source['content']}")
                
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise'''