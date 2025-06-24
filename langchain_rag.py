
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone


from dotenv import load_dotenv
load_dotenv()

import os

## Lets Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')
'''len(doc)
first_doc = doc[0]  
print(first_doc.page_content)  
print(first_doc.metadata) 
'''

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(doc)
# print(f"Number of chunks: {len(chunked_docs)}")



from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


# Embed a single query
vectors=embeddings.embed_query("How are you?")
# len(vectors)


from pinecone import Pinecone
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=api_key)
index_name = "mybiodata"
index = pc.Index(index_name)


# Embed chunked documents
texts = [doc.page_content for doc in chunked_docs]
vectors = embeddings.embed_documents(texts)


# Prepare data for upsert
data = [
    (
        str(i),                    # Unique ID for each vector
        vectors[i],                # Embedding vector
        {"text": texts[i], **chunked_docs[i].metadata}  # Metadata (text and original metadata)
    )
    for i in range(len(vectors))
]

# Upsert to Pinecone
index.upsert(vectors=data)


# Verify upsert
stats = index.describe_index_stats()
# print(f"Index stats: {stats}")


def pinecone_similarity_search(query, k=2):
    query_embedding = embeddings.embed_query(query)
    response = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return response


from langchain.chains.question_answering import load_qa_chain


api_key_groq = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=api_key_groq, 
    model="allam-2-7b",
    temperature=0.7
)
chain=load_qa_chain(llm,chain_type="stuff")


# Initialize Pinecone vector store for langchain
from langchain_pinecone import PineconeVectorStore
api_key = os.getenv("PINECONE_API_KEY")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)


# Set up RetrievalQA chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)


# Search answers
def retrieve_answers(query):
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    try:
        response = qa_chain.run(query)
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve answer: {str(e)}")

# # Test
# query = "Name of rafin wife"
# answer = retrieve_answers(query)
# print(f"Answer: {answer}")