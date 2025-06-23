from langchain_rag import retrieve_answers

def process_query(query: str) -> str:
    try:
        response = retrieve_answers(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"