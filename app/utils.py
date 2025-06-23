def generate_session_title(content: str) -> str:
    # Simple title generation based on first 30 characters
    return content[:30] + "..." if len(content) > 30 else content