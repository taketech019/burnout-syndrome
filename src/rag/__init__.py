def chat(query: str) -> str:
    from .chatbot import chat as _chat
    return _chat(query)


def search(query: str, collection_name: str = "clinical_references", n_results: int = 3):
    from .retriever import search as _search
    return _search(query, collection_name, n_results)


__all__ = ["chat", "search"]
