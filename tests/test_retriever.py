from rag.retriever import search_similar


def test_search_similar_empty_query_returns_empty():
    assert search_similar("", top_k=3) == []

