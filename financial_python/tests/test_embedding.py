from unittest.mock import Mock, patch

from financial_rag.embedding.e5_adapter import E5EmbeddingAdapter


@patch("financial_rag.embedding.e5_adapter.requests.post")
def test_e5_adapter_uses_teI_endpoint_and_single_query_prefix(mock_post):
    response = Mock()
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    mock_post.return_value = response
    adapter = E5EmbeddingAdapter(base_url="http://e5:80", model="intfloat/multilingual-e5-large")

    vector = adapter.embed_query("query: Tổng tài sản FPT")

    assert vector == [0.1, 0.2]
    args, kwargs = mock_post.call_args
    assert args[0] == "http://e5:80/v1/embeddings"
    assert kwargs["json"]["input"] == ["query: Tổng tài sản FPT"]


@patch("financial_rag.embedding.e5_adapter.requests.post")
def test_e5_adapter_batches_passages_in_input_order(mock_post):
    response = Mock()
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.2]},
            {"index": 0, "embedding": [0.1]},
        ]
    }
    mock_post.return_value = response
    adapter = E5EmbeddingAdapter(base_url="http://e5:80")

    vectors = adapter.embed_passages(["BCTC ABB", "Tổng tài sản"])

    assert vectors == [[0.1], [0.2]]
    assert mock_post.call_args.kwargs["json"]["input"] == ["passage: BCTC ABB", "passage: Tổng tài sản"]
