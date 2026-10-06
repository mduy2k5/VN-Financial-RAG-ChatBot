from typing import List, Dict, Tuple, Optional

def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Thuật toán Reciprocal Rank Fusion (RRF) kết hợp danh sách xếp hạng từ nhiều bộ tìm kiếm (BM25, Vector, v.v.).
    RRF Score(d) = sum_{r in rankings} (1 / (k + rank(r, d)))
    """
    scores: Dict[str, float] = {}

    for index, ranking in enumerate(rankings):
        weight = weights[index] if weights and index < len(weights) else 1.0
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (weight / (k + rank))

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
