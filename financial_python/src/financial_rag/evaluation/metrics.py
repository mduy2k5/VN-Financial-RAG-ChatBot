from typing import Hashable, Optional, Sequence

class RetrievalEvaluator:
    """Đánh giá chất lượng truy xuất (Recall@K, MRR)."""

    @classmethod
    def recall_at_k(
        cls,
        retrieved_pages: Sequence[Hashable],
        ground_truth_pages: Sequence[Hashable],
        k: int = 5,
    ) -> float:
        if not ground_truth_pages:
            return 1.0
        top_k_pages = set(retrieved_pages[:k])
        hits = [p for p in ground_truth_pages if p in top_k_pages]
        return len(hits) / len(ground_truth_pages)

    @classmethod
    def precision_at_k(
        cls,
        retrieved_pages: Sequence[Hashable],
        ground_truth_pages: Sequence[Hashable],
        k: int = 5,
    ) -> float:
        if k <= 0:
            return 0.0
        top_k_results = retrieved_pages[:k]
        if not top_k_results:
            return 0.0
        ground_truth = set(ground_truth_pages)
        return sum(page in ground_truth for page in top_k_results) / k

    @classmethod
    def precision_of_returned_results(
        cls,
        retrieved_pages: Sequence[Hashable],
        ground_truth_pages: Sequence[Hashable],
    ) -> float:
        if not retrieved_pages:
            return 0.0
        ground_truth = set(ground_truth_pages)
        return sum(page in ground_truth for page in retrieved_pages) / len(retrieved_pages)

    @classmethod
    def reciprocal_rank(
        cls,
        retrieved_pages: Sequence[Hashable],
        ground_truth_pages: Sequence[Hashable],
    ) -> float:
        gt_set = set(ground_truth_pages)
        for rank, p in enumerate(retrieved_pages, start=1):
            if p in gt_set:
                return 1.0 / rank
        return 0.0


class CitationEvaluator:
    """Đo mức độ citation trỏ tới đúng các trang bằng chứng đã gán nhãn."""

    @classmethod
    def page_recall(
        cls,
        cited_pages: Sequence[Optional[Hashable]],
        ground_truth_pages: Sequence[Hashable],
    ) -> float:
        if not ground_truth_pages:
            return 1.0
        cited = {page for page in cited_pages if page is not None}
        return len(cited & set(ground_truth_pages)) / len(set(ground_truth_pages))

    @classmethod
    def page_precision(
        cls,
        cited_pages: Sequence[Optional[Hashable]],
        ground_truth_pages: Sequence[Hashable],
    ) -> float:
        cited = {page for page in cited_pages if page is not None}
        if not cited:
            return 0.0
        return len(cited & set(ground_truth_pages)) / len(cited)

class NumericEvaluator:
    """Đánh giá độ khớp số liệu (Exact Match, Normalized Match)."""

    @classmethod
    def exact_match(cls, predicted: str, target: str) -> bool:
        return predicted.strip().replace(".", "").replace(",", "") == target.strip().replace(".", "").replace(",", "")
