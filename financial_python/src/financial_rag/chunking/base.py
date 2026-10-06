from abc import ABC, abstractmethod
from typing import List
from ..schemas import FinancialChunk

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[FinancialChunk]:
        """Chia nhỏ văn bản thành các FinancialChunk."""
        pass
