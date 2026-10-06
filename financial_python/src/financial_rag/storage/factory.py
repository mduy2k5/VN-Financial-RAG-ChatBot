"""Construct persistence without importing or initializing model/API runtimes."""
import os

from .sqlite_store import SQLiteFinancialStore


def create_document_store():
    backend = os.getenv("FINANCIAL_STORAGE_BACKEND", "sqlite").strip().casefold()
    if backend in {"mongo_qdrant", "mongodb_qdrant"}:
        from .mongo_store import MongoFinancialStore
        return MongoFinancialStore(os.getenv("MONGODB_URI", ""),
                                   os.getenv("FINANCIAL_MONGO_DATABASE", "financial_rag"))
    if backend != "sqlite":
        raise ValueError(f"Unsupported financial storage backend: {backend}")
    path = os.getenv("FINANCIAL_DB_PATH", "").strip()
    return SQLiteFinancialStore(path) if path else None
