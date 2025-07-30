"""
VectorDB Tool for Research Assistant

This module provides a simple, provider-agnostic wrapper around a **persistent**
Chroma vector database.  It is inspired by the design of `ModelBuilder` so that
changing the embedding backend (OpenAI, HuggingFace, Ollama, etc.) requires only
one line of configuration.

Key Features
------------
1.  **Provider-agnostic embeddings** – easily switch between OpenAI or any other
    embedding implementation that exposes a `embed_documents` / `embed_query`
    callable.
2.  **Persistence** – vectors are stored on disk (via `persist_directory`) so
    that subsequent runs reuse the same collection without re-embedding.
3.  **Simple API** – `add_texts`, `similarity_search`, and `as_retriever` mirror
    common LangChain patterns so the tool can be dropped into an agent/tooling
    stack immediately.

Dependencies
------------
* `chromadb`
* `tqdm` (optional, for progress bars when adding many texts)
* Either `openai`, `langchain-openai` **or** any embedding implementation you
  inject yourself, this is expandable to other providers.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional – fall back to no progress bar

    def tqdm(x, **kwargs):  # type: ignore
        return x


logger = logging.getLogger(__name__)


# Embedding provider abstraction
class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider that mimics the LangChain interface."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (strings)."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider that uses OpenAI text-embedding-3-small by default."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found – cannot initialise OpenAI embeddings"
            )

        # We *avoid* importing openai unless necessary so that projects without the package can still run when another provider is chosen.
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def _call(self, inputs: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=inputs)
        # The API returns embeddings in the same order as inputs
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, query: str) -> List[float]:
        return self._call([query])[0]


# Showcasing extensibility of the vector db tool
class DummyEmbeddingProvider(BaseEmbeddingProvider):
    """A deterministic fake embedding provider – useful for tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[hash(text) % 1_000_000 / 1_000_000.0] for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return [hash(query) % 1_000_000 / 1_000_000.0]


EMBEDDING_PROVIDERS: Dict[str, Any] = {
    "openai": OpenAIEmbeddingProvider,
    "dummy": DummyEmbeddingProvider,  # always available fallback
}


class VectorDBTool:
    """Wrapper around a persistent ChromaDB collection with pluggable embeddings."""

    def __init__(
        self,
        persist_directory: str = ".vectordb",
        collection_name: str = "research_assistant",
        embedding_provider: str = "openai",
        embedding_provider_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        provider_cls = EMBEDDING_PROVIDERS.get(embedding_provider)
        if provider_cls is None:
            raise ValueError(
                f"Unknown embedding provider '{embedding_provider}'. Available: {list(EMBEDDING_PROVIDERS.keys())}"
            )
        embedding_provider_kwargs = embedding_provider_kwargs or {}
        self.embedding_provider: BaseEmbeddingProvider = provider_cls(
            **embedding_provider_kwargs
        )

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        # Use Chroma's optional builtin embedding wrapper so that similarity
        # search happens server-side if possible. Otherwise we embed manually.
        if embedding_provider == "openai":
            chroma_embed_fn = OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            chroma_embed_fn = None  # we will supply embeddings manually

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, embedding_function=chroma_embed_fn
        )
        logger.info(
            "VectorDBTool initialised (provider=%s, collection='%s', dir='%s')",
            embedding_provider,
            self.collection_name,
            self.persist_directory,
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> None:
        """Add documents to the collection (embeds them if server doesn't)."""
        metadatas = metadatas or [{} for _ in texts]
        if ids is None:
            ids = [f"doc_{i}_{abs(hash(t)) % 1_000_000}" for i, t in enumerate(texts)]

        if self._collection._embedding_function is None:  # type: ignore[attr-defined]
            logger.info(
                "Embedding %d documents locally (batch_size=%d)…",
                len(texts),
                batch_size,
            )
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                batch_embeds = self.embedding_provider.embed_documents(batch)
                self._collection.add(
                    ids=ids[i : i + batch_size],
                    documents=batch,
                    embeddings=batch_embeds,
                    metadatas=metadatas[i : i + batch_size],
                )
        else:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info(
            "Added %d documents to Chroma collection '%s'",
            len(texts),
            self.collection_name,
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k most similar documents to *query*."""
        if self._collection._embedding_function is None:  # type: ignore[attr-defined]
            query_embed = self.embedding_provider.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embed], n_results=k
            )
        else:
            results = self._collection.query(query_texts=[query], n_results=k)

        # Format results nicely
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    # making it easy to use langchain if desired
    def as_retriever(self, k: int = 5):  # type: ignore[return-value]
        """Return a LangChain-style retriever object (if LangChain installed)."""
        try:
            from langchain.retrievers import BaseRetriever  # type: ignore
            from langchain.schema import Document  # type: ignore

            class _VectorDBRetriever(BaseRetriever):
                def __init__(self, parent: "VectorDBTool", k: int = 5):  # noqa: D401
                    """Thin wrapper so we can plug directly into LC chains."""
                    self.parent = parent
                    self.k = k

                def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
                    res = self.parent.similarity_search(query, self.k)
                    return [
                        Document(page_content=r["document"], metadata=r["metadata"])
                        for r in res
                    ]

                async def _aget_relevant_documents(self, query: str):  # type: ignore[override]
                    # Fallback to sync for simplicity
                    return self._get_relevant_documents(query)

            return _VectorDBRetriever(self, k)
        except ImportError:
            raise ImportError(
                "langchain not installed – cannot create retriever. Run `pip install langchain`."
            )
