"""
RAG Pipeline Compiler — compiles heterogeneous media sources (HTML, PDF,
Markdown, JSON) into a unified retrieval-augmented generation pipeline
with chunking, embedding, and semantic search.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional
import hashlib
import re


# --- Source types ---

class SourceType:
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    PLAINTEXT = "plaintext"


@dataclass
class Document:
    source_id: str
    source_type: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


# --- Parsers ---

class SourceParser:
    """Registry of source-type parsers."""

    def __init__(self):
        self._parsers: dict[str, Callable[[str], str]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self._parsers[SourceType.PLAINTEXT] = lambda x: x
        self._parsers[SourceType.MARKDOWN] = self._strip_markdown
        self._parsers[SourceType.JSON] = self._flatten_json

    def register(self, source_type: str, parser: Callable[[str], str]):
        self._parsers[source_type] = parser

    def parse(self, source_type: str, raw: str) -> str:
        parser = self._parsers.get(source_type, lambda x: x)
        return parser(raw)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"#{1,6}\s", "", text)
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        return text.strip()

    @staticmethod
    def _flatten_json(text: str) -> str:
        import json
        try:
            data = json.loads(text)
            return " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
        except Exception:
            return text


# --- Chunker ---

class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{doc.source_id}_{idx}",
                    doc_id=doc.source_id,
                    text=chunk_text,
                    metadata={**doc.metadata, "chunk_index": idx},
                ))
                idx += 1
            start = end - self.overlap if end < len(text) else len(text)
        return chunks


# --- Pipeline ---

class RAGPipelineCompiler:
    """
    Compiles documents from multiple source types into a searchable
    RAG index. Supports pluggable parsers and embedding functions.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ):
        self.parser = SourceParser()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        self.embed_fn = embed_fn
        self._index: list[Chunk] = []

    def ingest(self, raw: str, source_type: str, metadata: Optional[dict] = None) -> list[Chunk]:
        parsed = self.parser.parse(source_type, raw)
        doc = Document(
            source_id=hashlib.sha256(raw.encode()).hexdigest()[:12],
            source_type=source_type,
            content=parsed,
            metadata=metadata or {},
        )
        chunks = self.chunker.chunk(doc)
        if self.embed_fn:
            embeddings = self.embed_fn([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        self._index.extend(chunks)
        return chunks

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        import math
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x ** 2 for x in a))
            norm_b = math.sqrt(sum(x ** 2 for x in b))
            return dot / (norm_a * norm_b + 1e-8)

        scored = [
            (c, cosine(query_embedding, c.embedding))
            for c in self._index if c.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def stats(self) -> dict:
        return {
            "total_chunks": len(self._index),
            "embedded_chunks": sum(1 for c in self._index if c.embedding),
            "unique_docs": len({c.doc_id for c in self._index}),
        }
