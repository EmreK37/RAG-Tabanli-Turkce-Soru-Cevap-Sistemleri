"""Erişim (retrieval) sistemleri: FAISS dense + BM25 sözcüksel.

Ortak arayüz: retrieve(sorgu, k) -> [(chunk_id, skor), ...]
skora göre azalan sıralı. Her embedding modeli için AYRI indeks
kurulur; vektörler karıştırılmaz.
"""
from __future__ import annotations

import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .data import Chunk
from .embeddings import DenseEmbedder

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(metin: str) -> list[str]:
    return _TOKEN_RE.findall(metin.lower())


class FaissRetriever:
    """Bir embedding modeline ait bağımsız FAISS indeksi."""

    def __init__(self, embedder: DenseEmbedder, chunklar: list[Chunk]):
        self.embedder = embedder
        self.chunklar = chunklar
        vektorler = embedder.belgeleri_kodla([c.metin for c in chunklar])
        self.indeks = faiss.IndexFlatIP(vektorler.shape[1])
        self.indeks.add(vektorler.astype(np.float32))
        self.vektorler = vektorler

    def retrieve(self, sorgu: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.embedder.sorguyu_kodla(sorgu).astype(np.float32).reshape(1, -1)
        skorlar, indisler = self.indeks.search(q, k)
        return [
            (self.chunklar[i].chunk_id, float(s))
            for s, i in zip(skorlar[0], indisler[0])
            if i >= 0
        ]


class Bm25Retriever:
    """Embedding kullanmayan sözcüksel baz çizgisi."""

    def __init__(self, chunklar: list[Chunk]):
        self.chunklar = chunklar
        self.bm25 = BM25Okapi([_tokenize(c.metin) for c in chunklar])

    def retrieve(self, sorgu: str, k: int = 10) -> list[tuple[str, float]]:
        skorlar = self.bm25.get_scores(_tokenize(sorgu))
        sirali = np.argsort(-skorlar)[:k]
        return [(self.chunklar[i].chunk_id, float(skorlar[i])) for i in sirali]
