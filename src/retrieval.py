"""Erişim (retrieval) sistemleri: FAISS dense + BM25 sözcüksel.

Ortak arayüz: retrieve(sorgu, k) -> [(chunk_id, skor), ...]
skora göre azalan sıralı. Her embedding modeli için AYRI indeks
kurulur; vektörler karıştırılmaz.
"""
from __future__ import annotations

import re
from typing import Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .data import Chunk
from .embeddings import DenseEmbedder

VARSAYILAN_TOP_K = 10
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class Retriever(Protocol):
    """Tüm erişim sistemlerinin uyguladığı ortak protokol."""

    vektor_boyutu: int

    def retrieve(self, sorgu: str, k: int = VARSAYILAN_TOP_K) -> list[tuple[str, float]]:
        """Sorguya en yakın `k` parçayı `(chunk_id, skor)` çiftleri olarak döndürür."""
        ...

    def indeks_boyutu_kb(self) -> float:
        """İndeksin bellekte kapladığı yaklaşık boyutu kilobayt (KB) cinsinden döndürür."""
        ...



def _tokenize(metin: str) -> list[str]:
    """Metni küçük harfe çevirip Unicode sözcük sınırlarına göre böler."""
    return _TOKEN_RE.findall(metin.lower())


def _chunk_listesini_dogrula(chunklar: list[Chunk]) -> None:
    """İndeks kurulumu öncesinde chunk listesinin boş olmadığını doğrular."""
    if not chunklar:
        raise ValueError("İndeks kurmak için en az bir Chunk gereklidir.")


def _top_k_dogrula(k: int) -> None:
    """Erişimde istenen `k` değerinin pozitif olduğunu doğrular."""
    if k <= 0:
        raise ValueError(f"Top-k değeri pozitif olmalıdır (verilen: {k}).")


class FaissRetriever:
    """Bir embedding modeline ait bağımsız FAISS indeksi."""

    def __init__(self, embedder: DenseEmbedder, chunklar: list[Chunk]) -> None:
        _chunk_listesini_dogrula(chunklar)
        self.embedder = embedder
        self.chunklar = chunklar
        vektorler = embedder.belgeleri_kodla([c.metin for c in chunklar]).astype(np.float32)
        self.vektor_boyutu = int(vektorler.shape[1])
        self.indeks = faiss.IndexFlatIP(self.vektor_boyutu)
        self.indeks.add(vektorler)
        self.vektorler = vektorler

    def indeks_boyutu_kb(self) -> float:
        """FAISS float32 vektör matrisinin bellekteki boyutunu (KB) döndürür."""
        return round(float(self.vektorler.nbytes) / 1024.0, 2)

    def retrieve(self, sorgu: str, k: int = VARSAYILAN_TOP_K) -> list[tuple[str, float]]:
        """Sorgu vektörüyle iç çarpım (kosinüs benzerliği) araması yapar."""
        _top_k_dogrula(k)
        etkin_k = min(k, len(self.chunklar))
        q = self.embedder.sorguyu_kodla(sorgu).astype(np.float32).reshape(1, -1)
        skorlar, indisler = self.indeks.search(q, etkin_k)
        return [
            (self.chunklar[i].chunk_id, float(s))
            for s, i in zip(skorlar[0], indisler[0])
            if i >= 0
        ]


class Bm25Retriever:
    """Embedding kullanmayan sözcüksel baz çizgisi."""

    def __init__(self, chunklar: list[Chunk]) -> None:
        _chunk_listesini_dogrula(chunklar)
        self.chunklar = chunklar
        tokenli_korpus = [_tokenize(c.metin) for c in chunklar]
        self.bm25 = BM25Okapi(tokenli_korpus)
        self.vektor_boyutu = len(self.bm25.idf)
        self._yaklasik_bayt = (
            len(self.bm25.idf) * 16
            + sum(len(df) * 16 for df in self.bm25.doc_freqs)
            + len(self.bm25.doc_len) * 8
        )

    def indeks_boyutu_kb(self) -> float:
        """BM25 ters indeks yapısının bellekteki yaklaşık boyutunu (KB) döndürür."""
        return round(float(self._yaklasik_bayt) / 1024.0, 2)

    def retrieve(self, sorgu: str, k: int = VARSAYILAN_TOP_K) -> list[tuple[str, float]]:
        """BM25Okapi skorlarına göre en yüksek skorlu `k` parçayı döndürür."""
        _top_k_dogrula(k)
        etkin_k = min(k, len(self.chunklar))
        skorlar = self.bm25.get_scores(_tokenize(sorgu))
        sirali = np.argsort(-skorlar)[:etkin_k]
        return [(self.chunklar[i].chunk_id, float(skorlar[i])) for i in sirali]


