"""Retrieval metrikleri ve istatistik.

Metrikler tek ilgili kanıt (binary relevance) varsayımıyla hesaplanır:
sorunun `kanit_chunk_id` alanı. Çok kanıtlı sorular için qrels sözlüğü
genişletilebilir.

İstatistik: sorgu bazında eşleştirilmiş bootstrap güven aralığı
(aynı sorular üzerinde iki sistemin farkı).
"""
from __future__ import annotations

import numpy as np


def recall_at_k(getirilen: list[str], ilgili: str, k: int) -> float:
    return 1.0 if ilgili in getirilen[:k] else 0.0


def hit_at_k(getirilen: list[str], ilgili: str, k: int) -> float:
    return recall_at_k(getirilen, ilgili, k)


def mrr_at_k(getirilen: list[str], ilgili: str, k: int) -> float:
    for sira, chunk_id in enumerate(getirilen[:k], start=1):
        if chunk_id == ilgili:
            return 1.0 / sira
    return 0.0


def ndcg_at_k(getirilen: list[str], ilgili: str, k: int) -> float:
    # binary kazanç; ideal DCG = 1 (tek ilgili ilk sırada)
    for sira, chunk_id in enumerate(getirilen[:k], start=1):
        if chunk_id == ilgili:
            return 1.0 / np.log2(sira + 1)
    return 0.0


_METRIKLER = {
    "recall": recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def soru_skorlari(
    sonuclar: list[list[str]], ilgililer: list[str], metrik: str, k: int
) -> np.ndarray:
    """Her soru için metrik skoru (bootstrap girdisi)."""
    fn = _METRIKLER[metrik]
    return np.array(
        [fn(getirilen, ilgili, k) for getirilen, ilgili in zip(sonuclar, ilgililer)],
        dtype=float,
    )


def ortalama(skorlar: np.ndarray) -> float:
    return float(np.mean(skorlar)) if len(skorlar) else 0.0


def eslestirilmis_bootstrap_fark(
    skorlar_a: np.ndarray,
    skorlar_b: np.ndarray,
    tekrar: int = 1000,
    tohum: int = 42,
    guven: float = 0.95,
) -> dict:
    """A - B farkının eşleştirilmiş bootstrap güven aralığı.

    Dönen sözlük: {fark, alt, ust, anlamli}
    """
    rng = np.random.default_rng(tohum)
    farklar = skorlar_a - skorlar_b
    n = len(farklar)
    ornekler = np.array(
        [np.mean(rng.choice(farklar, size=n, replace=True)) for _ in range(tekrar)]
    )
    alfa = (1 - guven) / 2
    alt, ust = np.quantile(ornekler, [alfa, 1 - alfa])
    return {
        "fark": float(np.mean(farklar)),
        "alt": float(alt),
        "ust": float(ust),
        "anlamli": bool(alt > 0 or ust < 0),
    }
