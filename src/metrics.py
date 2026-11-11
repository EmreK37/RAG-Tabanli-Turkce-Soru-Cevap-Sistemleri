"""Retrieval metrikleri ve istatistik.

Metrikler tek ilgili kanıt (binary relevance) varsayımıyla hesaplanır:
sorunun `kanit_chunk_id` alanı. Çok kanıtlı sorular için qrels sözlüğü
genişletilebilir.

İstatistik: sorgu bazında eşleştirilmiş bootstrap güven aralığı
(aynı sorular üzerinde iki sistemin farkı).
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

VARSAYILAN_BOOTSTRAP_TEKRAR = 1000
VARSAYILAN_TOHUM = 42
VARSAYILAN_GUVEN_DUZEYI = 0.95

_KELIME_RE = re.compile(r"\w+", re.UNICODE)
_TR_STOPWORDS = frozenset(
    {
        "ve",
        "veya",
        "ile",
        "için",
        "bir",
        "bu",
        "şu",
        "o",
        "de",
        "da",
        "ki",
        "mi",
        "mı",
        "mu",
        "mü",
        "olarak",
        "göre",
        "kadar",
        "sonra",
        "önce",
        "ise",
        "ancak",
        "fakat",
        "her",
        "tüm",
        "bütün",
        "ilgili",
        "tarafından",
        "olan",
        "olup",
        "olur",
        "olması",
        "halinde",
        "durumunda",
        "üzerine",
        "ait",
        "dair",
        "üzere",
        "en",
        "daha",
        "çok",
        "az",
        "gibi",
    }
)


def _top_k_dogrula(k: int) -> None:
    """Metrik hesabında `k` değerinin pozitif olduğunu doğrular."""
    if k <= 0:
        raise ValueError(f"Metrik hesaplaması için k pozitif olmalıdır (verilen: {k}).")


def recall_at_k(getirilen: Sequence[str], ilgili: str, k: int) -> float:
    """İlk `k` sonuç içinde ilgili parça varsa 1.0, yoksa 0.0 döndürür."""
    _top_k_dogrula(k)
    return 1.0 if ilgili in getirilen[:k] else 0.0


def hit_at_k(getirilen: Sequence[str], ilgili: str, k: int) -> float:
    """Tek kanıtlı (binary relevance) kurguda Recall@k ile özdeştir."""
    return recall_at_k(getirilen, ilgili, k)


def mrr_at_k(getirilen: Sequence[str], ilgili: str, k: int) -> float:
    """İlk `k` sonuçta ilgili parçanın ters sıralama skorunu (1/sıra) döndürür."""
    _top_k_dogrula(k)
    for sira, chunk_id in enumerate(getirilen[:k], start=1):
        if chunk_id == ilgili:
            return 1.0 / sira
    return 0.0


def ndcg_at_k(getirilen: Sequence[str], ilgili: str, k: int) -> float:
    """İlk `k` sonuç için normalize edilmiş indirimli kümülatif kazancı (nDCG@k) döndürür."""
    _top_k_dogrula(k)
    # binary kazanç; ideal DCG = 1 (tek ilgili ilk sırada)
    for sira, chunk_id in enumerate(getirilen[:k], start=1):
        if chunk_id == ilgili:
            return float(1.0 / np.log2(sira + 1))
    return 0.0


_METRIKLER: dict[str, Callable[[Sequence[str], str, int], float]] = {
    "recall": recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def soru_skorlari(
    sonuclar: Sequence[Sequence[str]],
    ilgililer: Sequence[str],
    metrik: str,
    k: int,
) -> np.ndarray:
    """Her soru için metrik skoru (bootstrap girdisi)."""
    if metrik not in _METRIKLER:
        raise ValueError(
            f"Bilinmeyen metrik: '{metrik}'. Desteklenenler: {list(_METRIKLER)}"
        )
    if len(sonuclar) != len(ilgililer):
        raise ValueError(
            f"Sonuç sayısı ({len(sonuclar)}) ile ilgili kanıt sayısı "
            f"({len(ilgililer)}) eşleşmiyor."
        )
    fn = _METRIKLER[metrik]
    return np.array(
        [fn(getirilen, ilgili, k) for getirilen, ilgili in zip(sonuclar, ilgililer)],
        dtype=float,
    )


def ortalama(skorlar: np.ndarray | Sequence[float]) -> float:
    """Skor dizisinin aritmetik ortalamasını döndürür; dizi boşsa 0.0 döner."""
    return float(np.mean(skorlar)) if len(skorlar) else 0.0


def eslestirilmis_bootstrap_fark(
    skorlar_a: np.ndarray,
    skorlar_b: np.ndarray,
    tekrar: int = VARSAYILAN_BOOTSTRAP_TEKRAR,
    tohum: int = VARSAYILAN_TOHUM,
    guven: float = VARSAYILAN_GUVEN_DUZEYI,
) -> dict[str, Any]:
    """A - B farkının eşleştirilmiş bootstrap güven aralığı.

    Dönen sözlük: {fark, alt, ust, anlamli}
    """
    a_arr = np.asarray(skorlar_a, dtype=float)
    b_arr = np.asarray(skorlar_b, dtype=float)
    if len(a_arr) != len(b_arr):
        raise ValueError(
            f"Eşleştirilmiş bootstrap için dizi uzunlukları eşit olmalıdır "
            f"({len(a_arr)} != {len(b_arr)})."
        )
    if len(a_arr) == 0:
        raise ValueError("Bootstrap hesabı için en az bir gözlem gereklidir.")

    rng = np.random.default_rng(tohum)
    farklar = a_arr - b_arr
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


def icerik_tokenlari(metin: str) -> list[str]:
    """Metindeki Türkçe etkisiz kelimeler (stopwords) hariç içerik tokenlarını döndürür."""
    return [
        t
        for t in _KELIME_RE.findall(metin.lower())
        if len(t) > 1 and t not in _TR_STOPWORDS
    ]


def kaynak_baglilik_skoru(yanit: str, kaynak_metinler: Sequence[str]) -> float:
    """Üretilen yanıttaki içerik sözcüklerinin kaynak bağlamda bulunma oranını hesaplar."""
    if "bu bilgi verilen kaynaklarda yer almamaktadır" in yanit.lower():
        return 1.0
    yanit_tok = icerik_tokenlari(yanit)
    if not yanit_tok:
        return 0.0
    kaynak_kumesi = set(icerik_tokenlari(" ".join(kaynak_metinler)))
    eslesen = sum(1 for t in yanit_tok if t in kaynak_kumesi)
    return float(eslesen / len(yanit_tok))


def token_f1(yanit: str, referans: str) -> float:
    """Üretilen yanıt ile referans yanıt arasındaki içerik token F1 skorunu hesaplar."""
    y_tok = icerik_tokenlari(yanit)
    r_tok = icerik_tokenlari(referans)
    if not y_tok or not r_tok:
        return 0.0
    ortak = sum((Counter(y_tok) & Counter(r_tok)).values())
    if ortak == 0:
        return 0.0
    kesinlik = ortak / len(y_tok)
    duyarlilik = ortak / len(r_tok)
    return float(2 * kesinlik * duyarlilik / (kesinlik + duyarlilik))

