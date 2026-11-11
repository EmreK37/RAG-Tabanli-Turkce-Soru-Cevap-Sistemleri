"""Sorguya duyarlı embedding modeli seçici — RQ5.

Sorgu özelliklerinden, hangi embedding modelinin bu sorguda daha iyi
sonuç vereceğini tahmin eden hafif sınıflandırıcı.

Disiplin: Seçici YALNIZCA train/val bölmeleriyle eğitilir. Test
sonuçları seçimde kullanılmaz. En az 5 tohumla eğitilip eşleştirilmiş
güven aralıklarıyla raporlanır (bkz. src/metrics.py).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

VARSAYILAN_LOGREG_ITER = 1000
VARSAYILAN_RF_AGAC = 300
VARSAYILAN_TOHUM = 42
DESTEKLENEN_ROUTER_TURLERI = ("logreg", "rf")

_SORU_KELIMELERI = ("ne", "nedir", "nasıl", "hangi", "kaç", "kim", "nere")


def sorgu_ozellikleri(
    sorgu: str, bm25_top1: float, dense_top1: float
) -> dict[str, float | int]:
    """Bir sorgudan router özellikleri çıkarır."""
    tokenlar = sorgu.lower().split()
    ortalama_uzunluk = (
        float(np.mean([len(t) for t in tokenlar])) if tokenlar else 0.0
    )
    return {
        "sorgu_uzunlugu": len(sorgu),
        "token_sayisi": len(tokenlar),
        "soru_tipi": 1 if any(k in tokenlar for k in _SORU_KELIMELERI) else 0,
        "bm25_dense_skor_farki": float(bm25_top1 - dense_top1),
        "ortalama_token_uzunlugu": ortalama_uzunluk,
    }


def ozellik_matrisi(
    kayitlar: Sequence[Mapping[str, Any]], alanlar: Sequence[str]
) -> np.ndarray:
    """Özellik sözlüklerinden belirtilen alanları seçerek 2B float matrisi üretir."""
    if not alanlar:
        raise ValueError("Özellik matrisi için en az bir alan adı belirtilmelidir.")
    return np.array([[k[alan] for alan in alanlar] for k in kayitlar], dtype=float)


def _siniflandirici_olustur(
    tur: str, tohum: int
) -> LogisticRegression | RandomForestClassifier:
    """Belirtilen router türüne göre scikit-learn sınıflandırıcısı döndürür."""
    if tur == "logreg":
        return LogisticRegression(
            max_iter=VARSAYILAN_LOGREG_ITER, random_state=tohum
        )
    if tur == "rf":
        return RandomForestClassifier(
            n_estimators=VARSAYILAN_RF_AGAC, random_state=tohum
        )
    raise ValueError(
        f"Bilinmeyen router türü: {tur}. Desteklenenler: {DESTEKLENEN_ROUTER_TURLERI}"
    )


class ModelSecici:
    """Kazanan modeli tahmin eden sınıflandırıcı sarmalayıcısı."""

    def __init__(self, tur: str = "logreg", tohum: int = VARSAYILAN_TOHUM) -> None:
        self.tur = tur
        self.tohum = tohum
        self.model = _siniflandirici_olustur(tur, tohum)

    def egit(self, X: np.ndarray, y: np.ndarray) -> None:
        """y: her sorguda en iyi sonucu veren modelin etiketi."""
        self.model.fit(X, y)

    def tahmin(self, X: np.ndarray) -> np.ndarray:
        """Verilen özellik matrisi için model etiketi tahminlerini döndürür."""
        return self.model.predict(X)

