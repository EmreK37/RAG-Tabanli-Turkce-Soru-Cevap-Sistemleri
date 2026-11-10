"""Sorguya duyarlı embedding modeli seçici — RQ5.

Sorgu özelliklerinden, hangi embedding modelinin bu sorguda daha iyi
sonuç vereceğini tahmin eden hafif sınıflandırıcı.

Disiplin: Seçici YALNIZCA train/val bölmeleriyle eğitilir. Test
sonuçları seçimde kullanılmaz. En az 5 tohumla eğitilip eşleştirilmiş
güven aralıklarıyla raporlanır (bkz. src/metrics.py).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

_SORU_KELIMELERI = ("ne", "nedir", "nasıl", "hangi", "kaç", "kim", "nere")


def sorgu_ozellikleri(sorgu: str, bm25_top1: float, dense_top1: float) -> dict:
    """Bir sorgudan router özellikleri çıkarır."""
    tokenlar = sorgu.lower().split()
    return {
        "sorgu_uzunlugu": len(sorgu),
        "token_sayisi": len(tokenlar),
        "soru_tipi": 1 if any(k in tokenlar for k in _SORU_KELIMELERI) else 0,
        "bm25_dense_skor_farki": float(bm25_top1 - dense_top1),
        "ortalama_token_uzunlugu": float(
            np.mean([len(t) for t in tokenlar]) if tokenlar else 0.0
        ),
    }


def ozellik_matrisi(kayitlar: list[dict], alanlar: list[str]) -> np.ndarray:
    return np.array([[k[alan] for alan in alanlar] for k in kayitlar], dtype=float)


class ModelSecici:
    """Kazanan modeli tahmin eden sınıflandırıcı sarmalayıcısı."""

    def __init__(self, tur: str = "logreg", tohum: int = 42):
        if tur == "logreg":
            self.model = LogisticRegression(max_iter=1000, random_state=tohum)
        elif tur == "rf":
            self.model = RandomForestClassifier(
                n_estimators=300, random_state=tohum
            )
        else:
            raise ValueError(f"Bilinmeyen router türü: {tur}")

    def egit(self, X: np.ndarray, y: np.ndarray) -> None:
        """y: her sorguda en iyi sonucu veren modelin etiketi."""
        self.model.fit(X, y)

    def tahmin(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
