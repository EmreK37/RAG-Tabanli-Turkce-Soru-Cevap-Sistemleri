"""Metrik fonksiyonları için birim testler.

Çalıştırma: pytest tests/ -q
"""
from __future__ import annotations

import numpy as np

from src.metrics import (
    eslestirilmis_bootstrap_fark,
    hit_at_k,
    kaynak_baglilik_skoru,
    mrr_at_k,
    ndcg_at_k,
    ortalama,
    recall_at_k,
    soru_skorlari,
    token_f1,
)


def test_recall() -> None:
    """Recall@k ve Hit@k hesaplamalarını doğrular."""
    assert recall_at_k(["a", "b", "c"], "b", 3) == 1.0
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0
    assert recall_at_k(["a", "b", "c"], "z", 3) == 0.0
    assert hit_at_k(["a", "b", "c"], "b", 2) == 1.0


def test_mrr() -> None:
    """MRR@k ters sıra puanlamasını doğrular."""
    assert mrr_at_k(["a", "b", "c"], "a", 3) == 1.0
    assert mrr_at_k(["a", "b", "c"], "c", 3) == 1.0 / 3.0
    assert mrr_at_k(["a", "b", "c"], "c", 2) == 0.0


def test_ndcg() -> None:
    """nDCG@k logaritmik indirimli kazanç hesabını doğrular."""
    assert ndcg_at_k(["a", "b"], "a", 2) == 1.0
    assert abs(ndcg_at_k(["a", "b"], "b", 2) - 1.0 / np.log2(3)) < 1e-9
    assert ndcg_at_k(["a", "b"], "z", 2) == 0.0


def test_soru_skorlari_ve_ortalama() -> None:
    """Toplu soru skoru vektörü ve ortalama hesabını doğrular."""
    sonuclar = [["c1", "c2"], ["c3", "c4"]]
    ilgililer = ["c1", "c9"]
    skorlar = soru_skorlari(sonuclar, ilgililer, "recall", 2)
    assert list(skorlar) == [1.0, 0.0]
    assert ortalama(skorlar) == 0.5
    assert ortalama(np.array([])) == 0.0


def test_bootstrap_ayni_sistem() -> None:
    """Özdeş iki sistem arasında farkın 0 ve anlamsız olduğunu doğrular."""
    rng = np.random.default_rng(0)
    skorlar = rng.random(50)
    sonuc = eslestirilmis_bootstrap_fark(skorlar, skorlar, tekrar=200)
    assert sonuc["fark"] == 0.0
    assert sonuc["anlamli"] is False


def test_bootstrap_belirgin_fark() -> None:
    """Tamamen ayrık iki sistem arasında farkın istatistiksel anlamlı olduğunu doğrular."""
    a = np.ones(50)
    b = np.zeros(50)
    sonuc = eslestirilmis_bootstrap_fark(a, b, tekrar=200)
    assert sonuc["fark"] == 1.0
    assert sonuc["anlamli"] is True


def test_rag_metrikleri() -> None:
    """Kaynak bağlılığı (faithfulness) ve içerik token F1 fonksiyonlarını doğrular."""
    kaynak = ["Staj uygulama esasları Senato tarafından belirlenir."]
    yanit = "Staj esasları Senato tarafından belirlenir."
    assert kaynak_baglilik_skoru(yanit, kaynak) == 1.0
    assert abs(token_f1(yanit, "Staj uygulama esasları Senato tarafından belirlenir.") - (8.0 / 9.0)) < 1e-6
    assert kaynak_baglilik_skoru("Bu bilgi verilen kaynaklarda yer almamaktadır.", kaynak) == 1.0


