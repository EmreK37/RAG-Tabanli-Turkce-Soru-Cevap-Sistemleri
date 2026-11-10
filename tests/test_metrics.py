"""Metrik fonksiyonları için birim testler.

Çalıştırma: pytest tests/ -q
"""
import numpy as np

from src.metrics import (
    eslestirilmis_bootstrap_fark,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall():
    assert recall_at_k(["a", "b", "c"], "b", 3) == 1.0
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0
    assert recall_at_k(["a", "b", "c"], "z", 3) == 0.0


def test_mrr():
    assert mrr_at_k(["a", "b", "c"], "a", 3) == 1.0
    assert mrr_at_k(["a", "b", "c"], "c", 3) == 1.0 / 3.0
    assert mrr_at_k(["a", "b", "c"], "c", 2) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "b"], "a", 2) == 1.0
    assert abs(ndcg_at_k(["a", "b"], "b", 2) - 1.0 / np.log2(3)) < 1e-9
    assert ndcg_at_k(["a", "b"], "z", 2) == 0.0


def test_bootstrap_ayni_sistem():
    rng = np.random.default_rng(0)
    skorlar = rng.random(50)
    sonuc = eslestirilmis_bootstrap_fark(skorlar, skorlar, tekrar=200)
    assert sonuc["fark"] == 0.0
    assert sonuc["anlamli"] is False


def test_bootstrap_belirgin_fark():
    a = np.ones(50)
    b = np.zeros(50)
    sonuc = eslestirilmis_bootstrap_fark(a, b, tekrar=200)
    assert sonuc["fark"] == 1.0
    assert sonuc["anlamli"] is True
