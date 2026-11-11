"""Türkçe RAG araştırma projesi — pipeline modülleri.

Modüller:
- config: deney konfigürasyonu (configs/deney.yaml) yükleyici
- data: korpus parçalama ve soru seti giriş/çıkışı
- embeddings: dense embedding modeli sarmalayıcıları
- retrieval: FAISS (dense) ve BM25 (sözcüksel) erişim
- metrics: Recall@k, nDCG@k, MRR@k, Hit@k ve bootstrap güven aralığı
- generation: sabit LLM ile kaynaklı yanıt üretimi
- router: sorguya duyarlı embedding modeli seçici (RQ5)
"""
from __future__ import annotations

__all__ = [
    "config",
    "data",
    "embeddings",
    "generation",
    "metrics",
    "retrieval",
    "router",
]

