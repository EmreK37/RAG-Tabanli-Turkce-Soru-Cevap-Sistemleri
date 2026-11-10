"""Dense embedding modeli sarmalayıcısı.

Sentence-Transformers üzerinden çalışır. Modele özgü zorunluluklar
(E5 "query:/passage:" ön ekleri, Qwen3 sorgu yönergesi) konfigürasyondan
okunur; burada sabitlenmez.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    def __init__(
        self,
        hf_adi: str,
        sorgu_oneki: str = "",
        belge_oneki: str = "",
        sorgu_yonergesi: str = "",
        cihaz: str | None = None,
    ):
        self.model = SentenceTransformer(hf_adi, device=cihaz)
        self.sorgu_oneki = sorgu_oneki
        self.belge_oneki = belge_oneki
        self.sorgu_yonergesi = sorgu_yonergesi
        self.boyut = self.model.get_sentence_embedding_dimension()

    def _sorgu_metni(self, sorgu: str) -> str:
        if self.sorgu_yonergesi:
            # Qwen3 tarzı instruction-aware kullanım
            return f"Instruct: {self.sorgu_yonergesi}\nQuery: {sorgu}"
        return f"{self.sorgu_oneki}{sorgu}"

    def belgeleri_kodla(self, metinler: list[str], batch: int = 32) -> np.ndarray:
        hazir = [f"{self.belge_oneki}{m}" for m in metinler]
        return self.model.encode(
            hazir,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def sorguyu_kodla(self, sorgu: str) -> np.ndarray:
        return self.model.encode(
            [self._sorgu_metni(sorgu)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
