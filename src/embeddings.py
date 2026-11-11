"""Dense embedding modeli sarmalayıcısı.

Sentence-Transformers üzerinden çalışır. Modele özgü zorunluluklar
(E5 "query:/passage:" ön ekleri, Qwen3 sorgu yönergesi) konfigürasyondan
okunur; burada sabitlenmez.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

VARSAYILAN_BATCH_BOYUTU = 32


def _embedding_boyutu_al(model: SentenceTransformer) -> int:
    """SentenceTransformer sürüm farklarını soyutlayarak vektör boyutunu döndürür."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


class DenseEmbedder:
    """Çok dilli ve Türkçe dense embedding modelleri için ortak kodlayıcı."""

    def __init__(
        self,
        hf_adi: str,
        sorgu_oneki: str = "",
        belge_oneki: str = "",
        sorgu_yonergesi: str = "",
        cihaz: str | None = None,
    ) -> None:
        if not hf_adi or not hf_adi.strip():
            raise ValueError("HuggingFace model adı (`hf_adi`) boş olamaz.")

        self.model = SentenceTransformer(hf_adi, device=cihaz)
        self.sorgu_oneki = sorgu_oneki
        self.belge_oneki = belge_oneki
        self.sorgu_yonergesi = sorgu_yonergesi
        self.boyut = _embedding_boyutu_al(self.model)

    def _sorgu_metni(self, sorgu: str) -> str:
        """Sorguyu modele özgü yönerge veya ön ek ile biçimlendirir."""
        if self.sorgu_yonergesi:
            # Qwen3 tarzı instruction-aware kullanım
            return f"Instruct: {self.sorgu_yonergesi}\nQuery: {sorgu}"
        return f"{self.sorgu_oneki}{sorgu}"

    def belgeleri_kodla(
        self,
        metinler: Sequence[str],
        batch: int = VARSAYILAN_BATCH_BOYUTU,
    ) -> np.ndarray:
        """Belge metinlerini L2-normalize edilmiş birim vektörlere dönüştürür."""
        if not metinler:
            return np.empty((0, self.boyut), dtype=np.float32)

        hazir = [f"{self.belge_oneki}{m}" for m in metinler]
        return self.model.encode(
            hazir,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def sorguyu_kodla(self, sorgu: str) -> np.ndarray:
        """Tek bir sorguyu L2-normalize edilmiş 1B vektöre dönüştürür."""
        return self.model.encode(
            [self._sorgu_metni(sorgu)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

