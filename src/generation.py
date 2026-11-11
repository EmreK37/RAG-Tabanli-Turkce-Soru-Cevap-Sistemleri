"""Sabit LLM ile kaynaklı yanıt üretimi (RAG generation).

Tüm embedding sistemleri AYNI LLM, prompt ve üretim ayarlarını kullanır;
böylece yanıt kalitesi farkı yalnızca erişim katmanına bağlanabilir.

Colab T4 (16 GB) için varsayılan: 4-bit kuantize 7-8B model.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

VARSAYILAN_LLM_MODELI = "Qwen/Qwen2.5-7B-Instruct"
VARSAYILAN_MAKS_YENI_TOKEN = 512
VARSAYILAN_PROMPT_YOLU = "prompts/rag_tr.txt"


def _model_yukleme_argumanlari(dort_bit: bool) -> dict[str, Any]:
    """HuggingFace model yüklemesi için cihaz ve 4-bit kuantizasyon ayarlarını hazırlar."""
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if dort_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return kwargs


class RagGenerator:
    """Erişilen kaynak parçalarını kullanarak sabit LLM ile yanıt üreten sarmalayıcı."""

    def __init__(
        self,
        model_adi: str = VARSAYILAN_LLM_MODELI,
        dort_bit: bool = True,
        maks_yeni_token: int = VARSAYILAN_MAKS_YENI_TOKEN,
        prompt_yolu: str | Path = VARSAYILAN_PROMPT_YOLU,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_adi)
        kwargs = _model_yukleme_argumanlari(dort_bit)
        self.model = AutoModelForCausalLM.from_pretrained(model_adi, **kwargs)
        self.maks_yeni_token = maks_yeni_token
        self.sablon = Path(prompt_yolu).read_text(encoding="utf-8-sig")

    def _sohbet_metni_hazirla(self, soru: str, kaynak_metinler: Sequence[str]) -> str:
        """Kaynak metinleri ve soruyu şablona yerleştirip chat formatına dönüştürür."""
        context = "\n\n".join(kaynak_metinler)
        prompt = self.sablon.format(context=context, question=soru)
        mesajlar = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            mesajlar, tokenize=False, add_generation_prompt=True
        )

    def uret(self, soru: str, kaynak_metinler: list[str]) -> str:
        """Kaynak metinleri bağlam yapıp yanıt üretir."""
        metin = self._sohbet_metni_hazirla(soru, kaynak_metinler)
        girdiler = self.tokenizer([metin], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            cikti = self.model.generate(
                **girdiler,
                max_new_tokens=self.maks_yeni_token,
                do_sample=False,  # sıcaklık 0: tekrarlanabilir üretim
                pad_token_id=self.tokenizer.eos_token_id,
            )
        yeni = cikti[0][girdiler.input_ids.shape[1]:]
        return self.tokenizer.decode(yeni, skip_special_tokens=True).strip()

