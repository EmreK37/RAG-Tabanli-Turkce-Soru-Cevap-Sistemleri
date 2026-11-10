"""Sabit LLM ile kaynaklı yanıt üretimi (RAG generation).

Tüm embedding sistemleri AYNI LLM, prompt ve üretim ayarlarını kullanır;
böylece yanıt kalitesi farkı yalnızca erişim katmanına bağlanabilir.

Colab T4 (16 GB) için varsayılan: 4-bit kuantize 7-8B model.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class RagGenerator:
    def __init__(
        self,
        model_adi: str = "Qwen/Qwen2.5-7B-Instruct",
        dort_bit: bool = True,
        maks_yeni_token: int = 512,
        prompt_yolu: str | Path = "prompts/rag_tr.txt",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_adi)
        kwargs: dict = {"device_map": "auto"}
        if dort_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_adi, **kwargs)
        self.maks_yeni_token = maks_yeni_token
        self.sablon = Path(prompt_yolu).read_text(encoding="utf-8")

    def uret(self, soru: str, kaynak_metinler: list[str]) -> str:
        """Kaynak metinleri bağlam yapıp yanıt üretir."""
        context = "\n\n".join(kaynak_metinler)
        prompt = self.sablon.format(context=context, question=soru)
        mesajlar = [{"role": "user", "content": prompt}]
        metin = self.tokenizer.apply_chat_template(
            mesajlar, tokenize=False, add_generation_prompt=True
        )
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
