"""Retrieval deney çalıştırıcısı (İP-3 çekirdeği).

Kullanım:
    python scripts/deney_calistir.py --config configs/deney.yaml

Akış: konfigürasyonu oku -> korpusu parçala -> soruları yükle ->
her sistem (dense modeller + BM25) için indeks kur ve değerlendir ->
soru bazında skorları ve özet tabloyu results/ altına yaz.

Üretim (LLM) aşaması burada YOKTUR; o adım notebook 03'te yürür.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import chunklari_kaydet, jsonl_oku, korpus_yukle, yonetmelik_parcala
from src.embeddings import DenseEmbedder
from src.metrics import ortalama, soru_skorlari
from src.retrieval import Bm25Retriever, FaissRetriever

METRIKLER = [("recall", 5), ("recall", 10), ("ndcg", 10), ("mrr", 10), ("hit", 5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deney.yaml")
    ap.add_argument("--cihaz", default=None, help="cuda / cpu (boş: otomatik)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    damga = datetime.now().strftime("%Y%m%d-%H%M%S")
    cikti_dir = Path("results") / f"{cfg['deney']['ad']}-{damga}"
    cikti_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, cikti_dir / "kullanilan-config.yaml")

    # --- Veri ---
    metin = korpus_yukle(cfg["veri"]["korpus"])
    chunklar = yonetmelik_parcala(
        metin,
        boyut=cfg["parcalama"]["boyut_karakter"],
        ortusme=cfg["parcalama"]["ortusme_karakter"],
    )
    chunklari_kaydet(chunklar, cikti_dir / "chunklar.jsonl")
    print(f"Chunk sayısı: {len(chunklar)}")

    sorular = jsonl_oku(cfg["veri"]["sorular"])
    test_sorulari = [s for s in sorular if s.get("bolme", "test") == "test"]
    print(f"Test sorusu: {len(test_sorulari)}")
    chunk_metin = {c.chunk_id: c.metin for c in chunklar}

    # --- Sistemler ---
    sistemler: dict[str, object] = {}
    indeks_sureleri: dict[str, float] = {}
    for model in cfg["modeller"]:
        ad = model["ad"]
        t0 = time.time()
        if model["tur"] == "sozcuksel":
            sistemler[ad] = Bm25Retriever(chunklar)
        else:
            if not model.get("hf") or model["hf"] == "DOLDURULACAK":
                print(f"ATLANIYOR (HF adı tanımsız): {ad}")
                continue
            embedder = DenseEmbedder(
                model["hf"],
                sorgu_oneki=model.get("sorgu_oneki", ""),
                belge_oneki=model.get("belge_oneki", ""),
                sorgu_yonergesi=model.get("sorgu_yonergesi", ""),
                cihaz=args.cihaz,
            )
            sistemler[ad] = FaissRetriever(embedder, chunklar)
        indeks_sureleri[ad] = time.time() - t0
        print(f"İndeks hazır: {ad} ({indeks_sureleri[ad]:.1f} sn)")

    # --- Değerlendirme ---
    maks_k = max(cfg["erisim"]["top_k"])
    ozet_satirlar: list[dict] = []
    soru_satirlar: list[dict] = []
    router_satirlar: list[dict] = []
    for ad, sistem in sistemler.items():
        sonuclar: list[list[str]] = []
        gecikmeler: list[float] = []
        for soru in test_sorulari:
            t0 = time.time()
            bulunan = sistem.retrieve(soru["soru"], k=maks_k)
            gecikmeler.append((time.time() - t0) * 1000)
            sonuclar.append([c for c, _ in bulunan])
        ilgililer = [s["kanit_chunk_id"] for s in test_sorulari]

        satir: dict = {
            "sistem": ad,
            "indeks_suresi_sn": round(indeks_sureleri[ad], 2),
            "ort_gecikme_ms": round(float(sum(gecikmeler) / len(gecikmeler)), 2),
        }
        for metrik, k in METRIKLER:
            skorlar = soru_skorlari(sonuclar, ilgililer, metrik, k)
            satir[f"{metrik}@{k}"] = round(ortalama(skorlar), 4)
            for soru, skor, bulunan in zip(test_sorulari, skorlar, sonuclar):
                soru_satirlar.append(
                    {
                        "sistem": ad,
                        "soru_id": soru["soru_id"],
                        "tip": soru.get("tip", ""),
                        "metrik": f"{metrik}@{k}",
                        "skor": float(skor),
                        "ilk_sonuc": bulunan[0] if bulunan else "",
                    }
                )
        ozet_satirlar.append(satir)
        print(satir)

        # Router (RQ5) için tüm bölmelerde (train/val/test) top-1 skor ve ndcg@10 kaydı
        tum_sonuclar: list[list[str]] = []
        tum_top1_skor: list[float] = []
        for soru in sorular:
            bulunan = sistem.retrieve(soru["soru"], k=10)
            tum_sonuclar.append([c for c, _ in bulunan])
            tum_top1_skor.append(float(bulunan[0][1]) if bulunan else 0.0)
        tum_ilgililer = [s["kanit_chunk_id"] for s in sorular]
        tum_ndcg10 = soru_skorlari(tum_sonuclar, tum_ilgililer, "ndcg", 10)
        tum_recall5 = soru_skorlari(tum_sonuclar, tum_ilgililer, "recall", 5)
        for soru, top1_s, ndcg_s, rec5_s, bulunan_id in zip(
            sorular, tum_top1_skor, tum_ndcg10, tum_recall5, tum_sonuclar
        ):
            router_satirlar.append(
                {
                    "sistem": ad,
                    "soru_id": soru["soru_id"],
                    "bolme": soru.get("bolme", "test"),
                    "tip": soru.get("tip", ""),
                    "varyant_grup": soru.get("varyant_grup", ""),
                    "top1_skor": round(top1_s, 6),
                    "ndcg@10": round(float(ndcg_s), 6),
                    "recall@5": round(float(rec5_s), 6),
                    "ilk_sonuc": bulunan_id[0] if bulunan_id else "",
                }
            )

    with open(cikti_dir / "ozet.csv", "w", newline="", encoding="utf-8") as f:
        yazar = csv.DictWriter(f, fieldnames=list(ozet_satirlar[0].keys()))
        yazar.writeheader()
        yazar.writerows(ozet_satirlar)
    with open(cikti_dir / "soru-bazinda.csv", "w", newline="", encoding="utf-8") as f:
        yazar = csv.DictWriter(f, fieldnames=list(soru_satirlar[0].keys()))
        yazar.writeheader()
        yazar.writerows(soru_satirlar)
    with open(cikti_dir / "router-veri.csv", "w", newline="", encoding="utf-8") as f:
        yazar = csv.DictWriter(f, fieldnames=list(router_satirlar[0].keys()))
        yazar.writeheader()
        yazar.writerows(router_satirlar)
    # üretim aşaması için chunk metinlerine erişim notu
    (cikti_dir / "NOT.txt").write_text(
        f"Chunk metinleri: chunklar.jsonl\nKorpus: {cfg['veri']['korpus']}\n",
        encoding="utf-8",
    )
    print(f"Sonuçlar: {cikti_dir}")


if __name__ == "__main__":
    main()
