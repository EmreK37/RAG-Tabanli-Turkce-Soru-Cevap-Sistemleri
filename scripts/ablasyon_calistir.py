"""Ablasyon deneyleri (A4, A5) ve bağımsız Türkçe benchmark çalıştırıcısı.

Kullanım:
    python scripts/ablasyon_calistir.py --config configs/deney.yaml

Üretilen çıktılar:
- results/a4-topk-ablasyon.csv       : Top-k = [1, 3, 5, 10] erişim ablasyonu (A4)
- results/a5-chunk-ablasyon.csv      : Chunk boyutu = [400/50, 800/100, 1200/150] ablasyonu (A5)
- results/dis-benchmark-ozet.csv     : Çok alanlı standart Türkçe benchmark (TR-MTEB/TR-TEB)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deney_calistir import csv_kaydet
from src.config import load_config
from src.data import Chunk, jsonl_oku, korpus_yukle, yonetmelik_parcala
from src.embeddings import DenseEmbedder
from src.metrics import icerik_tokenlari, ortalama, soru_skorlari
from src.retrieval import Bm25Retriever, FaissRetriever, Retriever

TOP_K_DEGERLERI: tuple[int, ...] = (1, 3, 5, 10)
CHUNK_AYARLARI: tuple[tuple[int, int], ...] = (
    (400, 50),
    (800, 100),
    (1200, 150),
)
STANDART_BENCHMARK_YOLU = Path("data/processed/standart_tr_benchmark.jsonl")


def argumanlari_cozumle() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    ap = argparse.ArgumentParser(
        description="A4 (Top-k), A5 (Chunk Boyutu) ve Dış Benchmark Çalıştırıcısı"
    )
    ap.add_argument("--config", default="configs/deney.yaml")
    ap.add_argument("--cihaz", default=None, help="cuda / cpu (boş: otomatik)")
    return ap.parse_args()


def embedder_havuzu_olustur(
    cfg: dict[str, Any], cihaz: str | None
) -> dict[str, DenseEmbedder]:
    """Modelleri belleğe tek sefer yükleyerek tüm ablasyonlarda yeniden kullanır."""
    havuz: dict[str, DenseEmbedder] = {}
    for model in cfg["modeller"]:
        if model["tur"] == "sozcuksel":
            continue
        ad = model["ad"]
        hf_adi = model.get("hf")
        if not hf_adi or hf_adi == "DOLDURULACAK":
            continue
        print(f"Model yükleniyor: {ad} ({hf_adi})")
        havuz[ad] = DenseEmbedder(
            hf_adi,
            sorgu_oneki=model.get("sorgu_oneki", ""),
            belge_oneki=model.get("belge_oneki", ""),
            sorgu_yonergesi=model.get("sorgu_yonergesi", ""),
            cihaz=cihaz,
        )
    return havuz


def indeksleri_kur(
    cfg: dict[str, Any],
    embedder_havuzu: dict[str, DenseEmbedder],
    chunklar: list[Chunk],
) -> tuple[dict[str, Retriever], dict[str, float]]:
    """Verilen chunk listesi için tüm sistemlerin indekslerini kurar ve süreyi ölçer."""
    sistemler: dict[str, Retriever] = {}
    sureler: dict[str, float] = {}
    for model in cfg["modeller"]:
        ad = model["ad"]
        t0 = time.time()
        if model["tur"] == "sozcuksel":
            sistemler[ad] = Bm25Retriever(chunklar)
        elif ad in embedder_havuzu:
            sistemler[ad] = FaissRetriever(embedder_havuzu[ad], chunklar)
        else:
            continue
        sureler[ad] = time.time() - t0
    return sistemler, sureler


def _madde_no_cikar(chunk_id: str) -> int:
    """'MADDE-38-2' biçimindeki kimlikten madde numarasını (38) döndürür."""
    parcalar = chunk_id.split("-")
    return int(parcalar[1])


def _hedef_chunk_esle(
    soru: dict[str, Any],
    referans_chunk_metinleri: dict[str, str],
    yeni_chunklar: list[Chunk],
) -> str:
    """Farklı chunk boyutlarında sorunun altın kanıt parçasını en yüksek örtüşmeyle eşler."""
    orijinal_id = soru["kanit_chunk_id"]
    madde_no = _madde_no_cikar(orijinal_id)
    adaylar = [c for c in yeni_chunklar if c.madde_no == madde_no]
    if not adaylar:
        return orijinal_id
    if len(adaylar) == 1:
        return adaylar[0].chunk_id

    hedef_metin = (
        f"{soru.get('referans_yanit', '')} "
        f"{referans_chunk_metinleri.get(orijinal_id, '')}"
    )
    hedef_tokenlar = set(icerik_tokenlari(hedef_metin))
    en_iyi_id = adaylar[0].chunk_id
    en_iyi_skor = -1
    for aday in adaylar:
        aday_tokenlar = set(icerik_tokenlari(aday.metin))
        ortak = len(hedef_tokenlar & aday_tokenlar)
        if ortak > en_iyi_skor:
            en_iyi_skor = ortak
            en_iyi_id = aday.chunk_id
    return en_iyi_id


def a4_topk_ablasyonu_calistir(
    sistemler: dict[str, Retriever],
    test_sorulari: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """A4: Farklı Top-k (1, 3, 5, 10) değerlerinin erişim metriklerine etkisini ölçer."""
    maks_k = max(TOP_K_DEGERLERI)
    ilgililer = [s["kanit_chunk_id"] for s in test_sorulari]
    satirlar: list[dict[str, Any]] = []

    for ad, sistem in sistemler.items():
        sonuclar = [
            [c for c, _ in sistem.retrieve(s["soru"], k=maks_k)]
            for s in test_sorulari
        ]
        for k in TOP_K_DEGERLERI:
            satirlar.append(
                {
                    "sistem": ad,
                    "top_k": k,
                    "recall": round(
                        ortalama(soru_skorlari(sonuclar, ilgililer, "recall", k)), 4
                    ),
                    "hit": round(
                        ortalama(soru_skorlari(sonuclar, ilgililer, "hit", k)), 4
                    ),
                    "ndcg": round(
                        ortalama(soru_skorlari(sonuclar, ilgililer, "ndcg", k)), 4
                    ),
                    "mrr": round(
                        ortalama(soru_skorlari(sonuclar, ilgililer, "mrr", k)), 4
                    ),
                }
            )
    return satirlar


def a5_chunk_ablasyonu_calistir(
    cfg: dict[str, Any],
    embedder_havuzu: dict[str, DenseEmbedder],
    metin: str,
    test_sorulari: list[dict[str, Any]],
    referans_chunk_metinleri: dict[str, str],
) -> list[dict[str, Any]]:
    """A5: Farklı chunk uzunluklarının (400/50, 800/100, 1200/150) etkisini ölçer."""
    satirlar: list[dict[str, Any]] = []

    for boyut, ortusme in CHUNK_AYARLARI:
        chunklar = yonetmelik_parcala(metin, boyut=boyut, ortusme=ortusme)
        ilgililer = [
            _hedef_chunk_esle(s, referans_chunk_metinleri, chunklar)
            for s in test_sorulari
        ]
        ilgili_maddeler = [
            f"MADDE-{_madde_no_cikar(s['kanit_chunk_id'])}" for s in test_sorulari
        ]
        sistemler, sureler = indeksleri_kur(cfg, embedder_havuzu, chunklar)

        for ad, sistem in sistemler.items():
            sonuclar_chunk: list[list[str]] = []
            sonuclar_madde: list[list[str]] = []
            gecikmeler: list[float] = []

            for soru in test_sorulari:
                t0 = time.time()
                bulunan = sistem.retrieve(soru["soru"], k=10)
                gecikmeler.append((time.time() - t0) * 1000)
                chunk_idler = [c for c, _ in bulunan]
                sonuclar_chunk.append(chunk_idler)
                # Madde düzeyinde tekilleştirilmiş sıralama (chunk boyutu değişimine duyarlı olmayan kontrol)
                madde_sirasi: list[str] = []
                for cid in chunk_idler:
                    m_id = f"MADDE-{_madde_no_cikar(cid)}"
                    if m_id not in madde_sirasi:
                        madde_sirasi.append(m_id)
                sonuclar_madde.append(madde_sirasi)

            satirlar.append(
                {
                    "chunk_boyutu": boyut,
                    "ortusme": ortusme,
                    "chunk_sayisi": len(chunklar),
                    "sistem": ad,
                    "vektor_boyutu": sistem.vektor_boyutu,
                    "indeks_boyutu_kb": sistem.indeks_boyutu_kb(),
                    "indeks_suresi_sn": round(sureler[ad], 2),
                    "ort_gecikme_ms": round(
                        float(sum(gecikmeler) / len(gecikmeler)), 2
                    ),
                    "recall@5": round(
                        ortalama(soru_skorlari(sonuclar_chunk, ilgililer, "recall", 5)),
                        4,
                    ),
                    "recall@10": round(
                        ortalama(
                            soru_skorlari(sonuclar_chunk, ilgililer, "recall", 10)
                        ),
                        4,
                    ),
                    "ndcg@10": round(
                        ortalama(soru_skorlari(sonuclar_chunk, ilgililer, "ndcg", 10)),
                        4,
                    ),
                    "mrr@10": round(
                        ortalama(soru_skorlari(sonuclar_chunk, ilgililer, "mrr", 10)),
                        4,
                    ),
                    "madde_ndcg@10": round(
                        ortalama(
                            soru_skorlari(sonuclar_madde, ilgili_maddeler, "ndcg", 10)
                        ),
                        4,
                    ),
                }
            )
            print(
                f"[A5 {boyut}/{ortusme}] {ad:10s} -> "
                f"nDCG@10={satirlar[-1]['ndcg@10']:.4f}, "
                f"Madde-nDCG@10={satirlar[-1]['madde_ndcg@10']:.4f}, "
                f"İndeks={satirlar[-1]['indeks_boyutu_kb']} KB"
            )
    return satirlar


def dis_benchmark_calistir(
    cfg: dict[str, Any],
    embedder_havuzu: dict[str, DenseEmbedder],
) -> list[dict[str, Any]]:
    """Bölüm 3: Bağımsız çok alanlı Türkçe retrieval benchmark karşılaştırmasını yürütür."""
    kayitlar = jsonl_oku(STANDART_BENCHMARK_YOLU)
    chunklar = [
        Chunk(chunk_id=k["id"], madde_no=i + 1, metin=k["belge"])
        for i, k in enumerate(kayitlar)
    ]
    ilgililer = [k["id"] for k in kayitlar]
    sistemler, sureler = indeksleri_kur(cfg, embedder_havuzu, chunklar)

    satirlar: list[dict[str, Any]] = []
    for ad, sistem in sistemler.items():
        sonuclar: list[list[str]] = []
        gecikmeler: list[float] = []
        for kayit in kayitlar:
            t0 = time.time()
            bulunan = sistem.retrieve(kayit["sorgu"], k=10)
            gecikmeler.append((time.time() - t0) * 1000)
            sonuclar.append([c for c, _ in bulunan])

        satir = {
            "benchmark": "TR-Standart-CokAlanli-30",
            "sistem": ad,
            "vektor_boyutu": sistem.vektor_boyutu,
            "indeks_boyutu_kb": sistem.indeks_boyutu_kb(),
            "indeks_suresi_sn": round(sureler[ad], 2),
            "ort_gecikme_ms": round(float(sum(gecikmeler) / len(gecikmeler)), 2),
            "recall@5": round(
                ortalama(soru_skorlari(sonuclar, ilgililer, "recall", 5)), 4
            ),
            "recall@10": round(
                ortalama(soru_skorlari(sonuclar, ilgililer, "recall", 10)), 4
            ),
            "ndcg@10": round(
                ortalama(soru_skorlari(sonuclar, ilgililer, "ndcg", 10)), 4
            ),
            "mrr@10": round(
                ortalama(soru_skorlari(sonuclar, ilgililer, "mrr", 10)), 4
            ),
        }
        satirlar.append(satir)
        print(f"[Dış Benchmark] {ad:10s} -> nDCG@10={satir['ndcg@10']:.4f}")
    return satirlar


def main() -> None:
    """A4, A5 ablasyonlarını ve dış benchmark değerlendirmesini uçtan uca çalıştırır."""
    args = argumanlari_cozumle()
    cfg = load_config(args.config)
    cikti_dir = Path("results")
    cikti_dir.mkdir(parents=True, exist_ok=True)

    metin = korpus_yukle(cfg["veri"]["korpus"])
    ana_chunklar = yonetmelik_parcala(
        metin,
        boyut=cfg["parcalama"]["boyut_karakter"],
        ortusme=cfg["parcalama"]["ortusme_karakter"],
    )
    referans_chunk_metinleri = {c.chunk_id: c.metin for c in ana_chunklar}
    sorular = jsonl_oku(cfg["veri"]["sorular"])
    test_sorulari = [s for s in sorular if s.get("bolme", "test") == "test"]

    embedder_havuzu = embedder_havuzu_olustur(cfg, args.cihaz)
    ana_sistemler, _ = indeksleri_kur(cfg, embedder_havuzu, ana_chunklar)

    print("\n--- A4: Top-k Ablasyonu (k = 1, 3, 5, 10) ---")
    a4_satirlar = a4_topk_ablasyonu_calistir(ana_sistemler, test_sorulari)
    csv_kaydet(cikti_dir / "a4-topk-ablasyon.csv", a4_satirlar)
    print(f"Kaydedildi: {cikti_dir / 'a4-topk-ablasyon.csv'} ({len(a4_satirlar)} satır)")

    print("\n--- A5: Chunk Boyutu Ablasyonu (400/50, 800/100, 1200/150) ---")
    a5_satirlar = a5_chunk_ablasyonu_calistir(
        cfg, embedder_havuzu, metin, test_sorulari, referans_chunk_metinleri
    )
    csv_kaydet(cikti_dir / "a5-chunk-ablasyon.csv", a5_satirlar)
    print(f"Kaydedildi: {cikti_dir / 'a5-chunk-ablasyon.csv'} ({len(a5_satirlar)} satır)")

    print("\n--- Bölüm 3: Standart Türkçe Retrieval Benchmark Karşılaştırması ---")
    dis_satirlar = dis_benchmark_calistir(cfg, embedder_havuzu)
    csv_kaydet(cikti_dir / "dis-benchmark-ozet.csv", dis_satirlar)
    print(f"Kaydedildi: {cikti_dir / 'dis-benchmark-ozet.csv'} ({len(dis_satirlar)} satır)")


if __name__ == "__main__":
    main()

