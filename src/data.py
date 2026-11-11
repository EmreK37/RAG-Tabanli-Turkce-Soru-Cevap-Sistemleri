"""Korpus parçalama (chunking) ve soru seti giriş/çıkışı.

Korpus: yönetmelik metni. Metin önce "MADDE n" başlıklarına bölünür,
uzun maddeler cümle sınırlarına saygılı karakter pencereleriyle
parçalanır. Her chunk kaynağını (madde no) taşır.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VARSAYILAN_CHUNK_BOYUTU = 800
VARSAYILAN_CHUNK_ORTUSME = 100

_MADDE_RE = re.compile(r"^MADDE\s+(\d+)\s*[-–]?\s*", re.MULTILINE)
_CUMLE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """Yönetmelik korpusundan üretilen tek bir metin parçası."""

    chunk_id: str          # örn. "MADDE-38" veya "MADDE-38-2"
    madde_no: int
    metin: str


def _pencere_parametrelerini_dogrula(boyut: int, ortusme: int) -> None:
    """Pencereleme parametrelerinin geçerliliğini denetler."""
    if boyut <= 0:
        raise ValueError(f"Chunk boyutu pozitif olmalıdır (verilen: {boyut}).")
    if ortusme < 0:
        raise ValueError(f"Chunk örtüşmesi negatif olamaz (verilen: {ortusme}).")
    if ortusme >= boyut:
        raise ValueError(
            f"Chunk örtüşmesi ({ortusme}) boyuttan ({boyut}) küçük olmalıdır."
        )


def _ortusme_cumlelerini_sec(cumleler: list[str], ortusme: int) -> tuple[list[str], int]:
    """Mevcut pencerenin sonundan `ortusme` sınırını aşmayan cümleleri seçer."""
    tasma: list[str] = []
    tasma_uzunluk = 0
    for onceki in reversed(cumleler):
        if tasma_uzunluk + len(onceki) > ortusme:
            break
        tasma.insert(0, onceki)
        tasma_uzunluk += len(onceki) + 1
    return tasma, tasma_uzunluk


def _pencerele(metin: str, boyut: int, ortusme: int) -> list[str]:
    """Metni cümle sınırlarını koruyarak yaklaşık `boyut` karakterlik
    parçalara böler; parçalar arasında `ortusme` karakter pay bırakır."""
    _pencere_parametrelerini_dogrula(boyut, ortusme)
    cumleler = [c.strip() for c in _CUMLE_RE.split(metin) if c.strip()]
    parcalar: list[str] = []
    mevcut: list[str] = []
    uzunluk = 0

    for cumle in cumleler:
        if mevcut and uzunluk + len(cumle) > boyut:
            parcalar.append(" ".join(mevcut))
            # örtüşme: son cümleleri taşı
            mevcut, uzunluk = _ortusme_cumlelerini_sec(mevcut, ortusme)
        mevcut.append(cumle)
        uzunluk += len(cumle) + 1

    if mevcut:
        parcalar.append(" ".join(mevcut))
    return parcalar


def yonetmelik_parcala(
    metin: str,
    boyut: int = VARSAYILAN_CHUNK_BOYUTU,
    ortusme: int = VARSAYILAN_CHUNK_ORTUSME,
) -> list[Chunk]:
    """Yönetmelik metnini madde bazlı chunk listesine çevirir."""
    _pencere_parametrelerini_dogrula(boyut, ortusme)
    parcalar = _MADDE_RE.split(metin)
    # split çıktısı: [giris, no1, govde1, no2, govde2, ...]
    chunklar: list[Chunk] = []
    for i in range(1, len(parcalar), 2):
        madde_no = int(parcalar[i])
        govde = parcalar[i + 1].strip() if i + 1 < len(parcalar) else ""
        if not govde:
            continue
        alt = _pencerele(govde, boyut, ortusme)
        for j, alt_metin in enumerate(alt):
            ek = f"-{j + 1}" if len(alt) > 1 else ""
            chunklar.append(
                Chunk(
                    chunk_id=f"MADDE-{madde_no}{ek}",
                    madde_no=madde_no,
                    metin=f"MADDE {madde_no} - {alt_metin}",
                )
            )
    return chunklar


def korpus_yukle(yol: str | Path) -> str:
    """Ham korpus dosyasını UTF-8 (BOM toleranslı) olarak okur."""
    return Path(yol).read_text(encoding="utf-8-sig")


def chunklari_kaydet(chunklar: list[Chunk], yol: str | Path) -> None:
    """Chunk listesini JSONL formatında diske kaydeder."""
    hedef = Path(yol)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with open(hedef, "w", encoding="utf-8") as f:
        for c in chunklar:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def jsonl_oku(yol: str | Path) -> list[dict[str, Any]]:
    """JSONL dosyasındaki boş olmayan satırları sözlük listesi olarak okur."""
    with open(yol, encoding="utf-8-sig") as f:
        return [json.loads(satir) for satir in f if satir.strip()]


def jsonl_yaz(kayitlar: list[dict[str, Any]], yol: str | Path) -> None:
    """Sözlük listesini JSONL formatında diske yazar."""
    hedef = Path(yol)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with open(hedef, "w", encoding="utf-8") as f:
        for kayit in kayitlar:
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")

