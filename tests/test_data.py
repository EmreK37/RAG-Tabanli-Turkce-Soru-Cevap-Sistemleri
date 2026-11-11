"""Chunking davranışı için birim testler."""
from src.data import yonetmelik_parcala

ORNEK = """Önsöz metni burada yer alır.

MADDE 1 -
(1) Bu Yönetmeliğin amacı örnek olarak yazılmıştır. İkinci cümle de
buradadır. Üçüncü cümle biraz daha uzun tutulmuştur ki pencereleme
davranışı gözlenebilsin.

MADDE 2 -
(1) İkinci madde tek fıkradan oluşur.
"""


def test_madde_bolme():
    chunklar = yonetmelik_parcala(ORNEK, boyut=800, ortusme=100)
    assert [c.chunk_id for c in chunklar] == ["MADDE-1", "MADDE-2"]
    assert chunklar[0].madde_no == 1
    assert "amacı" in chunklar[0].metin


def test_uzun_madde_pencereleme():
    uzun = "MADDE 5 -\n" + " ".join(
        f"Bu örnek cümle numara {i} olarak yazılmıştır." for i in range(60)
    )
    chunklar = yonetmelik_parcala(uzun, boyut=200, ortusme=50)
    assert len(chunklar) > 1
    assert chunklar[0].chunk_id == "MADDE-5-1"
    assert all(c.madde_no == 5 for c in chunklar)


def test_ayni_satirda_degisiklik_ibaresi():
    metin = "MADDE 27 – (Değişik:RG-22/8/2022-31931)\nStaj esasları Senato tarafından belirlenir."
    chunklar = yonetmelik_parcala(metin, boyut=800, ortusme=100)
    assert len(chunklar) == 1
    assert chunklar[0].chunk_id == "MADDE-27"
    assert chunklar[0].madde_no == 27
