"""Chunking ve konfigürasyon davranışı için birim testler."""
from __future__ import annotations

from src.config import load_config, model_configs
from src.data import yonetmelik_parcala

ORNEK = """Önsöz metni burada yer alır.

MADDE 1 -
(1) Bu Yönetmeliğin amacı örnek olarak yazılmıştır. İkinci cümle de
buradadır. Üçüncü cümle biraz daha uzun tutulmuştur ki pencereleme
davranışı gözlenebilsin.

MADDE 2 -
(1) İkinci madde tek fıkradan oluşur.
"""


def test_madde_bolme() -> None:
    """Standart madde başlıklarının doğru ayrıştırıldığını doğrular."""
    chunklar = yonetmelik_parcala(ORNEK, boyut=800, ortusme=100)
    assert [c.chunk_id for c in chunklar] == ["MADDE-1", "MADDE-2"]
    assert chunklar[0].madde_no == 1
    assert "amacı" in chunklar[0].metin


def test_uzun_madde_pencereleme() -> None:
    """Boyut sınırını aşan uzun maddelerin alt parçalara bölündüğünü doğrular."""
    uzun = "MADDE 5 -\n" + " ".join(
        f"Bu örnek cümle numara {i} olarak yazılmıştır." for i in range(60)
    )
    chunklar = yonetmelik_parcala(uzun, boyut=200, ortusme=50)
    assert len(chunklar) > 1
    assert chunklar[0].chunk_id == "MADDE-5-1"
    assert all(c.madde_no == 5 for c in chunklar)


def test_ayni_satirda_degisiklik_ibaresi() -> None:
    """Aynı satırda değişiklik ibaresi taşıyan maddelerin (örn. MADDE 27) yakalandığını doğrular."""
    metin = "MADDE 27 – (Değişik:RG-22/8/2022-31931)\nStaj esasları Senato tarafından belirlenir."
    chunklar = yonetmelik_parcala(metin, boyut=800, ortusme=100)
    assert len(chunklar) == 1
    assert chunklar[0].chunk_id == "MADDE-27"
    assert chunklar[0].madde_no == 27


def test_gecersiz_pencere_parametreleri() -> None:
    """Geçersiz boyut veya örtüşme değerlerinde ValueError fırlatıldığını doğrular."""
    for boyut, ortusme in [(0, 10), (100, -1), (100, 100), (100, 120)]:
        hata_yakalandi = False
        try:
            yonetmelik_parcala(ORNEK, boyut=boyut, ortusme=ortusme)
        except ValueError:
            hata_yakalandi = True
        assert hata_yakalandi, f"({boyut}, {ortusme}) için ValueError bekleniyordu."


def test_konfigurasyon_yukleme() -> None:
    """Varsayılan deney konfigürasyonunun eksiksiz okunduğunu doğrular."""
    cfg = load_config()
    modeller = model_configs(cfg)
    assert cfg["deney"]["ad"] == "ana-karsilastirma"
    assert len(modeller) == 6


