# RAG Tabanlı Türkçe Soru-Cevap Sistemlerinde Embedding Modellerinin Karşılaştırılması

*Evaluating Multilingual Embedding Models for Turkish Retrieval-Augmented Question Answering: Retrieval Quality, Answer Faithfulness, and Computational Efficiency*

Bu repo, Türkçe RAG (Retrieval-Augmented Generation) sistemlerinde farklı embedding modellerinin erişim başarısını, uçtan uca yanıt kalitesini ve hesaplama maliyetini karşılaştıran araştırma projesinin kod, veri ve deney altyapısını içerir. Projenin özgün katkısı, sorgu özelliklerine duyarlı hafif bir embedding modeli seçicisidir (router).

## Araştırma soruları

| Kod | Soru |
|---|---|
| RQ1 | Farklı embedding modelleri Türkçe anlamsal erişimde anlamlı performans farkı gösteriyor mu? |
| RQ2 | Türkçenin eklemeli yapısı ve sorguların yeniden ifade edilmesi erişim başarısını nasıl etkiliyor? |
| RQ3 | Daha yüksek retrieval performansı yanıt doğruluğunu ve faithfulness'ı her zaman artırıyor mu? |
| RQ4 | Embedding boyutu, indeksleme maliyeti ve sorgu gecikmesi ile erişim başarısı arasında nasıl bir ilişki var? |
| RQ5 | Sorgu tipine göre model seçen hafif bir yönlendirici, tek modele göre daha iyi kalite-maliyet dengesi sağlar mı? |

## Karşılaştırılan sistemler

| Model | Rol |
|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Hafif çok dilli referans |
| intfloat/multilingual-e5-base | Çok dilli dense retrieval |
| BAAI/bge-m3 | Güçlü çok dilli erişim (dense + sparse + multi-vector) |
| Qwen/Qwen3-Embedding-0.6B | Yeni nesil embedding referansı |
| TR-MTEB Türkçe modeli | Türkçe odaklı karşılaştırma (tam HF adı kurulumda doğrulanır) |
| BM25 | Sözcüksel (embedding'siz) baz çizgisi |

Her model için ayrı FAISS indeksi kurulur; farklı modellerin vektörleri aynı indekste karıştırılmaz. Tüm sistemlerde aynı LLM, prompt, chunk yöntemi, Top-k ve üretim ayarları kullanılır.

## Hızlı başlangıç (Google Colab — önerilen)

1. Aşağıdaki notebook'u Colab'da açın: **notebooks/00_kurulum_ve_kontrol.ipynb**
   - Colab bağlantısı: https://colab.research.google.com/github/EmreK37/RAG-Tabanl-T-rk-e-Soru-Cevap-Sistemleri/blob/main/notebooks/00_kurulum_ve_kontrol.ipynb
2. Çalışma zamanı türünü GPU (T4) seçin: Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU.
3. Hücreleri sırayla çalıştırın. Notebook Drive'ı bağlar, repoyu klonlar, bağımlılıkları kurar ve ortamı doğrular.
4. Deneyler sırayla `01`–`05` numaralı notebook'larla yürür; sonuçlar Drive'a ve `results/` klasörüne yazılır.

## Yerel kurulum (alternatif)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -c "from src.retrieval import Bm25Retriever; print('kurulum tamam')"
```

## Deney çalıştırma

```bash
python scripts/deney_calistir.py --config configs/deney.yaml
```

Tüm ayarlar (modeller, chunk boyutu, Top-k, tohumlar, LLM) tek dosyada tutulur: [configs/deney.yaml](configs/deney.yaml). Her çalışma, kullanılan konfigürasyonun kopyasını sonuçlarla birlikte kaydeder.

## Repo yapısı

| Klasör | İçerik |
|---|---|
| `notebooks/` | Colab notebook'ları (kurulum, veri, retrieval, RAG, router, analiz) |
| `src/` | Pipeline modülleri: veri, embedding, retrieval, üretim, router, metrikler |
| `scripts/` | Komut satırından deney çalıştırıcılar |
| `configs/` | Deney konfigürasyonları (tek doğruluk kaynağı) |
| `data/` | Veri şeması ve ham/işlenmiş veriler (büyük dosyalar Git'e girmez) |
| `results/` | Deney çıktıları (CSV/JSON) ve özet tablolar |
| `docs/` | Literatür matrisi, veri planı, toplantı notları |

## Veri

Birincil kurumsal korpus: Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (`data/raw/yonetmelik.txt`, 46 madde). Hedef: 300–500 özgün soru, her soru için kanıt chunk kimliği, referans yanıt ve paraphrase varyantları. Aynı sorunun varyantları farklı veri bölmelerine dağıtılmaz. Standart karşılaştırma için TR-MTEB ve TR-TEB retrieval görevleri kullanılır. Veri şeması: [data/README.md](data/README.md).

## Ekip ve iş paketleri (8 hafta)

| İş paketi | Hafta | Çıktı |
|---|---|---|
| İP-1 Literatür ve veri setleri | 1 | Literatür matrisi, veri planı |
| İP-2 Veri hazırlama | 2 | Değerlendirme veri seti |
| İP-3 Embedding karşılaştırmaları | 3 | İlk erişim tabloları |
| İP-4 RAG sistemi | 4 | Çalışan RAG altyapısı |
| İP-5 Özgün yöntem (router) | 5 | Adaptif model seçici |
| İP-6 Uçtan uca deneyler | 6 | RAG sonuç tabloları |
| İP-7 Ablasyon ve istatistik | 7 | Ablasyon ve istatistik raporu |
| İP-8 Sonuç ve teslim | 8 | Kod, veri, teknik rapor |

Görev dağılımı: Araştırmacı 1 veri setleri · Araştırmacı 2 embedding/BM25 · Araştırmacı 3 RAG altyapısı · Araştırmacı 4 model seçici · Araştırmacı 5 deney otomasyonu/istatistik. Katkı kuralları için [CONTRIBUTING.md](CONTRIBUTING.md).

## Değerlendirme

- **Retrieval:** Recall@5/10, nDCG@10, MRR@10, Hit@k, sorgu gecikmesi, indeks boyutu
- **RAG yanıt:** Referans yanına göre doğruluk, faithfulness (kaynağa bağlılık), cevaplanabilirlik, kaçınma davranışı
- **İstatistik:** En az 5 tohum, sorgu bazında eşleştirilmiş bootstrap güven aralıkları

## Lisans

Araştırma amaçlı kullanım içindir. Model ve veri seti lisansları ilgili kartlardan doğrulanmalıdır.
