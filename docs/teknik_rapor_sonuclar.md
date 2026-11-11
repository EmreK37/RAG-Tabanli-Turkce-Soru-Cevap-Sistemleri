# İP-7 & İP-8: Deneysel Bulgular, Ablasyon ve İstatistiksel Analiz Raporu

**Proje:** RAG Tabanlı Türkçe Soru-Cevap Sistemlerinde Embedding Modellerinin Karşılaştırılması  
*(Evaluating Multilingual Embedding Models for Turkish Retrieval-Augmented Question Answering: Retrieval Quality, Answer Faithfulness, and Computational Efficiency)*

---

## 1. Deneysel Kurulum Özeti

- **Kurumsal Korpus:** Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (`data/raw/yonetmelik.txt`, 46 madde, **82 chunk**, `boyut=800`, `ortusme=100`).
- **Değerlendirme Veri Seti (`data/processed/sorular.jsonl`):** 82 varyant grubunda (`G001`–`G082`) toplam **328 soru** (`156 train`, `52 val`, `120 test`; her grupta `olgusal`, `paraphrase`, `bicimbilimsel`, `cok-maddeli`). Varyant grupları bölmeler arasında sızıntı yapmayacak şekilde ayrılmıştır (`sizinti = yok`).
- **Standart Türkçe Dış Benchmark (`data/processed/standart_tr_benchmark.jsonl`):** Bilim (`scifact-tr`), finans (`fiqa-tr`), hukuk, teknoloji ve eğitim alanlarından oluşan 30 standart Türkçe sorgu–belge çifti.
- **Cevaplanabilirlik / Kaçınma Veri Seti (`data/processed/kacinma_sorulari.jsonl`):** Yönetmelik kapsamı dışında kalan 16 cevaplanamaz (unanswerable) kontrol sorusu.
- **Sabit Üretim Katmanı:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 kuantize, `sicaklik=0.0`, `ana_top_k=5`, `prompts/rag_tr.txt`).
- **İstatistiksel Protokol:** 5 bağımsız tohum (`[42, 43, 44, 45, 46]`) ve 1000 tekrarlı eşleştirilmiş bootstrap güven aralığı (`%95 GA`).

---

## 2. RQ1 & RQ4: Erişim Kalitesi, İndeks Boyutu ve Hesaplama Verimliliği Karşılaştırması

### 2.1. Kurumsal Yönetmelik Korpusu Sonuçları (120 Test Sorusu)

| Sistem | Model (`hf`) | Vektör Boyutu | İndeks Boyutu (KB) | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | İndeks Süresi (sn) | Ort. Gecikme (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`e5-base`** | `intfloat/multilingual-e5-base` | 768 | 246.00 | 0.9750 | **0.9917** | **0.9246** | **0.9031** | 17.53 | 11.81 |
| **`bge-m3`** | `BAAI/bge-m3` | 1024 | 328.00 | **0.9833** | **0.9917** | 0.9189 | 0.8943 | 60.86 | 20.46 |
| **`bm25`** | `BM25Okapi` (Sözcüksel) | 1651 (Sözlük) | **101.67** | 0.9250 | 0.9417 | 0.8479 | 0.8165 | **0.01** | **0.41** |
| **`tr-model`** | `trmteb/turkish-embedding-model` | 768 | 246.00 | 0.9333 | 0.9583 | 0.8336 | 0.7924 | 10.38 | 15.76 |
| **`qwen3-0.6b`** | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 328.00 | 0.9083 | 0.9417 | 0.8165 | 0.7751 | 36.80 | 62.74 |
| **`minilm`** | `paraphrase-multilingual-MiniLM-L12-v2` | **384** | 123.00 | 0.8500 | 0.9083 | 0.7477 | 0.6954 | 10.09 | 12.57 |

- **H1 Doğrulaması (`e5-base` vs `bm25`):** `nDCG@10` farkı **`+0.0767`**, %95 Eşleştirilmiş Bootstrap Güven Aralığı **`[0.0269, 0.1283]`** (`p < 0.05`, **anlamlı: True**).
- **RQ4 (Pareto ve Bellek Analizi):** `e5-base` (`768` boyut, `246 KB` FAISS indeksi), `bge-m3` (`1024` boyut, `328 KB` FAISS indeksi) ile eşdeğer erişim başarısını **%25 daha düşük indeks belleği**, **3.5x daha düşük indeksleme süresi** ve **~2x daha düşük sorgu gecikmesi** ile sağlayarak tekil modeller arasında en güçlü Pareto noktasını oluşturmuştur.

### 2.2. Bölüm 3: Standart Türkçe Çok Alanlı Benchmark Karşılaştırması (`results/dis-benchmark-ozet.csv`)

| Sistem | Vektör Boyutu | İndeks Boyutu (KB) | Ort. Gecikme (ms) | Recall@5 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`e5-base`** | 768 | 90.00 | 11.47 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **`bge-m3`** | 1024 | 120.00 | 21.10 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **`qwen3-0.6b`** | 1024 | 120.00 | 45.25 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **`tr-model`** | 768 | 90.00 | 11.58 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **`minilm`** | 384 | 45.00 | 11.35 | **1.0000** | **1.0000** | 0.9564 | 0.9417 |
| **`bm25`** | 523 (Sözlük) | **17.52** | **0.20** | 0.9667 | **1.0000** | 0.9517 | 0.9367 |

---

## 3. RQ2 (H2): Biçimbilimsel Karmaşıklık ve Paraphrase Sorguların Etkisi (`nDCG@10`)

| Sistem | `olgusal` | `cok-maddeli` | `bicimbilimsel` | `paraphrase` | `olgusal -> paraphrase` Kaybı |
|---|---:|---:|---:|---:|---:|
| **`bge-m3`** | 0.8841 | 0.9254 | **0.9441** | **0.9218** | **+0.0377** (Kayıp yok) |
| **`e5-base`** | **0.9249** | **0.9544** | 0.9350 | 0.8842 | -0.0407 |
| **`tr-model`** | 0.8188 | 0.8977 | 0.8585 | 0.7592 | -0.0596 |
| **`qwen3-0.6b`** | 0.8121 | 0.8714 | 0.8632 | 0.7193 | -0.0928 |
| **`minilm`** | 0.7186 | 0.7936 | 0.8256 | 0.6531 | -0.0655 |
| **`bm25`** | 0.9206 | 0.9462 | 0.8841 | **0.6407** | **-0.2799 (-%30.4)** |

- **H2 Doğrulaması:** Yönetmelik terimlerinin doğrudan geçtiği `olgusal` sorularda `bm25` (`0.9206`) çok güçlüdür; ancak soru günlük öğrenci diliyle yeniden ifade edildiğinde (`paraphrase`) sözcüksel örtüşme kırıldığı için `bm25` **`0.6407`'ye (-28 puan)** çökmektedir. Yoğun temsil kullanan `bge-m3` (`0.9218`) ve `e5-base` (`0.8842`) ise anlamsal eşleştirmeyle bu düşüşe direnç göstermektedir.

---

## 4. RQ3 (H3): Retrieval Başarısının Uçtan Uca RAG Yanıt Kalitesine ve Kaçınma Davranışına Etkisi

### 4.1. Tüm Modeller ve Adaptif Router İçin RAG Yanıt Kalitesi (`results/rq3-rag-ozet.csv`)

| Sistem | Hit@5 (Örneklem) | Faithfulness (Kaynağa Bağlılık) | Answer Correctness | Doğru Madde Atfı (`[MADDE-XX]`) |
|---|---:|---:|---:|---:|
| **`e5-base`** | **0.9500** | 0.8922 | 0.6714 | **0.7500** |
| **`qwen3-0.6b`** | **0.9500** | 0.8839 | 0.6664 | 0.6000 |
| **`bge-m3`** | 0.9000 | 0.8739 | **0.6793** | **0.7500** |
| **`router-rf`** *(Önerilen)* | 0.9000 | 0.8864 | 0.6685 | 0.7000 |
| **`tr-model`** | 0.7000 | **0.8945** | 0.6653 | 0.5000 |
| **`bm25`** | 0.6500 | 0.8657 | 0.6604 | 0.5500 |
| **`minilm`** | 0.6500 | 0.8662 | 0.6382 | 0.6000 |

- **Kanıt Varlığına Göre Kırılım:**
  - `Hit@5 = 0.0` iken: `Faithfulness = 0.8684`, `Answer Correctness = 0.5504`, `Doğru Atıf = 0.0952`
  - `Hit@5 = 1.0` iken: `Faithfulness = 0.8854`, `Answer Correctness = 0.6985`, `Doğru Atıf = 0.7797`
- **H3 Doğrulaması:** Retrieval tarafındaki +30 puanlık `Hit@5` artışı (`0.65 -> 0.95`), `Faithfulness` (`0.866 -> 0.892`) ve `Answer Correctness` (`0.660 -> 0.671`) metriklerine doğrusal yansımamaktadır. Buna karşın doğru maddeye atıf başarısı (`0.0952 -> 0.7797`) doğrudan erişim kalitesiyle belirlenmektedir.

### 4.2. Bölüm 6.2: Cevaplanabilirlik ve Yanıt Vermekten Kaçınma (Abstention) Testi (`results/rq3-kacinma-ozet.csv`)

| Sistem | Cevaplanamaz Soru Sayısı | Kaçınma Doğruluğu (`Abstention`) | Halüsinasyon Oranı |
|---|---:|---:|---:|
| **`e5-base`** | 16 | **1.0000 (%100)** | **0.0000 (%0)** |
| **`bge-m3`** | 16 | **1.0000 (%100)** | **0.0000 (%0)** |
| **`tr-model`** | 16 | **1.0000 (%100)** | **0.0000 (%0)** |
| **`bm25`** | 16 | **1.0000 (%100)** | **0.0000 (%0)** |

- **Kaçınma Bulgusu:** Yönetmelik dışı 16 kontrol sorusunun tamamında (`U001`–`U016`), tüm erişim sistemleriyle getirilen bağlamlar üzerinde sabit LLM (`Qwen2.5-7B-Instruct`, `sicaklik=0.0`), `prompts/rag_tr.txt` içindeki 2. kurala sadık kalarak *"Bu bilgi verilen kaynaklarda yer almıyor."* yanıtını üretmiş ve **%0 halüsinasyon** göstermiştir.

---

## 5. RQ5 (H4) ve Ablasyon Deneyleri (`A1`–`A6`)

### 5.1. `A1`, `A2`, `A3`, `A6`: Sorguya Duyarlı Router ve Özellik/Maliyet Ablasyonları (`results/rq5-router-ozet.csv`)

| Sistem / Ablasyon | nDCG@10 | Recall@5 | Ort. Gecikme (ms) | Yorum |
|---|---:|---:|---:|---|
| **`A1: Router (RF - Tam)`** | **0.9018** | **0.9667** | **2.26** | 104 sorguyu `bm25`'e, 16 zor sorguyu `e5-base`'e yönlendirir. |
| **`A1: Router (LogReg)`** | 0.8879 | **0.9667** | **1.35** | Doğrusal sınırla **9.1x** hızlanma, `0.9667` Recall@5. |
| **`A2: Soru Tipi Çıkarıldı`** | 0.8985 | 0.9750 | 2.24 | `soru_tipi` çıkarıldığında `nDCG@10` 0.33 puan düşer. |
| **`A3: Biçimbilimsel Özellik Çıkarıldı`** | 0.9042 | 0.9633 | 2.39 | `ortalama_token_uzunlugu` çıkarıldığında `Recall@5` 0.34 puan düşer ve gecikme artar. |
| **`A6: Ön-Çıkarım Maliyeti Dâhil`** | 0.9018 | 0.9667 | 19.58 | `bm25_dense_skor_farki` için ek dense çıkarım yapılırsa gecikme avantajı kaybolur. |
| **`Tek Model: bm25`** | 0.8479 | 0.9250 | 0.57 | Hızlı fakat `paraphrase` sorgularda düşük erişim. |
| **`Tek Model: e5-base`** | 0.9246 | 0.9750 | 12.33 | Yüksek kalite fakat her sorguda GPU çıkarım maliyeti. |
| **`Tek Model: bge-m3`** | 0.9189 | 0.9833 | 26.07 | En yüksek `Recall@5`, en yüksek gecikme. |

- **İstatistiksel Testler:**
  - `Router (RF) - BM25`: Fark = **`+0.0539`**, %95 GA = **`[0.0221, 0.0889]`**, **`anlamlı: True`** (Router, BM25'i istatistiksel olarak anlamlı biçimde geçer).
  - `Router (RF) - e5-base`: Fark = **`-0.0228`**, %95 GA = **`[-0.0628, 0.0148]`**, **`anlamlı: False`** (Router, `e5-base` ile istatistiksel olarak eşdeğer erişim kalitesini **5.5 kat daha düşük gecikmeyle (`2.26 ms` vs `12.33 ms`)** sunar).

### 5.2. `A4`: Farklı Top-k Değerleri Ablasyonu (`results/a4-topk-ablasyon.csv` & `results/a4-rag-topk-ozet.csv`)

**Erişim Katmanında Top-k (`k = 1, 3, 5, 10`) Etkisi (`Recall@k` / `nDCG@k`):**

| Sistem | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`e5-base`** | **0.8583** | 0.9333 | 0.9750 | **0.9917** | **0.8583** | 0.9024 | **0.9196** | **0.9246** |
| **`bge-m3`** | 0.8250 | **0.9583** | **0.9833** | **0.9917** | 0.8250 | **0.9059** | 0.9159 | 0.9189 |
| **`bm25`** | 0.7333 | 0.9000 | 0.9250 | 0.9417 | 0.7333 | 0.8319 | 0.8420 | 0.8479 |
| **`tr-model`** | 0.6917 | 0.8750 | 0.9333 | 0.9583 | 0.6917 | 0.8008 | 0.8252 | 0.8336 |
| **`qwen3-0.6b`** | 0.6667 | 0.8750 | 0.9083 | 0.9417 | 0.6667 | 0.7927 | 0.8070 | 0.8180 |
| **`minilm`** | 0.5667 | 0.7917 | 0.8500 | 0.9083 | 0.5667 | 0.7043 | 0.7287 | 0.7477 |

**Üretim (RAG) Katmanında Top-k (`k = 1, 3, 5`) Etkisi (`e5-base` üzerinde):**

| Sistem | Top-k | Hit@k | Faithfulness (Kaynağa Bağlılık) | İçerik Token F1 |
|---|---:|---:|---:|---:|
| **`e5-base`** | `k = 1` | 0.5000 | 0.6909 | 0.2916 |
| **`e5-base`** | `k = 3` | 0.6667 | 0.8679 | 0.2927 |
| **`e5-base`** | `k = 5` | **1.0000** | **0.8921** | **0.4083** |

- **A4 Bulgusu:** `k=1`'den `k=5`'e geçiş hem `Recall@k` değerini (`0.8583 -> 0.9750`) hem de RAG yanıt doğruluğunu (`Token F1: 0.2916 -> 0.4083`, `Faithfulness: 0.6909 -> 0.8921`) belirgin biçimde artırmaktadır. `k=5`'ten `k=10`'a geçişte ise erişim kazancı doygunluğa (`+0.0167`) ulaşırken LLM bağlam uzunluğunu ikiye katlamaktadır; bu nedenle `ana_top_k = 5` seçimi optimaldir.

### 5.3. `A5`: Farklı Belge Parçası (Chunk) Uzunlukları Ablasyonu (`results/a5-chunk-ablasyon.csv`)

| Chunk Boyutu / Örtüşme | Toplam Chunk | Sistem | İndeks Boyutu (KB) | Recall@5 | nDCG@10 (Chunk) | nDCG@10 (Madde) |
|---:|---:|---|---:|---:|---:|---:|
| **`400 / 50`** | 146 | `e5-base` | 438.00 | 0.8417 | 0.7114 | 0.9246 |
| **`400 / 50`** | 146 | `bge-m3` | 584.00 | 0.8583 | 0.7319 | **0.9384** |
| **`400 / 50`** | 146 | `bm25` | 109.28 | 0.7667 | 0.6219 | 0.8443 |
| **`800 / 100` *(Ana)*** | **82** | **`e5-base`** | **246.00** | **0.9750** | **0.9246** | **0.9369** |
| **`800 / 100` *(Ana)*** | **82** | **`bge-m3`** | **328.00** | **0.9833** | 0.9189 | 0.9294 |
| **`800 / 100` *(Ana)*** | **82** | **`bm25`** | **101.67** | 0.9250 | 0.8479 | 0.8632 |
| **`1200 / 150`** | 66 | `e5-base` | 198.00 | 0.9750 | 0.9079 | 0.9322 |
| **`1200 / 150`** | 66 | `bge-m3` | 264.00 | 0.9833 | 0.9007 | 0.9295 |
| **`1200 / 150`** | 66 | `bm25` | 97.78 | 0.9417 | 0.8565 | 0.8727 |

- **A5 Bulgusu:** `400/50` karakter penceresi maddeleri aşırı böldüğü için (`146 chunk`) indeks boyutunu %78 büyütmekte ve fıkralar arası bütünlüğü kopararak chunk bazlı `nDCG@10` skorunu `0.9246`'dan `0.7114`'e düşürmektedir. `1200/150` penceresinde ise uzun maddelerde birden fazla alt konu tek parçada birleştiği için yoğun modellerde sıralama keskinliği (`0.9246 -> 0.9079`) hafifçe gerilemektedir. **`800 karakter / 100 örtüşme`** hem erişim kalitesi hem de indeks maliyeti açısından en dengeli noktadır.
