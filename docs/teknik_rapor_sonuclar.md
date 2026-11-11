# İP-7 & İP-8: Deneysel Bulgular, Ablasyon ve İstatistiksel Analiz Raporu

**Proje:** RAG Tabanlı Türkçe Soru-Cevap Sistemlerinde Embedding Modellerinin Karşılaştırılması  
*(Evaluating Multilingual Embedding Models for Turkish Retrieval-Augmented Question Answering: Retrieval Quality, Answer Faithfulness, and Computational Efficiency)*

---

## 1. Deneysel Kurulum Özeti

- **Kurumsal Korpus:** Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (`data/raw/yonetmelik.txt`, 46 madde, **82 chunk**, `boyut=800`, `ortusme=100`).
- **Değerlendirme Veri Seti (`data/processed/sorular.jsonl`):** 82 varyant grubunda (`G001`–`G082`) toplam **328 soru** (`156 train`, `52 val`, `120 test`; her grupta `olgusal`, `paraphrase`, `bicimbilimsel`, `cok-maddeli`). Varyant grupları bölmeler arasında sızıntı yapmayacak şekilde ayrılmıştır (`sizinti = yok`).
- **Sabit Üretim Katmanı:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 kuantize, `sicaklik=0.0`, `ana_top_k=5`, `prompts/rag_tr.txt`).
- **İstatistiksel Protokol:** 5 bağımsız tohum (`[42, 43, 44, 45, 46]`) ve 1000 tekrarlı eşleştirilmiş bootstrap güven aralığı (`%95 GA`).

---

## 2. RQ1 & RQ4: Erişim Kalitesi ve Hesaplama Verimliliği Karşılaştırması (120 Test Sorusu)

| Sistem | Model (`hf`) | Vektör Boyutu | İndeks Boyutu (KB) | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | İndeks Süresi (sn) | Ort. Gecikme (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`e5-base`** | `intfloat/multilingual-e5-base` | 768 | 246.00 | 0.9750 | **0.9917** | **0.9246** | **0.9031** | 17.53 | 11.81 |
| **`bge-m3`** | `BAAI/bge-m3` | 1024 | 328.00 | **0.9833** | **0.9917** | 0.9189 | 0.8943 | 60.86 | 20.46 |
| **`bm25`** | `BM25Okapi` (Sözcüksel) | 1651 (Sözlük) | **101.67** | 0.9250 | 0.9417 | 0.8479 | 0.8165 | **0.01** | **0.41** |
| **`tr-model`** | `trmteb/turkish-embedding-model` | 768 | 246.00 | 0.9333 | 0.9583 | 0.8336 | 0.7924 | 10.38 | 15.76 |
| **`qwen3-0.6b`** | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 328.00 | 0.9083 | 0.9417 | 0.8165 | 0.7751 | 36.80 | 62.74 |
| **`minilm`** | `paraphrase-multilingual-MiniLM-L12-v2` | **384** | 123.00 | 0.8500 | 0.9083 | 0.7477 | 0.6954 | 10.09 | 12.57 |


- **H1 Doğrulaması (`e5-base` vs `bm25`):** `nDCG@10` farkı **`+0.0767`**, %95 Eşleştirilmiş Bootstrap Güven Aralığı **`[0.0269, 0.1283]`** (`p < 0.05`, **anlamlı: True**).
- **RQ4 (Pareto Analizi):** `e5-base`, `bge-m3` ile eşdeğer erişim başarısını **3.5x daha düşük indeksleme süresi** ve **~2x daha düşük sorgu gecikmesi** ile sağlayarak tekil modeller arasında en güçlü Pareto noktasını oluşturmuştur.

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

## 4. RQ3 (H3): Retrieval Başarısının Uçtan Uca RAG Yanıt Kalitesine Etkisi

| Sistem | Hit@5 (Örneklem) | Faithfulness (Kaynağa Bağlılık) | Answer Correctness | Doğru Madde Atfı (`[MADDE-XX]`) |
|---|---:|---:|---:|---:|
| **`e5-base`** | **0.9500** | **0.8964** | **0.6712** | **0.7500** |
| **`tr-model`** | 0.7000 | 0.8832 | 0.6662 | 0.4500 |
| **`bm25`** | 0.6500 | 0.8843 | 0.6630 | 0.6000 |
| **`minilm`** | 0.6500 | 0.8597 | 0.6382 | 0.6000 |

- **Kanıt Varlığına Göre Kırılım:**
  - `Hit@5 = 0.0` iken: `Faithfulness = 0.8684`, `Answer Correctness = 0.5504`, `Doğru Atıf = 0.0952`
  - `Hit@5 = 1.0` iken: `Faithfulness = 0.8854`, `Answer Correctness = 0.6985`, `Doğru Atıf = 0.7797`
- **H3 Doğrulaması:** Retrieval tarafındaki +30 puanlık `Hit@5` artışı, `Faithfulness` ve `Answer Correctness` metriklerine doğrusal yansımamaktadır. Katı sistem yönergesi (`prompts/rag_tr.txt`) sayesinde LLM, kanıt bulunamadığında dahi uydurma yapmayıp kaynağa sadık kalmakta; buna karşın doğru maddeye atıf başarısı (`0.0952 -> 0.7797`) doğrudan erişim kalitesiyle belirlenmektedir.

---

## 5. RQ5 (H4): Sorguya Duyarlı Model Seçici (Router) ve Ablasyon Analizi (A2, A3, A6)

| Sistem / Ablasyon | nDCG@10 | Recall@5 | Ort. Gecikme (ms) | Yorum |
|---|---:|---:|---:|---|
| **`Router (RF - Tam)`** | **0.9018** | **0.9667** | **2.26** | 104 sorguyu `bm25`'e, 16 zor sorguyu `e5-base`'e yönlendirir. |
| **`Router (LogReg)`** | 0.8879 | **0.9667** | **1.35** | Doğrusal sınırla **9.1x** hızlanma, `0.9667` Recall@5. |
| **`A2: Soru Tipi Çıkarıldı`** | 0.8985 | 0.9750 | 2.24 | `soru_tipi` çıkarıldığında `nDCG@10` 0.33 puan düşer. |
| **`A3: Biçimbilimsel Özellik Çıkarıldı`** | 0.9042 | 0.9633 | 2.39 | `ortalama_token_uzunlugu` çıkarıldığında `Recall@5` 0.34 puan düşer ve gecikme artar. |
| **`A6: Ön-Çıkarım Maliyeti Dâhil`** | 0.9018 | 0.9667 | 19.58 | `bm25_dense_skor_farki` için ek dense çıkarım yapılırsa gecikme avantajı kaybolur. |
| **`Tek Model: bm25`** | 0.8479 | 0.9250 | 0.57 | Hızlı fakat `paraphrase` sorgularda düşük erişim. |
| **`Tek Model: e5-base`** | 0.9246 | 0.9750 | 12.33 | Yüksek kalite fakat her sorguda GPU çıkarım maliyeti. |
| **`Tek Model: bge-m3`** | 0.9189 | 0.9833 | 26.07 | En yüksek `Recall@5`, en yüksek gecikme. |

- **İstatistiksel Testler:**
  - `Router (RF) - BM25`: Fark = **`+0.0539`**, %95 GA = **`[0.0221, 0.0889]`**, **`anlamlı: True`** (Router, BM25'i istatistiksel olarak anlamlı biçimde geçer).
  - `Router (RF) - e5-base`: Fark = **`-0.0228`**, %95 GA = **`[-0.0628, 0.0148]`**, **`anlamlı: False`** (Router, `e5-base` ile istatistiksel olarak eşdeğer erişim kalitesini **5.5 kat daha düşük gecikmeyle (`2.26 ms` vs `12.33 ms`)** sunar).
  - **A6 Tasarım Çıkarımı:** Gerçek zamanlı üretim ortamında yönlendiricinin yoğun model ön-çıkarımı yapmadan yalnızca sıfır maliyetli sorgu ve BM25 özellikleri (`sorgu_uzunlugu`, `token_sayisi`, `soru_tipi`, `bm25_top1_skor`) ile çalışması gerektiği kanıtlanmıştır.
