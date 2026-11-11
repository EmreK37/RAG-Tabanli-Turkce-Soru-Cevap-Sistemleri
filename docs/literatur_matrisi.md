# İP-1: Literatür Matrisi ve Özgün Değer Konumlandırması

Bu belge, Türkçe RAG (Retrieval-Augmented Generation) sistemlerinde embedding modellerinin karşılaştırılması ve sorguya duyarlı model seçici (router) araştırmasının temel literatür dayanaklarını özetler.

---

## 1. Temel Literatür Matrisi

| Çalışma | Yıl / Yayın | Odak & Görev | Kullanılan Modeller / Yöntemler | Veri Setleri & Ölçek | Temel Bulgular | Projemizle İlişkisi & Boşluk |
|---|---|---|---|---|---|---|
| **Gurbanov & Bilgin** *(Bursa Uludağ Ü.)* | 2024 | Türkçe RAG sistemlerinde geri çağırma (retrieval) yöntemlerinin karşılaştırılması | EBR (Sentence-BERT), DPR, BM25 + FAISS (`800` karakter chunk, `100` örtüşme) | TQuAD, TDV İslam Ansiklopedisi (~120B içerik), Türkçe Vikipedi (750 başlık) | EBR (`Recall@1: 0.45`, `Recall@3: 0.65`, Güven `0.46`) en başarılı; DPR Türkçe'de düşük (`0.15–0.26`); BM25 sözcüksel baz çizgisi olarak rekabetçi (`0.30–0.35`). | Chunk boyutu (`800/100`) ve metodolojik kurgu için temel referans. Ancak yeni nesil çok dilli modeller (`BGE-M3`, `Qwen3`, `E5`) ve sorguya duyarlı yönlendirici (router) incelenmemiştir. |
| **Baysan & Güngör (TR-MTEB)** | EMNLP 2025 | Türkçe metin gömme (text embedding) modelleri için kapsamlı kıyaslama paketi | `trmteb/turkish-embedding-model`, çok dilli ve Türkçe BERT tabanlı yoğun (dense) modeller | 6 görev kategorisinde 26+ Türkçe veri seti (`msmarco-tr`, `fiqa-tr`, `scifact-tr`, `nfcorpus-tr`, `stsb-tr` vb.) | Türkçe odaklı ince ayar (fine-tuning) yapılan modeller genel benzerlikte güçlüdür; ancak görev tipine göre çok dilli büyük modellerle rekabet değişkendir. | `tr-model` (`trmteb/turkish-embedding-model`) seçimimizin kaynağı. TR-MTEB salt embedding sıralaması sunar; uçtan uca RAG yanıt sadakati (faithfulness) ve router içermez. |
| **TR-TEB** | LREC 2026 | Türkçe metin gömme modellerinin geniş ölçekli değerlendirmesi | 45 farklı açık ve ticari embedding modeli | Çoklu Türkçe erişim, sınıflandırma ve anlamsal benzerlik görevleri | Model boyutu ve indeksleme maliyeti arttıkça erişim artışı azalan verim (diminishing returns) gösterir. | Projemizin **RQ3** (retrieval -> yanıt kalitesi doğrusal mı?) ve **RQ5** (sorguya duyarlı router) ile salt model sıralamasının ötesine geçmesini motive eder. |
| **Wang et al. (Multilingual E5)** | 2024 | Çok dilli kontrastif ön-eğitimli metin gömme modelleri | `intfloat/multilingual-e5-base` (`query: ` ve `passage: ` asimetrik ön ekleri) | 100+ dilde milyar ölçekli metin çifti + MTEB / MIRACL | Asimetrik sorgu/belge ön ekleri kullanıldığında çok dilli yoğun erişimde güçlü sıfır-örnek (zero-shot) başarımı sağlar. | Konfigürasyonumuzdaki `e5-base` sisteminin zorunlu `query: ` / `passage: ` ön ek disiplininin kaynağı. |
| **Chen et al. (BGE-M3)** | 2024 | Çok dilli, çok işlevli ve çok taneli (multi-granularity) erişim | `BAAI/bge-m3` (Dense + Sparse + Multi-vector, 8192 token bağlam) | MIRACL, MKQA, MLDR (uzun belge erişimi) | Uzun ve karmaşık sorgularda yüksek boyutlu (1024-d) çok dilli temsil kapasitesiyle güçlü performans, ancak yüksek hesaplama/bellek maliyeti. | Projemizde yüksek kapasiteli çok dilli referans model (`RQ1`, `RQ4`). |
| **Qwen3-Embedding** | 2025 | Yönerge-duyarlı (instruction-aware) yeni nesil LLM tabanlı gömme modeli | `Qwen/Qwen3-Embedding-0.6B` (`Instruct: ...\nQuery: ...` formatı) | Çok dilli MTEB ve alan-özgü erişim görevleri | Görev yönergesi (`sorgu_yonergesi`) verildiğinde kurumsal/hukuki alan erişiminde küçük parametre boyutuna rağmen yüksek doğruluk. | Yeni nesil instruction-aware embedding referansı (`RQ1`, `RQ4`). |
| **Es et al. (RAGAS)** | EACL 2024 | RAG sistemlerinin referanssız ve referanslı otomatik değerlendirilmesi | LLM-as-a-Judge (`faithfulness`, `answer_correctness`) | Soru-bağlam-yanıt üçlüleri | Erişim (retrieval) başarısı yüksek olsa bile üretim katmanında halüsinasyon veya eksik sentez oluşabilir; sadakat (faithfulness) ayrı ölçülmelidir. | **RQ3 / H3** hipotezimizin (retrieval artışı yanıt doğruluğuna doğrusal yansımaz) değerlendirme çerçevesi. |

---

## 2. Araştırma Soruları ve Hipotez Eşlemesi

- **RQ1 (H1):** Farklı embedding modelleri Türkçe anlamsal erişimde istatistiksel olarak anlamlı performans farkı gösterir (`eslestirilmis_bootstrap_fark`, $p < 0.05$).
- **RQ2 (H2):** Türkçenin sondan eklemeli (agglutinative) yapısı (`bicimbilimsel` sorgular) ve yeniden ifade edilmiş (`paraphrase`) sorgular, sözcüksel BM25'te sert düşüşe yol açarken yoğun (dense) modellerde daha sınırlı kayıp yaratır.
- **RQ3 (H3):** Daha yüksek retrieval skoru (`Recall@5`, `nDCG@10`), LLM yanıt doğruluğunu ve sadakatini (`faithfulness`) her zaman doğrusal olarak artırmaz; belirli bir eşikten sonra doygunluk görülür.
- **RQ4:** Embedding boyutu, indeksleme süresi ve sorgu gecikmesi (`ms`) ile erişim kalitesi arasındaki Pareto sınırı analiz edilir.
- **RQ5 (H4 - Özgün Katkı):** Sorgu uzunluğu, soru tipi ve BM25–Dense skor farkı gibi hafif özelliklerle eğitilen `ModelSecici` (router), tek bir modele kıyasla daha iyi kalite–maliyet dengesi sağlar.

