# İP-1 & İP-2: Veri Planı ve Kalite Güvence Protokolü

## 1. Birincil Kurumsal Korpus

- **Kaynak:** Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (`data/raw/yonetmelik.txt`).
- **Kapsam:** 5 Bölüm, 46 Madde (~47.600 karakter).
- **Parçalama (Chunking) Stratejisi:**
  - Önce `MADDE n` başlıklarına göre ayrılır (`src/data.py -> yonetmelik_parcala`).
  - `800` karakteri aşan maddeler, cümle sınırlarını koruyarak `800` karakter pencere ve `100` karakter örtüşme (`ortusme_karakter: 100`) ile alt parçalara (`MADDE-38-1`, `MADDE-38-2` vb.) bölünür.
  - Toplam **82 chunk** (`data/processed/chunklar.jsonl`) üretilir; 46 maddenin tamamı eksiksiz temsil edilir.

---

## 2. Soru Seti Tasarımı (`data/processed/sorular.jsonl`)

### 2.1. Soru Kategorileri (`tip`)
1. **`olgusal`:** Yönetmelik maddesindeki açık bir kuralı, süreyi, AKTS kredisini veya oranı doğrudan soran sorular.
2. **`paraphrase`:** Aynı bilgiyi yönetmelikteki sözcükleri kullanmadan, eş anlamlı veya günlük öğrenci diliyle yeniden ifade eden sorular (RQ2).
3. **`bicimbilimsel`:** Türkçenin sondan eklemeli yapısını (iyelik, hal, zaman, olumsuzluk ve ettirgenlik/edilgenlik ekleri) yoğun kullanan çekimli sorgular (RQ2).
4. **`cok-maddeli`:** Birden fazla fıkra/koşul veya süreç adımını bir arada sorgulayan karmaşık sorular (RQ2, RQ3, RQ5).

### 2.2. Veri Bölmesi (`bolme`) ve Sızıntı Önleme Disiplini
- Her temel soru ve onun varyantları aynı `varyant_grup` (`G001`, `G002`, ...) kimliğini paylaşır.
- **Sıfır Sızıntı Kuralı:** Aynı `varyant_grup` altındaki tüm sorular **yalnızca tek bir bölmede** (`train`, `val` veya `test`) yer alır.
- **Bölme Oranları:**
  - `train` (~%50): Router (`ModelSecici`) eğitimi için kullanılır.
  - `val` (~%15): Router hiperparametre ve özellik seçimi için kullanılır.
  - `test` (~%35): Tüm retrieval (`02_retrieval_deneyleri.ipynb`), RAG üretim (`03_rag_uretim.ipynb`) ve nihai router (`04_model_secici.ipynb`) karşılaştırmalarında kullanılır.
