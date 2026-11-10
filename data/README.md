# Veri

## Korpus

| Dosya | Açıklama |
|---|---|
| `data/raw/yonetmelik.txt` | Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği tam metni (46 madde). Birincil kurumsal korpus. |

Chunk'lar madde/fıkra sınırlarına saygılı üretilir; her chunk kaynağını taşır (`madde_no`).

## Soru seti şeması (`sorular.jsonl`)

Her satır bir JSON nesnesi:

```json
{
  "soru_id": "S001",
  "soru": "Öğrencinin mezuniyet koşulları nelerdir?",
  "tip": "olgusal",
  "kanit_chunk_id": "MADDE-38",
  "referans_yanit": "Mezuniyet için ...",
  "varyant_grup": "G012",
  "bolme": "test"
}
```

| Alan | Anlam |
|---|---|
| `soru_id` | Benzersiz kimlik |
| `tip` | `olgusal` · `paraphrase` · `bicimbilimsel` · `cok-maddeli` |
| `kanit_chunk_id` | Doğru cevabı içeren chunk (qrels karşılığı) |
| `varyant_grup` | Aynı temel sorudan türeyen varyantların grup kimliği |
| `bolme` | `train` · `val` · `test` — aynı `varyant_grup` tek bölmede kalır |

## Kurallar

- Hedef: 300–500 özgün soru + paraphrase/bičimbilimsel varyantları.
- Her sorunun doğrulanmış kanıt chunk'ı ve mümkünse referans yanıtı olur.
- Standart benchmark: TR-MTEB / TR-TEB retrieval görevleri Hugging Face `datasets` ile indirilir; buraya kopyalanmaz.

## Klasörler

- `data/raw/` — özgün metinler (PDF Git'e girmez, txt girer)
- `data/processed/` — üretilmiş chunk/soru dosyaları (büyük indeksler Git'e girmez)
