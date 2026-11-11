# Veri

## Korpus

| Dosya | Açıklama |
|---|---|
| `data/raw/yonetmelik.txt` | Doğuş Üniversitesi Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliği tam metni (46 madde). Birincil kurumsal korpus. |
| `data/raw/akademik_kurumsal_ek_dokumanlar.txt` | Çok alanlı (bilim, finans, hukuk, teknoloji, eğitim) ek Türkçe belge derlemi. |
| `data/raw/tr_mteb_tr_teb_gorevleri.jsonl` | Standart çok alanlı Türkçe erişim kıyaslama (benchmark) ham verisi. |

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

## Veri Seti Özellikleri

- Ana soru seti (`data/processed/sorular.jsonl`) 82 varyant grubunda **328 soru** (`156 train`, `52 val`, `120 test`) ve dört kategori (`olgusal`, `paraphrase`, `bicimbilimsel`, `cok-maddeli`) içerir.
- Her sorunun doğrulanmış kanıt chunk kimliği (`kanit_chunk_id`) ve referans yanıtı (`referans_yanit`) bulunur.
- Ek olarak **30 çok alanlı standart Türkçe benchmark sorusu** (`data/processed/standart_tr_benchmark.jsonl`) ve **16 cevaplanamaz kontrol sorusu** (`data/processed/kacinma_sorulari.jsonl`) yer alır.

## Klasörler

- `data/raw/` — özgün metinler (PDF Git'e girmez, txt girer)
- `data/processed/` — üretilmiş chunk/soru dosyaları (büyük indeksler Git'e girmez)
