# Katkı Kuralları

5 araştırmacı aynı kod tabanında çalışır. Aşağıdaki düzen, deneylerin tekrarlanabilirliği ve sonuçların birleştirilebilmesi için zorunludur.

## Branch stratejisi

- `main` her zaman çalışır durumda kalır; doğrudan push yapılmaz.
- Her iş için kısa ömürlü branch: `ip2/soru-seti`, `ip3/bge-m3-indeks`, `fix/ndcg-hesabi` gibi.
- İş bitince Pull Request açılır, en az bir ekip arkadaşı gözden geçirir, sonra `main`'e alınır.

## Commit mesajları

Kısa ve açıklayıcı, Türkçe veya İngilizce (birini seçip tutarlı olun):

- `feat(retrieval): BM25 retriever ve birim testi eklendi`
- `exp(ip3): e5-base recall sonuçları (tohum 42-46)`
- `fix(metrics): nDCG kazanç hesabı düzeltildi`

## Deney disiplini (ihlal = sonuç geçersiz)

1. Deney ayarları yalnızca `configs/deney.yaml` üzerinden değiştirilir; kod içine sabit değer gömmeyin.
2. Her çalışma: konfigürasyon kopyası + tohum + model sürümü + sonuç CSV'si ile birlikte `results/` altına yazılır.
3. Aynı sorunun paraphrase varyantları train/val/test arasında bölünmez.
4. Router (model seçici) yalnızca train/val verisiyle eğitilir; test sonuçları seçimde kullanılmaz.
5. Farklı embedding modellerinin vektörleri aynı FAISS indeksinde karıştırılmaz; model başına ayrı indeks.

## Kod

- Notebook'lar ince kalır; tekrar kullanılan her şey `src/` altında modül olur.
- Yeni fonksiyon yazarken metrik/veri formatını değiştiriyorsanız ilgili testi de güncelleyin.
- Büyük dosyalar (model ağırlığı, FAISS indeksi, ham PDF) Git'e girmez; `.gitignore` bunu düzenler. Sonuç özetleri (küçük CSV/JSON) girer.

## Pull Request kontrol listesi

- [ ] Konfigürasyon değişikliği varsa `deney.yaml` güncellendi
- [ ] Sonuç ürettiyse `results/` altına CSV + çalıştırma notu eklendi
- [ ] README veya `data/README.md` etkilendiyse güncellendi
