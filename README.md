# CampUncharted Studio v2 — 7/24 Otonom YouTube Stüdyosu

**CampUncharted** (`@CampUncharted` · Kanal Kilidi: `UCwzbmfB6P-s7RrZqHA1JCqw`) için sıfırdan tasarlanmış, bilgisayarın kapalıyken bile **GitHub Actions** üzerinde 7/24 çalışan, yüksek izlenme ve abone dönüşümü odaklı yeni nesil YouTube üretim ve yayın hattı.

---

## v1'den v2'ye Neler Değişti?

| Alan | Eski Sistem (v1) | Yeni Stüdyo (v2) |
|---|---|---|
| **İçerik & Hikâye** | 16 sn, 24 kelimelik ansiklopedi listeleri | **35–48 sn**, 8–10 ritimli, ilk 2 sn merak kancalı **5 viral seri** + haftada **2 adet 6–10 dk sinematik Belgesel** |
| **Görsel Kurgu** | 6–11 sn sabit kareler, yatay videoyu dikeyde kırpma, MoviePy (>1 GB RAM) | **Saf FFmpeg (<250 MB RAM)**, dikey-öncelikli Pexels/Pixabay/Wikimedia, **2.5–3.8 sn** kesme hızı, yumuşak kamera hareketi ve renk derecelendirme |
| **Altyazı & Tipografi** | Linux'ta kırılan Windows font yolu (10px piksel yazı) | **Kelime-kelime altın vurgulu (karaoke) ASS altyazı**, Linux Roboto/DejaVu kalın fontlar, okunurluk gradyanı |
| **Ses & Atmosfer** | `-2%` yavaş ses + cümle arası +0.60 sn boşluk + korku filmi drone müziği | **Edge-TTS WordBoundary** (`+7%` Shorts / `+0%` Belgesel), sıkı sessizlik kesimi (`0.12 sn`), 5 doğa atmosferi (rüzgar/yağmur/kamp ateşi) + sidechain ducking + **-14 LUFS** |
| **Yükleme & Kota** | HTTP 308 hatası (45 yarım private video), dahili hash sızıntısı | **HTTP 308 fix**, kanal ID güvenlik kilidi, **peak UTC zamanlama**, kota-korumalı **5 dil altyazı/başlık (`en, es, tr, de, fr`)** |
| **Kendini Geliştirme** | Sabit 5 kamp alanı | Günlük **YouTube Analytics** döngüsü: izlenme süresi (AVP) ve abone getiren serilerin ağırlığını otomatik artırır |

---

## 7/24 Yayın Takvimi (Otomatik Cron)

1. **Shorts Factory** (`.github/workflows/shorts-factory.yml`) — **Günde 3 kez** (`07:00, 13:00, 19:00 UTC`)
   - En verimli küresel saatlere (`13:15, 18:15, 23:15 UTC`) otomatik zamanlar.
   - 5 seri rotasyonu: `CAMP_MISTAKES`, `WILD_STAYS`, `GEAR_AND_HACKS`, `EXTREME_WEATHER_NATURE`, `TURKEY_AND_EURASIA_HIDDEN`.
2. **Documentary Factory** (`.github/workflows/documentary-factory.yml`) — **Haftada 2 kez** (Çarşamba & Cumartesi `11:00 UTC` → `16:00 UTC` yayın)
   - 6 bölümlü, 1920×1080 sinematik belgesel, tıklanabilir YouTube bölüm zaman damgaları, özel 1280×720 kapak görseli ve otomatik oynatma listesi ekleme.
3. **Analytics Optimizer & Multi-Language Backfill** (`.github/workflows/analytics-optimizer.yml`) — **Günde 1 kez** (`05:00 UTC`)
   - Son 30 günlük YouTube Analytics verisini çeker, seri ağırlıklarını günceller, bekleyen ilk yorumları ve 5 dilli altyazı/yerelleştirmeleri günlük kota sınırını (`1.800/10.000`) aşmadan tamamlar.

---

## 3 Dakikada Aktivasyon (GitHub Secrets)

Repo ayarlarına git: **Settings → Secrets and variables → Actions → New repository secret**

Aşağıdaki 4 zorunlu secret'ı ekle:

| Secret Adı | Nereden Alınır / İçeriği |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API anahtarın |
| `PEXELS_API_KEY` | Pexels API anahtarın |
| `PIXABAY_API_KEY` | Pixabay API anahtarın |
| `YOUTUBE_TOKEN_JSON` | Bilgisayarındaki mevcut `token.json` dosyasının tam JSON içeriği (`refresh_token`, `client_id`, `client_secret` içermeli) |

*(İsteğe bağlı)* `YOUTUBE_CLIENT_SECRET_JSON`: Eğer `token.json` içinde `client_id`/`client_secret` eksikse `client_secret.json` içeriğini buraya ekleyebilirsin.

> **ÖNEMLİ — OAuth Token'ın 7 Günde Bir Süresinin Dolmaması İçin:**
> [Google Cloud Console → APIs & Services → OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent) ekranında uygulama durumu **"Testing"** ise Google `refresh_token`'ları 7 günde bir iptal eder. Durumu **"Publish App" (In Production)** yaparsan (kendi kanalın için doğrulama beklemeden kullanılabilir) `YOUTUBE_TOKEN_JSON` aylarca kesintisiz çalışır!

PC'den token'ı kontrol etmek veya yenilemek istersen:
```bash
python scripts/export_github_secrets.py --token /yol/token.json
# Yeniden yetkilendirme gerekirse:
python scripts/export_github_secrets.py --reauth --client-secret /yol/client_secret.json --token token.json
```

---

## İlk Testi Güvenle Çalıştırma

1. **Bağlantı & Kanal Kilidi Testi (20 sn):**
   GitHub'da **Actions → Analytics Optimizer & Subtitle Backfill → Run workflow** → `verify_only: true` seç → Çalıştır.
   Logda `[AUTH OK] Channel 'CampUncharted' (UCwzbmfB6P-s7RrZqHA1JCqw)` gördüğünde yetki tamamdır!
2. **YouTube'a Yüklemeden İlk Video Önizlemesi (Dry Run):**
   **Actions → Shorts Factory → Run workflow** → `dry_run: true` seç → Çalıştır.
   Bitince sayfanın altındaki **`short-output-...`** artifact'ından üretilen `.mp4` ve `.jpg` kapağı indirip kalitesini kendin izleyebilirsin.
3. **Canlıya Geçiş:**
   Beğendiğinde hiçbir şeye dokunmana gerek yok — zamanlanmış görevler günde 3 Short + haftada 2 Belgesel üretip otomatik yayınlayacak! (İstediğin an `workflow_dispatch` ile özel konu verip anında video da üretebilirsin.)

---

## Yerel Geliştirme & Test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
pip install --no-deps -e .
pytest -v
```
