# CampUncharted Studio v2 — Devam / Handoff Özeti

Bu dosya, farklı bir hesaptan veya yeni bir oturumdan kaldığımız yerden tek adımda devam edilebilmesi için kaydedilmiştir.

## 1. Şu Ana Kadar Tamamlananlar (%100 Kod + State + Otomasyon Hazır)

- **Hedef Kanal:** `CampUncharted` (`@CampUncharted`, Kanal ID kilidi: `UCwzbmfB6P-s7RrZqHA1JCqw`).
- **Özel GitHub Deposu:** `https://github.com/MFTTT11/campuncharted-studio` (branch: `main`).
- **Sıfırdan Yazılan v2 Stüdyo Paketi (`src/campuncharted/`):**
  - `config.py`: Kanal kilidi, peak UTC yayın slotları (`13:15, 18:15, 23:15 UTC` Shorts; `16:00 UTC` Belgesel), Gemini non-lite model önceliği, Edge-TTS sesleri (`+7%` Shorts / `+0%` Belgesel), kota bütçeleri (`1800/10000` dil kotası).
  - `state_store.py`: Atomik JSON kalıcı hafıza (`state/`).
  - `topic_catalog.py`: 5 dönen viral seride **44 küratörlü Short konusu** (`CAMP_MISTAKES`, `WILD_STAYS`, `GEAR_AND_HACKS`, `EXTREME_WEATHER_NATURE`, `TURKEY_AND_EURASIA_HIDDEN`) + **12 adet 6–10 dk Belgesel** + Analytics ağırlıklı seri rotasyonu.
  - `fact_researcher.py`: Wikipedia API + resmi park kaynaklı, 24 saat önbellekli doğrulayıcı.
  - `content_engine.py`: Pydantic şemalı Gemini senarist (8–10 ritimli, ilk 2 sn kancalı viral Shorts; 6 bölümlü Belgesel; dahili hash içermeyen temiz açıklamalar + çok dilli toplu çeviri).
  - `media_engine.py`: Dikey-öncelikli (`orientation=portrait`) Pexels, HTTP-400 düzeltmeli Pixabay sorgu temizleyici, Wikimedia Commons CC0/CC-BY, FFprobe kalite kontrolü ve tekrar koruması.
  - `audio_engine.py`: Edge-TTS `WordBoundary` kelime zamanlaması, FFmpeg sıkı sessizlik kesimi (`0.12 sn`), 5 doğa atmosferi (pure numpy stereo rüzgar/yağmur/kamp ateşi), FFmpeg sidechain ducking + `-14 LUFS` loudnorm.
  - `subtitle_engine.py`: Kelime-kelime altın vurgulu (karaoke) 9:16 ASS altyazı + 16:9 Belgesel bölüm kartları + SRT üretici (Linux Roboto/DejaVu kalın font otomatik algılama).
  - `video_engine.py`: Saf FFmpeg (<250 MB RAM, MoviePy yok), 2.5–3.8 sn hızlı kesme, yumuşak kamera hareketi, okunurluk gradyanı, otomatik görsel + ses QC (`verify_rendered_video`).
  - `thumbnail_engine.py`: En yüksek görsel enerjili kareyi seçen, kalın TrueType fontlu 1280×720 ve 1080×1920 JPEG kapak üreticisi.
  - `youtube_engine.py`: HTTP 308 fix (`redirect_codes -= {308}`), kanal ID kilidi, uzak tekrar kontrolü, peak UTC zamanlama, kota-korumalı 5 dil (`en, es, tr, de, fr`) altyazı/başlık yerelleştirme + gecikmeli ilk yorum.
  - `analytics_engine.py`: 30 günlük YouTube Analytics çekimi + AVP/abone performansına göre seri ağırlığı güncelleme (`[0.08, 0.45]`).
  - `cli.py`: `produce-short`, `produce-doc`, `optimize-analytics`, `backfill-subtitles`, `verify-auth`, `migrate-state`.
- **v1 Geçmişi Temiz Şekilde Aktarıldı (`state/`):**
  - 225 önceki lokasyon (`uploaded_locations.json`), 1.126 kullanılmış medya ID (`used_media_ids.json`), 229 içerik özeti (`content_hashes.json`), 10 eski belgesel (`documentary_archive.json` — tüm Windows dosya yolları temizlendi), v1 başlangıç analitiği (`channel_insights.json`).
  - Sıfır gizli anahtar, sıfır token, sıfır Windows dosya yolu.
- **GitHub Actions 7/24 Otomasyonları (`.github/workflows/`):**
  - `shorts-factory.yml`: Günde 3 Short (`07:00, 13:00, 19:00 UTC`) + manuel `dry_run / series / topic / publish_now`.
  - `documentary-factory.yml`: Haftada 2 Belgesel (Çarşamba & Cumartesi `11:00 UTC`).
  - `analytics-optimizer.yml`: Günlük `05:00 UTC` Analytics + çok dilli altyazı + manuel `verify_only` auth testi.
  - `ci-tests.yml`: Her push/PR'da `pytest -v`.

---

## 2. Kalan Tek Adım (Kullanıcı Tarafında 3 Dakika — Canlıya Alma)

Sistemin 7/24 otomatik video üretip yüklemeye başlaması için sadece şu 3 adım kaldı:

1. **GitHub Secrets Ekleme** (`https://github.com/MFTTT11/campuncharted-studio/settings/secrets/actions`):
   - `GEMINI_API_KEY`
   - `PEXELS_API_KEY`
   - `PIXABAY_API_KEY`
   - `YOUTUBE_TOKEN_JSON` (PC'deki `token.json` içeriği; kontrol/yenileme için `scripts/export_github_secrets.py` hazır)
   - *(Opsiyonel)* `YOUTUBE_CLIENT_SECRET_JSON`
2. **Google Cloud OAuth Consent Screen → "In Production":**
   - Token'ın 7 günde bir düşmemesi için OAuth uygulama durumunu **Publish App (In Production)** yap.
3. **İlk Test & Canlıya Geçiş:**
   - GitHub Actions → `Analytics Optimizer & Subtitle Backfill` → `verify_only: true` ile `[AUTH OK]` doğrula.
   - GitHub Actions → `Shorts Factory` → `dry_run: true` ile ilk videoyu yüklemeden önizle.
   - Beğenince zamanlanmış cron 7/24 otomatik çalışmaya devam eder!

---

## 3. Yeni Hesapta Devam Etmek İçin Yapıştırabileceğin Kısa Prompt

Yeni hesapta sohbete başlarken sadece şunu yazabilirsin:
> `MFTTT11/campuncharted-studio` reposundaki `HANDOFF.md` dosyasını oku, kaldığımız yerden devam edelim.
