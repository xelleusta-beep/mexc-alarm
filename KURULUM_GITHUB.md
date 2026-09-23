# MEXC Alarm - Ücretsiz 7/24 Kurulum (GitHub Actions)

Bilgisayarın kapalıyken bile Telegram'a sinyal gönderir. **Tamamen ücretsiz**, kredi kartı istemez.

## 1) GitHub hesabı + repo aç

1. https://github.com/signup ile hesap aç (varsa atla)
2. Yeni repo oluştur: **Public** seç (ücretsiz sınırsız dakika)
   - Repo adı örn: `mexc-alarm`
   - Public seçili olsun
3. Bu klasördeki dosyaları push et (aşağıdaki komutlar)

```powershell
cd "D:\Ahmet Proje Dosyaları\kripto"
git init
git add .
git commit -m "MEXC alarm bot"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADI/mexc-alarm.git
git push -u origin main
```

> `KULLANICI_ADI` ve repo adını kendi bilgilerinle değiştir.

## 2) Telegram Secrets ayarla

1. Repo sayfası → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret** ile ikisini ekle:

| Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | BotFather'dan aldığın token |
| `TELEGRAM_CHAT_ID` | Kendi chat id'n |

> Chat id bulmak için: `@userinfobot`'a mesaj at.

## 3) Alarm ekle

Repo'daki `alarm_config.json` dosyasını düzenleyip push et:

```json
{
  "telegram_bot_token": "",
  "telegram_chat_id": "",
  "leverage": 10,
  "train_ratio": 80,
  "check_interval_sec": 30,
  "alarms": [
    {
      "id": 1,
      "ticker": "BTC/USDT",
      "interval": "1d",
      "period": "1y",
      "period_candles": 365,
      "balance": 1000,
      "crypto_amount": 0,
      "margin": 0,
      "entry_price": 0,
      "last_signal": 0,
      "last_candle_ts": null,
      "last_price": 0,
      "is_active": true
    }
  ]
}
```

Token/chat_id'yi **boş bırak** — GitHub Secrets'tan gelir, repo'ya yazma.

## 4) Çalıştır

- **Actions** sekmesi → `MEXC Alarm Kontrol` → **Run workflow** ile test et
- Bundan sonra her **15 dakikada bir** otomatik çalışır
- Mum kapanınca sinyal varsa Telegram'a bildirim gider
- İşlem durumu (kasa, pozisyon) `alarm_config.json`'a yazılır ve repo'ya kaydedilir

## Yerel test

```powershell
.\venv\Scripts\python.exe alarms_daemon.py --once
```

## Notlar

- GitHub Actions cron'lar bazen 1-5 dk gecikmeli başlar — 1G mumlar için sorun yok
- Private repo seçersen aylık 2000 ücretsiz dakika yeter (günde ~96 çalıştırma)
- Streamlit paneli sadece backtest/analiz için; alarm botu GitHub'da çalışır
