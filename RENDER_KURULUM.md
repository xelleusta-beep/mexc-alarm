# Render Kurulumu

## Mimari (Önerilen)

| Servis | Nerede | Ne yapar |
|---|---|---|
| **Streamlit panel** | Render (web) | Backtest, grafik, analiz |
| **Alarm botu** | GitHub Actions | 15 dk'da bir sinyal + Telegram |

> Alarm durumu (kasa, pozisyon) GitHub'da `alarm_config.json` içinde kalır — Render yeniden başlasa da kaybolmaz.

## 1) Render'a bağlan

1. https://render.com → GitHub ile giriş yap (ücretsiz plan yeterli)
2. **New → Blueprint**
3. Repo'yu seç: `xelleusta-beep/mexc-alarm`
4. `render.yaml` otomatik algılanır → **Apply**

## 2) Secret (isteğe bağlı)

Panelde Telegram kullanacaksan Environment → Add Environment Variable:

| Key | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | token |
| `TELEGRAM_CHAT_ID` | chat id |

Alarm botu zaten GitHub Secrets'tan okuyor — Render'a şart değil.

## 3) Alarm botu GitHub'da kalsın

`.github/workflows/alarms.yml` zaten ayarlı. Render'a worker ekleme —
dosya sistemi geçici, kasa sıfırlanır. Alarm için GitHub Actions yeterli.

## Ücretsiz plan notları

- Servis **15 dk idle** olursa uyur, istek gelince uyanır (~30 sn)
- Backtest için sorun yok; alarm GitHub'da kesintisiz çalışır
- Aylık 500 saat ücretsiz (panel gece zaten az kullanılır)

## Render + GitHub birlikte

```
Sen → Render panel (backtest/analiz)
GitHub Actions → her 15 dk alarm kontrolü → Telegram
```
