"""
MEXC Alarm Daemon - Streamlit'siz 24/7 çalışır.
Yerel:    python alarms_daemon.py          (sürekli döngü)
Tek sefer:python alarms_daemon.py --once   (GitHub Actions için)
Durdur:   Ctrl+C
Ayarlar:  alarm_config.json (Telegram, kaldıraç, alarm listesi)
Telegram: env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID öncelikli
"""
import json
import time
import sys
import os
from datetime import datetime, timezone

import ccxt
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "alarm_config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if env_token:
        cfg["telegram_bot_token"] = env_token
    if env_chat:
        cfg["telegram_chat_id"] = env_chat
    return cfg


def save_config(cfg):
    cfg_to_save = dict(cfg)
    if os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        cfg_to_save["telegram_bot_token"] = ""
    if os.environ.get("TELEGRAM_CHAT_ID", "").strip():
        cfg_to_save["telegram_chat_id"] = ""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_to_save, f, ensure_ascii=False, indent=2)


def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        print("[WARN] Telegram token/chat_id bos - mesaj gonderilemedi")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] Telegram HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Telegram hatasi: {e}")
        return False


def get_crypto_data(symbol, limit, timeframe):
    exchange = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    limit = min(max(int(limit), 30), 50000)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    all_candles = []
    since = exchange.milliseconds() - (limit * tf_ms)
    for _ in range(100):
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        new_since = batch[-1][0] + tf_ms
        if new_since == since:
            break
        since = new_since
        if len(all_candles) >= limit:
            break
    if not all_candles or len(all_candles) <= 20:
        return pd.DataFrame()
    all_candles = all_candles[-limit:]
    df = pd.DataFrame(all_candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df[~df.index.duplicated(keep="last")][["Open", "High", "Low", "Close", "Volume"]]


def predict_signal(df, train_ratio=80):
    w = df.copy()
    w["Return"] = w["Close"].pct_change()
    delta = w["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    w["RSI"] = 100 - (100 / (1 + rs))
    w["SMA_20"] = w["Close"].rolling(20).mean()
    w["Price_to_SMA"] = w["Close"] / w["SMA_20"]
    w["Signal_Target"] = (w["Close"].shift(-1) > w["Close"]).astype(int)
    w.dropna(inplace=True)
    if len(w) < 30:
        return 0

    features = ["Return", "RSI", "Price_to_SMA"]
    X = w[features]
    y = w["Signal_Target"]

    n_splits = min(10, len(w) // 30)
    min_train = max(int(len(X) * train_ratio / 100), 30)
    test_size = max((len(X) - min_train) // n_splits, 5)
    w["Predicted_Signal"] = 0

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, len(X))
        if test_start >= len(X):
            break
        model = RandomForestClassifier(
            random_state=42, n_estimators=200, max_depth=10,
            min_samples_split=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1,
        )
        model.fit(X[:train_end], y[:train_end])
        preds = model.predict(X.iloc[test_start:test_end])
        w.iloc[test_start:test_end, w.columns.get_loc("Predicted_Signal")] = preds

    return int(w["Predicted_Signal"].iloc[-1])


def candle_closed(last_ts, interval):
    tf = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}.get(interval, pd.Timedelta(days=1))
    close_at = last_ts + tf
    if close_at.tzinfo is None:
        close_at = close_at.tz_localize("UTC")
    return pd.Timestamp.now(tz="UTC") >= close_at


def period_candles_for(interval, period="1y"):
    unit_factor = {
        "7d": 7, "30d": 30, "2mo": 60, "1y": 365, "2y": 730, "3y": 1095,
    }
    days = unit_factor.get(period, 365)
    per = 24 if interval == "1h" else 1
    return max(days * per, 100)


def process_alarm(alarm, cfg):
    leverage = int(cfg.get("leverage", 1))
    train_ratio = int(cfg.get("train_ratio", 80))
    token = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")

    interval = alarm.get("interval", "1d")
    ticker = alarm["ticker"]
    limit = alarm.get("period_candles") or period_candles_for(interval, alarm.get("period", "1y"))

    df = get_crypto_data(ticker, limit, interval)
    if df.empty or len(df) < 10:
        print(f"  [{ticker}] veri yok")
        return

    last_ts = df.index[-1]
    if not candle_closed(last_ts, interval):
        return
    if alarm.get("last_candle_ts") == str(last_ts):
        return

    signal = predict_signal(df, train_ratio)
    price = float(df["Close"].iloc[-1])
    alarm["last_candle_ts"] = str(last_ts)
    alarm["last_price"] = price

    if alarm.get("last_signal") == signal:
        save_config(cfg)
        return

    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    balance = float(alarm.get("balance", 1000.0))
    crypto_amount = float(alarm.get("crypto_amount", 0.0))
    entry_price = float(alarm.get("entry_price", 0.0))
    margin = float(alarm.get("margin", 0.0))
    action = ""

    if signal == 1 and balance > 0:
        notional = balance * leverage
        alarm["margin"] = balance
        alarm["crypto_amount"] = notional / price
        alarm["entry_price"] = price
        alarm["balance"] = 0.0
        action = (
            f"🟢 *ALIM SİNYALİ*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Fiyat:* `${price:,.4f}`\n"
            f"⚡ *Kaldıraç:* `{leverage}x`\n"
            f"💵 *Teminat:* `${balance:,.2f}`\n"
            f"📊 *Pozisyon:* `${notional:,.2f}`\n"
            f"📈 *Miktar:* `{alarm['crypto_amount']:.6f}` `{ticker.split('/')[1]}`\n"
            f"🏦 *Kasa:* `$0.00`"
        )
        print(f"  [{ticker}] ALIM @ {price:.4f} lev={leverage}x pos=${notional:.2f}")
    elif signal == 0 and crypto_amount > 0:
        notional_exit = crypto_amount * price
        pnl_raw = notional_exit - (crypto_amount * entry_price)
        pnl_usd = pnl_raw * leverage
        pnl_pct = ((price - entry_price) / entry_price * 100 * leverage) if entry_price else 0.0
        new_balance = max(margin + pnl_usd, 0.0)
        action = (
            f"🔴 *SATIM SİNYALİ*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Fiyat:* `${price:,.4f}`\n"
            f"⚡ *Kaldıraç:* `{leverage}x`\n"
            f"📊 *Satılan:* `{crypto_amount:.6f}` `{ticker.split('/')[1]}`\n"
            f"💵 *Pozisyon:* `${notional_exit:,.2f}`\n"
            f"{'✅' if pnl_usd >= 0 else '❌'} *Kâr/Zarar:* `{pnl_usd:+.2f}$` (`{pnl_pct:+.2f}%`)\n"
            f"🏦 *Toplam Kasa:* `${new_balance:,.2f}`"
        )
        alarm["balance"] = new_balance
        alarm["crypto_amount"] = 0.0
        alarm["margin"] = 0.0
        print(f"  [{ticker}] SATIM @ {price:.4f} pnl={pnl_usd:+.2f}$ kasa={new_balance:.2f}")
    elif signal == 1:
        action = (
            f"🟢 *SİNYAL GÜNCELLENDİ*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Fiyat:* `${price:,.4f}`\n"
            f"⚡ *Kaldıraç:* `{leverage}x`\n"
            f"🏦 *Kasa:* `${balance:,.2f}`\n"
            f"📊 *Pozisyon:* `{crypto_amount:.6f}`"
        )
        print(f"  [{ticker}] SINYAL=1 (pozisyon yok)")
    else:
        action = (
            f"🔴 *SİNYAL GÜNCELLENDİ*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Fiyat:* `${price:,.4f}`\n"
            f"⚡ *Kaldıraç:* `{leverage}x`\n"
            f"🏦 *Kasa:* `${balance:,.2f}`"
        )
        print(f"  [{ticker}] SINYAL=0")

    alarm["last_signal"] = signal

    if action:
        msg = (
            f"⚡ *MEXC ALARM* ⚡\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🪙 *Parite:* `{ticker}`\n"
            f"⏱ *Aralık:* `{interval}`\n"
            f"📅 *Tarih:* `{now_str}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{action}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏦 *Portföy:* `${alarm.get('balance', 0):,.2f}`"
        )
        send_telegram(token, chat_id, msg)

    save_config(cfg)


def run_once():
    cfg = load_config()
    alarms = [a for a in cfg.get("alarms", []) if a.get("is_active", True)]
    if not alarms:
        print("Aktif alarm yok.")
        return
    print(f"{len(alarms)} alarm kontrol ediliyor...")
    for alarm in alarms:
        try:
            process_alarm(alarm, cfg)
        except Exception as e:
            print(f"  [HATA] {alarm.get('ticker')}: {e}")
    save_config(cfg)
    print("Bitti.")


def main():
    if "--once" in sys.argv:
        run_once()
        return

    print("=" * 50)
    print("MEXC Alarm Daemon basliyor...")
    print(f"Config: {CONFIG_PATH}")
    print("Durdurmak icin: Ctrl+C")
    print("=" * 50)

    while True:
        try:
            cfg = load_config()
            alarms = [a for a in cfg.get("alarms", []) if a.get("is_active", True)]
            interval_sec = int(cfg.get("check_interval_sec", 30))

            if not alarms:
                print(f"[{datetime.now(timezone.utc):%H:%M:%S}] Aktif alarm yok - beklemede...")
                time.sleep(interval_sec)
                continue

            print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {len(alarms)} alarm kontrol ediliyor...")
            for alarm in alarms:
                try:
                    process_alarm(alarm, cfg)
                except Exception as e:
                    print(f"  [HATA] {alarm.get('ticker')}: {e}")

            save_config(cfg)
            time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("\nDaemon durduruldu.")
            sys.exit(0)
        except FileNotFoundError:
            print("[HATA] alarm_config.json bulunamadi!")
            sys.exit(1)
        except Exception as e:
            print(f"[HATA] {e}")
            time.sleep(interval_sec)


if __name__ == "__main__":
    main()
