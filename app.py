import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import ccxt
import warnings
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Yapay Zeka Çoklu Otomasyon Paneli")

st_autorefresh(interval=30 * 1000, key="keepalive")

import json as _json
import os as _os
_ALARM_CONFIG_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "alarm_config.json")

def _load_alarm_config():
    try:
        with open(_ALARM_CONFIG_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {"telegram_bot_token": "", "telegram_chat_id": "", "leverage": 1, "train_ratio": 80, "check_interval_sec": 30, "alarms": []}

def _save_alarm_config(cfg):
    try:
        with open(_ALARM_CONFIG_PATH, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "alarms" not in st.session_state:
    _cfg = _load_alarm_config()
    st.session_state.alarms = _cfg.get("alarms", [])
if "global_trade_history" not in st.session_state:
    st.session_state.global_trade_history = []

# --- 1. TELEGRAM BİLDİRİM FONKSİYONU ---
def send_telegram_signal(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

# --- 2. GÜVENLİ VERİ ÇEKME FONKSİYONU ---
def get_crypto_data(symbol_name, safe_limit, inv_str):
    try:
        exchange = ccxt.mexc({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        safe_limit = min(max(int(safe_limit), 30), 50000)

        timeframe_ms = exchange.parse_timeframe(inv_str) * 1000
        all_candles = []
        since = exchange.milliseconds() - (safe_limit * timeframe_ms)

        for _ in range(100):
            batch = exchange.fetch_ohlcv(symbol_name, timeframe=inv_str, since=since, limit=1000)
            if not batch:
                break
            all_candles.extend(batch)
            new_since = batch[-1][0] + timeframe_ms
            if new_since == since:
                break
            since = new_since
            if len(all_candles) >= safe_limit:
                break

        if all_candles and len(all_candles) > 20:
            all_candles = all_candles[-safe_limit:]
            df_res = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_res['Timestamp'] = pd.to_datetime(df_res['Timestamp'], unit='ms')
            df_res.set_index('Timestamp', inplace=True)
            df_res = df_res[~df_res.index.duplicated(keep='last')]
            return df_res[['Open', 'High', 'Low', 'Close', 'Volume']]
        return pd.DataFrame()
    except Exception as e:
        if "global_trade_history" in st.session_state:
            st.session_state.global_trade_history.append(f"⚠️ Veri Çekme Hatası ({symbol_name}): {str(e)}")
        return pd.DataFrame()

# --- 3. PANDAS BACKTEST MATEMATİK MOTORU (Walk-Forward) ---
def compute_strategy_performance(df_input, train_ratio, init_cash=10000.0, walk_forward=True, leverage=1, trade_margin=None, n_estimators=200, max_depth=10, min_samples_leaf=5, min_samples_split=10):
    try:
        from sklearn.ensemble import RandomForestClassifier
        working_df = df_input.copy()
        working_df['Return'] = working_df['Close'].pct_change()

        delta = working_df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        working_df['RSI'] = 100 - (100 / (1 + rs))

        working_df['SMA_20'] = working_df['Close'].rolling(window=20).mean()
        working_df['Price_to_SMA'] = working_df['Close'] / working_df['SMA_20']

        working_df['Signal_Target'] = (working_df['Close'].shift(-1) > working_df['Close']).astype(int)
        working_df.dropna(inplace=True)

        if len(working_df) < 30:
            return None, 0.0, init_cash, pd.DataFrame(), 0

        features = ['Return', 'RSI', 'Price_to_SMA']
        X = working_df[features]
        y = working_df['Signal_Target']

        if walk_forward and len(working_df) >= 60:
            n_splits = min(10, len(working_df) // 30)
            min_train = int(len(X) * (train_ratio / 100))
            min_train = max(min_train, 30)
            test_size = (len(X) - min_train) // n_splits
            test_size = max(test_size, 5)

            working_df['Predicted_Signal'] = 0
            working_df['Confidence'] = 0.0

            for fold in range(n_splits):
                train_end = min_train + fold * test_size
                test_start = train_end
                test_end = min(test_start + test_size, len(X))
                if test_start >= len(X):
                    break
                model = RandomForestClassifier(
                    random_state=42, n_estimators=n_estimators,
                    max_depth=max_depth if max_depth and max_depth > 0 else None, min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf, class_weight='balanced',
                    n_jobs=-1
                )
                model.fit(X[:train_end], y[:train_end])
                proba = model.predict_proba(X.iloc[test_start:test_end])
                conf = proba.max(axis=1)
                preds = model.predict(X.iloc[test_start:test_end])
                working_df.iloc[test_start:test_end, working_df.columns.get_loc('Predicted_Signal')] = preds
                working_df.iloc[test_start:test_end, working_df.columns.get_loc('Confidence')] = conf
        else:
            split_idx = int(len(X) * (train_ratio / 100))
            if split_idx == 0 or split_idx >= len(X):
                split_idx = int(len(X) * 0.8)
            model = RandomForestClassifier(
                random_state=42, n_estimators=n_estimators,
                max_depth=max_depth if max_depth and max_depth > 0 else None, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X[:split_idx], y[:split_idx])
            proba = model.predict_proba(X)
            conf = proba.max(axis=1)
            preds = model.predict(X)
            working_df['Predicted_Signal'] = preds
            working_df['Confidence'] = conf
            min_train = split_idx

        trade_logs = []
        in_pos = False
        ent_price = 0.0
        ent_date = None
        cash = init_cash
        margin_size = float(trade_margin) if trade_margin and trade_margin > 0 else cash
        units = 0.0
        pos_low = float('inf')
        trade_cash_in = 0.0
        cash_at_entry = 0.0
        start_idx = min_train if walk_forward else min_train

        for i in range(start_idx, len(working_df)):
            c_date = working_df.index[i]
            c_price = float(working_df['Close'].iloc[i])
            c_low = float(working_df['Low'].iloc[i])
            c_sig = int(working_df['Predicted_Signal'].iloc[i])

            if c_sig == 1 and not in_pos and cash > 0.01 and margin_size > 0.01:
                trade_cash_in = min(margin_size, cash)
                cash_at_entry = cash
                units = (trade_cash_in / c_price) * 0.9998
                ent_price = c_price
                ent_date = c_date
                in_pos = True
                pos_low = c_low

            if in_pos:
                if c_low < pos_low:
                    pos_low = c_low
                if trade_cash_in > 0 and leverage > 0:
                    liq_drop = cash_at_entry / (trade_cash_in * leverage)
                    liq_price = ent_price * (1.0 - liq_drop) if liq_drop < 1.0 else 0.0
                    if liq_drop < 1.0 and pos_low <= liq_price:
                        cash = 0.0
                        ret_pct = ((liq_price - ent_price) / ent_price) * 100
                        max_dd_pct = ((pos_low - ent_price) / ent_price) * 100
                        max_dd_lev_pct = max_dd_pct * leverage
                        trade_size = trade_cash_in
                        max_dd_usd = (max_dd_pct / 100.0) * trade_size * leverage
                        trade_logs.append({
                            "İşlem ID": len(trade_logs) + 1,
                            "Yön": "🟢 AL → 💥 LİQ",
                            "Giriş Tarihi": ent_date.strftime('%Y-%m-%d %H:%M'),
                            "Çıkış Tarihi": c_date.strftime('%Y-%m-%d %H:%M'),
                            "Giriş Fiyatı ($)": round(ent_price, 4),
                            "Çıkış Fiyatı ($)": round(liq_price, 4),
                            "Toplam Kasa ($)": 0.0,
                            "Miktar ($)": f"+${trade_size:.2f}",
                            "Net Kâr/Zarar ($)": round(-cash_at_entry, 2),
                            "Getiri (%)": f"{ret_pct * leverage:.2f}%",
                            "Max Düşüş (%)": f"{max_dd_pct:.2f}%",
                            "Max Düşüş (Kaldıraçlı %)": f"{max_dd_lev_pct:.2f}%",
                            "Max Düşüş ($)": round(max_dd_usd, 2),
                            "Sonuç": "💥 Likide"
                        })
                        units = 0.0
                        in_pos = False
                        pos_low = float('inf')
                        margin_size = 0.0
                        continue

            if c_sig == 0 and in_pos:
                cash_raw = (units * c_price) * 0.9998
                pnl_raw = cash_raw - trade_cash_in
                pnl = pnl_raw * leverage
                cash = max(cash_at_entry + pnl, 0.0)
                margin_size = max(margin_size + pnl, 0.0)
                ret_pct = ((c_price - ent_price) / ent_price) * 100
                max_dd_pct = ((pos_low - ent_price) / ent_price) * 100
                max_dd_lev_pct = max_dd_pct * leverage
                trade_size = trade_cash_in
                max_dd_usd = (max_dd_pct / 100.0) * trade_size * leverage
                is_liq = cash <= 0.0 and pnl < 0
                trade_logs.append({
                    "İşlem ID": len(trade_logs) + 1,
                    "Yön": "🟢 AL → 🔴 SAT",
                    "Giriş Tarihi": ent_date.strftime('%Y-%m-%d %H:%M'),
                    "Çıkış Tarihi": c_date.strftime('%Y-%m-%d %H:%M'),
                    "Giriş Fiyatı ($)": round(ent_price, 4),
                    "Çıkış Fiyatı ($)": round(c_price, 4),
                    "Toplam Kasa ($)": round(cash, 2),
                    "Miktar ($)": f"+${trade_size:.2f}",
                    "Net Kâr/Zarar ($)": round(pnl, 2),
                    "Getiri (%)": f"{ret_pct * leverage:.2f}%",
                    "Max Düşüş (%)": f"{max_dd_pct:.2f}%",
                    "Max Düşüş (Kaldıraçlı %)": f"{max_dd_lev_pct:.2f}%",
                    "Max Düşüş ($)": round(max_dd_usd, 2),
                    "Sonuç": "💥 Likide" if is_liq else ("✅ Başarılı" if ret_pct > 0 else "❌ Başarısız")
                })
                units = 0.0
                in_pos = False
                pos_low = float('inf')
                if is_liq:
                    margin_size = 0.0

        c_last_price = float(working_df['Close'].iloc[-1])
        if in_pos:
            unreal_raw = (units * c_last_price) * 0.9998 - trade_cash_in
            final_val = max(cash_at_entry + unreal_raw * leverage, 0.0)
        else:
            final_val = cash
        total_ret_pct = ((final_val - init_cash) / init_cash) * 100
        latest_signal_out = int(working_df['Predicted_Signal'].iloc[-1])

        return working_df, total_ret_pct, final_val, pd.DataFrame(trade_logs), latest_signal_out
    except Exception as e:
        st.error(f"Hesaplama hatası: {str(e)}")
        return None, 0.0, init_cash, pd.DataFrame(), 0

@st.cache_data(ttl=600, show_spinner="5 dakikalık mumlar çekiliyor...")
def load_ohlcv_5m(symbol_name, safe_limit):
    return get_crypto_data(symbol_name, safe_limit, "5m")

def compute_5m_open_strategy(df_5m, init_cash=10000.0, leverage=1, trade_margin=None, doji_threshold=0.10, stop_mode="base", stop_mult=3.0, atr_frac=0.30):
    try:
        if df_5m is None or df_5m.empty or len(df_5m) < 30:
            return None, 0.0, init_cash, pd.DataFrame(), 0
        working_df = df_5m.copy()

        atr14 = None
        if stop_mode == "atr":
            dd = working_df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
            prev_c = dd["Close"].shift(1)
            tr = pd.concat([dd["High"] - dd["Low"], (dd["High"] - prev_c).abs(), (dd["Low"] - prev_c).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().shift(1)

        trade_logs = []
        cash = init_cash
        margin_size = float(trade_margin) if trade_margin and trade_margin > 0 else cash
        latest_signal = 0
        fee2 = 0.9998 * 0.9998

        day_groups = working_df.groupby(working_df.index.normalize())
        for day_key, day_df in day_groups:
            if len(day_df) < 2:
                continue
            first = day_df.iloc[0]
            f_open = float(first['Open'])
            f_high = float(first['High'])
            f_low = float(first['Low'])
            f_close = float(first['Close'])
            f_range = f_high - f_low
            if f_range <= 0:
                latest_signal = 0
                continue
            body = f_close - f_open
            if abs(body) / f_range < doji_threshold:
                latest_signal = 0
                continue
            direction = 1 if body > 0 else -1
            latest_signal = direction

            entry_row = day_df.iloc[1]
            ent_price = float(entry_row['Open'])
            if ent_price <= 0:
                continue
            if stop_mode == "atr":
                a_val = atr14.get(day_key, np.nan) if atr14 is not None else np.nan
                if pd.isna(a_val) or a_val <= 0:
                    continue
                sd = atr_frac * float(a_val)
                stop = ent_price - sd if direction == 1 else ent_price + sd
            elif stop_mode == "mult":
                d0 = ent_price - f_low if direction == 1 else f_high - ent_price
                stop = ent_price - stop_mult * d0 if direction == 1 else ent_price + stop_mult * d0
            else:
                stop = f_low if direction == 1 else f_high
            ent_date = entry_row.name

            if cash <= 0.01 or margin_size <= 0.01:
                continue

            trade_cash_in = min(margin_size, cash)
            cash_at_entry = cash
            liq_drop = cash_at_entry / (trade_cash_in * leverage) if leverage > 0 else 2.0
            if liq_drop < 1.0:
                liq_price = ent_price * (1.0 - liq_drop) if direction == 1 else ent_price * (1.0 + liq_drop)
            else:
                liq_price = None

            exit_price = None
            exit_date = None
            is_liq = False
            is_stop = False
            adverse = ent_price

            for ts, row in day_df.iloc[1:].iterrows():
                r_open = float(row['Open'])
                r_high = float(row['High'])
                r_low = float(row['Low'])
                if direction == 1:
                    adverse = min(adverse, r_low)
                    first_hit = stop
                    hit_liq = False
                    if liq_price is not None and liq_price > stop:
                        first_hit = liq_price
                        hit_liq = True
                    if r_low <= first_hit:
                        exit_price = min(first_hit, r_open)
                        exit_date = ts
                        is_liq = hit_liq
                        is_stop = not hit_liq
                        break
                else:
                    adverse = max(adverse, r_high)
                    first_hit = stop
                    hit_liq = False
                    if liq_price is not None and liq_price < stop:
                        first_hit = liq_price
                        hit_liq = True
                    if r_high >= first_hit:
                        exit_price = max(first_hit, r_open)
                        exit_date = ts
                        is_liq = hit_liq
                        is_stop = not hit_liq
                        break

            if exit_price is None:
                last_row = day_df.iloc[-1]
                exit_price = float(last_row['Close'])
                exit_date = last_row.name

            ratio = exit_price / ent_price
            if direction == 1:
                pnl_raw = trade_cash_in * fee2 * ratio - trade_cash_in
            else:
                pnl_raw = trade_cash_in * fee2 * (2.0 - ratio) - trade_cash_in
            pnl = pnl_raw * leverage

            if is_liq:
                cash = 0.0
                margin_size = 0.0
                pnl = -cash_at_entry
                cash_after = 0.0
            else:
                cash = max(cash_at_entry + pnl, 0.0)
                margin_size = max(margin_size + pnl, 0.0)
                cash_after = cash

            ret_pct = direction * (ratio - 1.0) * 100.0
            dd_pct = direction * (adverse - ent_price) / ent_price * 100.0
            dd_lev_pct = dd_pct * leverage
            dd_usd = (dd_pct / 100.0) * trade_cash_in * leverage

            if is_liq:
                yon = "🟢 LONG → 💥 LİQ" if direction == 1 else "🔴 SHORT → 💥 LİQ"
            elif is_stop:
                yon = "🟢 LONG → ⏹️ STOP" if direction == 1 else "🔴 SHORT → ⏹️ STOP"
            else:
                yon = "🟢 LONG → 🔴 GÜN SONU" if direction == 1 else "🔴 SHORT → 🟢 GÜN SONU"

            trade_logs.append({
                "İşlem ID": len(trade_logs) + 1,
                "Yön": yon,
                "Giriş Tarihi": ent_date.strftime('%Y-%m-%d %H:%M'),
                "Çıkış Tarihi": exit_date.strftime('%Y-%m-%d %H:%M'),
                "Giriş Fiyatı ($)": round(ent_price, 4),
                "Çıkış Fiyatı ($)": round(exit_price, 4),
                "Toplam Kasa ($)": round(cash_after, 2),
                "Miktar ($)": f"+${trade_cash_in:.2f}",
                "Net Kâr/Zarar ($)": round(pnl, 2),
                "Getiri (%)": f"{ret_pct * leverage:.2f}%",
                "Max Düşüş (%)": f"{dd_pct:.2f}%",
                "Max Düşüş (Kaldıraçlı %)": f"{dd_lev_pct:.2f}%",
                "Max Düşüş ($)": round(dd_usd, 2),
                "Sonuç": "💥 Likide" if is_liq else ("✅ Başarılı" if ret_pct > 0 else "❌ Başarısız")
            })

        total_ret_pct = ((cash - init_cash) / init_cash) * 100
        return None, total_ret_pct, cash, pd.DataFrame(trade_logs), latest_signal
    except Exception as e:
        st.error(f"5M strateji hatası: {str(e)}")
        return None, 0.0, init_cash, pd.DataFrame(), 0

# --- 4. GRAFİK OLUŞTURMA FONKSİYONU ---
def build_candlestick_chart(data_df, label_text):
    try:
        candles = go.Candlestick(
            x=data_df.index,
            open=data_df['Open'],
            high=data_df['High'],
            low=data_df['Low'],
            close=data_df['Close'],
            name=label_text
        )
        fig_obj = go.Figure(data=[candles])
        fig_obj.update_layout(
            xaxis_rangeslider_visible=False,
            height=450,
            template="plotly_dark",
            title=label_text
        )
        fig_obj.update_xaxes(title_text='Zaman')
        fig_obj.update_yaxes(title_text='Fiyat (USDT)')
        return fig_obj
    except Exception:
        return None

# --- 5. CANLI ALARMLARI İŞLEYEN FONKSİYON ---
def _candle_closed(last_ts, interval):
    tf = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}.get(interval, pd.Timedelta(days=1))
    close_at = last_ts + tf
    if close_at.tzinfo is None:
        close_at = close_at.tz_localize("UTC")
    return pd.Timestamp.utcnow() >= close_at

def process_live_alarms(b_token, c_id, leverage=1):
    if not st.session_state.alarms:
        return
    for alarm in st.session_state.alarms:
        if not alarm["is_active"]:
            continue
        alarm_raw = get_crypto_data(alarm["ticker"], alarm["period_candles"], alarm["interval"])
        if alarm_raw.empty or len(alarm_raw) < 10:
            continue
        try:
            last_ts = alarm_raw.index[-1]
            if not _candle_closed(last_ts, alarm["interval"]):
                continue
            if alarm.get("last_candle_ts") == last_ts:
                continue

            res = compute_strategy_performance(alarm_raw, 80, alarm["balance"] if alarm["balance"] > 0 else 10000.0, leverage=leverage)
            if res is not None:
                _, _, _, _, a_signal = res
                a_price = float(alarm_raw['Close'].iloc[-1])
                prev_price = alarm["last_price"]
                price_change = ((a_price - prev_price) / prev_price * 100) if prev_price > 0 else 0.0
                alarm["last_price"] = a_price
                alarm["last_candle_ts"] = last_ts
                if alarm["last_signal"] != a_signal:
                    now_str = pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')
                    cur_val = alarm["balance"] if alarm["balance"] > 0 else (alarm["crypto_amount"] * a_price)
                    action = ""
                    entry_price = 0.0
                    pnl_pct = 0.0
                    pnl_usd = 0.0

                    if a_signal == 1 and alarm["balance"] > 0:
                        margin = alarm["balance"]
                        notional = margin * leverage
                        alarm["crypto_amount"] = notional / a_price
                        alarm["margin"] = margin
                        entry_price = a_price
                        alarm["entry_price"] = a_price
                        alarm["balance"] = 0.0
                        action = (
                            f"🟢 *ALIM SİNYALİ*\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"💰 *Fiyat:* `${a_price:,.4f}`\n"
                            f"⚡ *Kaldıraç:* `{leverage}x`\n"
                            f"💵 *Teminat:* `${margin:,.2f}`\n"
                            f"📊 *Pozisyon Büyüklüğü:* `${notional:,.2f}`\n"
                            f"📈 *Miktar:* `{alarm['crypto_amount']:.6f}` `{alarm['ticker'].split('/')[1]}`\n"
                            f"🏦 *Kalan Kasa:* `${alarm['balance']:,.2f}`"
                        )
                    elif a_signal == 0 and alarm["crypto_amount"] > 0:
                        entry_price = alarm.get("entry_price", a_price)
                        margin = alarm.get("margin", alarm["crypto_amount"] * entry_price / max(leverage, 1))
                        notional_exit = alarm["crypto_amount"] * a_price
                        pnl_raw = notional_exit - (alarm["crypto_amount"] * entry_price)
                        pnl_usd = pnl_raw * leverage
                        pnl_pct = ((a_price - entry_price) / entry_price * 100 * leverage) if entry_price > 0 else 0.0
                        alarm["balance"] = max(margin + pnl_usd, 0.0)
                        action = (
                            f"🔴 *SATIM SİNYALİ*\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"💰 *Fiyat:* `${a_price:,.4f}`\n"
                            f"⚡ *Kaldıraç:* `{leverage}x`\n"
                            f"📊 *Satılan:* `{alarm['crypto_amount']:.6f}` `{alarm['ticker'].split('/')[1]}`\n"
                            f"💵 *Pozisyon Büyüklüğü:* `${notional_exit:,.2f}`\n"
                            f"{'✅' if pnl_usd >= 0 else '❌'} *Kâr/Zarar:* `{pnl_usd:+.2f}$` (`{pnl_pct:+.2f}%`)\n"
                            f"📈 *Fiyat Değişimi:* `{((a_price - entry_price) / entry_price * 100) if entry_price > 0 else 0:+.2f}%`\n"
                            f"🏦 *Toplam Kasa:* `${alarm['balance']:,.2f}`"
                        )
                        alarm["crypto_amount"] = 0.0
                    elif a_signal == 1:
                        action = (
                            f"🟢 *SİNYAL GÜNCELLENDİ*\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"💰 *Fiyat:* `${a_price:,.4f}`\n"
                            f"⚡ *Kaldıraç:* `{leverage}x`\n"
                            f"🏦 *Kasa:* `${alarm['balance']:,.2f}`\n"
                            f"📊 *Pozisyon:* `{alarm['crypto_amount']:.6f}`"
                        )
                    elif a_signal == 0:
                        action = (
                            f"🔴 *SİNYAL GÜNCELLENDİ*\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"💰 *Fiyat:* `${a_price:,.4f}`\n"
                            f"⚡ *Kaldıraç:* `{leverage}x`\n"
                            f"🏦 *Kasa:* `${alarm['balance']:,.2f}`"
                        )

                    if a_signal == 1:
                        alarm["entry_price"] = a_price

                    alarm["last_signal"] = a_signal
                    if action and b_token and c_id:
                        msg = (
                            f"⚡ *MEXC ALARM* ⚡\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"🪙 *Parite:* `{alarm['ticker']}`\n"
                            f"⏱ *Aralık:* `{alarm['interval']}`\n"
                            f"📅 *Tarih:* `{now_str}`\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"{action}\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"🏦 *Portföy Değeri:* `${cur_val:,.2f}`\n"
                            f"{'✅' if cur_val >= 10000 else '❌'} *Başlangıçtan:* `{((cur_val - 10000) / 10000 * 100):+.2f}%`"
                        )
                        send_telegram_signal(b_token, c_id, msg)
                        st.session_state.global_trade_history.append(f"[{alarm['ticker']}] {action.split(chr(10))[0]} | Kasa: ${cur_val:,.2f}")
        except Exception as e:
            st.session_state.global_trade_history.append(f"Alarm işleme hatası: {str(e)}")

# --- 6. ARAYÜZ BİLEŞENLERİ PANELİ ---
st.sidebar.header("⚙️ 1. Telegram Bağlantı Ayarları")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", key="tg_token")
chat_id = st.sidebar.text_input("Telegram Chat ID", type="password", key="tg_chat_id")

st.sidebar.markdown("---")
st.sidebar.header("🔍 2. Kripto Seçimi & Backtest Ayarları")

crypto_list = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ZEC/USDT",
    "DOGE/USDT", "SHIB/USDT", "NEAR/USDT", "SUI/USDT", "LTC/USDT",
    "TRX/USDT", "ONE/USDT", "ONDO/USDT", "MUBARAK/USDT", "BROCCOLI/USDT"
]

if "custom_cryptos" not in st.session_state:
    st.session_state.custom_cryptos = []

ticker = st.sidebar.selectbox("Kripto Para Seçin (MEXC Canlı)", crypto_list + st.session_state.custom_cryptos, key="ticker_k")

st.sidebar.markdown("**➕ Özel Kripto Ekle**")
custom_input = st.sidebar.text_input("Parite Girin (örn: PEPE/USDT)", key="custom_crypto")

def mexc_symbol_exists(symbol):
    try:
        exchange = ccxt.mexc({'enableRateLimit': True})
        markets = exchange.load_markets()
        return symbol in markets
    except Exception:
        return False

if custom_input:
    normalized = custom_input.strip().upper()
    if "/" not in normalized:
        normalized = normalized + "/USDT"
    exists = mexc_symbol_exists(normalized)
    if exists:
        st.sidebar.success(f"✅ {normalized} MEXC'de mevcut")
    else:
        st.sidebar.error(f"❌ {normalized} MEXC'de bulunamadı")

    if exists and normalized not in crypto_list and normalized not in st.session_state.custom_cryptos:
        if st.sidebar.button(f"➕ {normalized} Listeye Ekle", width="stretch"):
            st.session_state.custom_cryptos.append(normalized)
            st.rerun()
    elif normalized in crypto_list or normalized in st.session_state.custom_cryptos:
        st.sidebar.info(f"ℹ️ {normalized} zaten listede")

interval_label = st.sidebar.selectbox("Veri Sıklığı (Grafik Mum Tipi)", ["1 Saat", "1 Gün"], index=1)

strategy_mode = st.sidebar.selectbox(
    "Backtest Stratejisi",
    ["🤖 ML RandomForest (Momentum)", "⏱️ 5M Açılış Mumu (Long/Short)"],
    key="strategy_mode_k"
)
USE_5M_STRATEGY = strategy_mode.startswith("⏱️")

if USE_5M_STRATEGY:
    doji_threshold_ui = st.sidebar.number_input(
        "Doji Eşiği (|Gövde| / Aralık < bu değerse işlem yok)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.05, key="doji_k"
    )
    stop_mode_ui = st.sidebar.selectbox(
        "Stop Yöntemi",
        ["İlk Mumun Karşı Ucu", "Stop Çarpanı (x)", "ATR (14g, %)"],
        key="stop_mode_k"
    )
    STOP_MODE_MAP = {"İlk Mumun Karşı Ucu": "base", "Stop Çarpanı (x)": "mult", "ATR (14g, %)": "atr"}
    stop_mode_param = STOP_MODE_MAP[stop_mode_ui]
    stop_mult_ui = 3.0
    atr_frac_ui = 0.30
    if stop_mode_param == "mult":
        stop_mult_ui = st.sidebar.number_input("Stop Çarpanı (x)", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="stop_mult_k")
    elif stop_mode_param == "atr":
        atr_frac_ui = st.sidebar.number_input("ATR Yüzdesi (%)", min_value=5.0, max_value=50.0, value=30.0, step=5.0, key="atr_pct_k") / 100.0
    st.sidebar.info("ℹ️ 5M stratejisi: Günün ilk 5 dk mumu yönü → 2. mumda giriş, stop seçili yöntemle, çıkış gün sonu. Eğitim oranı ve model ayarları bu modda kullanılmaz.")

interval_mapping = {"1 Saat": "1h", "1 Gün": "1d"}
period_mapping = {"7 Gün": "7d", "30 Gün": "30d", "2 Ay": "2mo", "1 Yıl": "1y", "3 Yıl": "3y"}

time_period_unit = st.sidebar.selectbox("Geçmiş Test Süresi Birimi", ["Yıl", "Ay", "Hafta", "Gün", "Saat", "Tarih Aralığı"])

if time_period_unit == "Tarih Aralığı":
    import datetime as dt
    st.sidebar.markdown("**Başlangıç & Bitiş Tarihi**")
    start_date = st.sidebar.date_input("Başlangıç Tarihi", value=dt.date.today() - dt.timedelta(days=90), key="start_date")
    end_date = st.sidebar.date_input("Bitiş Tarihi", value=dt.date.today(), key="end_date")
    time_period_value = 1

    if start_date >= end_date:
        st.sidebar.error("Bitiş tarihi başlangıçtan sonra olmalı!")
        period_candles = 30
        time_period = "Tarih Aralığı (geçersiz)"
    else:
        days_diff = (end_date - start_date).days
        interval_factor = 24 if interval_label == "1 Saat" else 1
        period_candles = int(days_diff * interval_factor)
        time_period = f"{start_date.strftime('%d.%m.%Y')} → {end_date.strftime('%d.%m.%Y')}"
else:
    time_period_value = st.sidebar.number_input("Geçmiş Test Süresi Değeri", min_value=1, max_value=100, value=2, step=1, key="period_val")

    unit_factor = {"Yıl": 365, "Ay": 30, "Hafta": 7, "Gün": 1, "Saat": 1/24}
    interval_factor = 24 if interval_label == "1 Saat" else 1
    period_candles = int(time_period_value * unit_factor[time_period_unit] * interval_factor)
    time_period = f"{time_period_value} {time_period_unit}"

train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 90, key="train_size_k")
backtest_balance = st.sidebar.number_input("İşlem Giriş Bakiyesi ($) (pozisyon temeli)", min_value=100.0, value=1000.0, step=100.0, key="tbalance_k")
leverage = st.sidebar.number_input("Kaldıraç (x)", min_value=1, max_value=125, value=10, step=1, key="lev_k")
total_cash_manual = st.sidebar.number_input("💰 Toplam Kasa / Backtest Sermayesi ($)", min_value=100.0, value=20000.0, step=500.0, key="tcash_k")

st.sidebar.markdown("**🤖 Model Ayarları**")

def _preset_btc():
    st.session_state.wf_k = True
    st.session_state.nest_k = 400
    st.session_state.mdepth_k = 0
    st.session_state.mleaf_k = 1
    st.session_state.tbalance_k = 20000.0
    st.session_state.lev_k = 1

def _preset_eth():
    st.session_state.wf_k = False
    st.session_state.nest_k = 200
    st.session_state.mdepth_k = 0
    st.session_state.mleaf_k = 5
    st.session_state.tbalance_k = 20000.0
    st.session_state.lev_k = 1

st.sidebar.markdown("**⚡ Hazır Ayarlar**")
c_p1, c_p2 = st.sidebar.columns(2)
c_p1.button("BTC En İyi", width="stretch", on_click=_preset_btc)
c_p2.button("ETH En İyi", width="stretch", on_click=_preset_eth)

walk_forward = st.sidebar.checkbox("Walk-Forward (oversampling koruması)", value=True, key="wf_k")
n_estimators_ui = st.sidebar.number_input("Ağaç Sayısı (n_estimators)", min_value=50, max_value=1000, value=200, step=50, key="nest_k")
max_depth_ui = st.sidebar.number_input("Maks Derinlik (0 = sınırsız)", min_value=0, max_value=50, value=10, step=1, key="mdepth_k")
min_leaf_ui = st.sidebar.number_input("Min Yaprak (min_samples_leaf)", min_value=1, max_value=50, value=5, step=1, key="mleaf_k")

st.sidebar.markdown("---")
st.sidebar.header("🚨 3. Alarm Oluşturma")
alarm_init_balance = st.sidebar.number_input("Bu Alarma Özel Başlangıç Bakiyesi ($)", min_value=10.0, value=1000.0, step=100.0)
_total_alarm_cash = sum(a.get("balance", 0.0) for a in st.session_state.alarms)
st.sidebar.metric("💰 Toplam Kasa (Tüm Alarmlar)", f"${_total_alarm_cash:,.2f}")

all_crypto_options = crypto_list + st.session_state.custom_cryptos
multi_tickers = st.sidebar.multiselect(
    "Alarm Eklenecek Kriptolar (birden fazla seçebilirsin)",
    all_crypto_options,
    default=[ticker]
)

if st.sidebar.button("🚨 SEÇİLEN COİNLERİ ALARMLARA EKLE", width="stretch"):
    if not multi_tickers:
        st.sidebar.warning("En az 1 kripto seçin!")
    else:
        added = 0
        skipped = 0
        for sel in multi_tickers:
            if any(a["ticker"] == sel and a["interval"] == interval_mapping[interval_label] for a in st.session_state.alarms):
                skipped += 1
                continue
            new_alarm = {
                "id": len(st.session_state.alarms) + 1,
                "ticker": sel,
                "interval": interval_mapping[interval_label],
                "period": time_period,
                "period_candles": period_candles,
                "balance": float(alarm_init_balance),
                "crypto_amount": 0.0,
                "margin": 0.0,
                "entry_price": 0.0,
                "last_signal": 0,
                "last_candle_ts": None,
                "last_price": 0.0,
                "is_active": True
            }
            st.session_state.alarms.append(new_alarm)
            added += 1
        if added:
            _cfg = _load_alarm_config()
            _cfg["alarms"] = st.session_state.alarms
            _cfg["telegram_bot_token"] = bot_token
            _cfg["telegram_chat_id"] = chat_id
            _cfg["leverage"] = leverage
            _save_alarm_config(_cfg)
            st.sidebar.success(f"✅ {added} kripto alarm havuzuna eklendi.")
        if skipped:
            st.sidebar.warning(f"⚠️ {skipped} kripto zaten mevcuttu.")

tab1, tab2, tab3 = st.tabs(["📊 1. Gelişmiş Backtest Alanı", "🚨 2. Canlı Alarm Havuzu & Excel", "🕒 3. Global İşlem Günlüğü"])

raw_df = get_crypto_data(ticker, period_candles, interval_mapping[interval_label])
if USE_5M_STRATEGY:
    days_count = max(period_candles // (24 if interval_label == "1 Saat" else 1), 1)
    df_5m = load_ohlcv_5m(ticker, min(days_count * 288, 50000))
    processed_df, total_net_return_pct, final_wallet_value, backtest_logs, latest_signal = compute_5m_open_strategy(
        df_5m, total_cash_manual, leverage=leverage, trade_margin=backtest_balance, doji_threshold=float(doji_threshold_ui),
        stop_mode=stop_mode_param, stop_mult=float(stop_mult_ui), atr_frac=float(atr_frac_ui)
    )
else:
    processed_df, total_net_return_pct, final_wallet_value, backtest_logs, latest_signal = compute_strategy_performance(raw_df, train_size, total_cash_manual, walk_forward=walk_forward, leverage=leverage, trade_margin=backtest_balance, n_estimators=int(n_estimators_ui), max_depth=int(max_depth_ui), min_samples_leaf=int(min_leaf_ui))

st.sidebar.markdown("---")
st.sidebar.header("📈 4. Gerçek Zamanlı İşlem")
if bot_token and chat_id:
    st.sidebar.success("✅ Telegram Bağlantısı Aktif")
else:
    st.sidebar.info("Telegram ayarları eksik")

# --- SEKME 1: BACKTEST VE ANALİZ ALANI ---
with tab1:
    if raw_df.empty:
        st.warning("Veri çekilemedi. Lütfen internet bağlantınızı kontrol edin.")
    else:
        st.markdown(f"### 📈 {ticker} MEXC Canlı Strateji Analiz Paneli")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("💰 Toplam Kasa (Başlangıç)", f"${total_cash_manual:,.2f}")
        with col2:
            st.metric("Son Kasa", f"${final_wallet_value:,.2f}")
        with col3:
            st.metric("Net Getiri", f"{total_net_return_pct:+.2f}%", delta=f"{total_net_return_pct:+.2f}%")
        with col4:
            if USE_5M_STRATEGY:
                signal_text = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⏳ YOK"}.get(latest_signal, "⏳ YOK")
            else:
                signal_text = "ALIM" if latest_signal == 1 else "SATIM"
            st.metric("Son Sinyal", signal_text)
        with col5:
            st.metric("Son Test Tarihi", raw_df.index[-1].strftime('%Y-%m-%d') if not raw_df.empty else "N/A")

        st.markdown("---")

        fig = build_candlestick_chart(raw_df, f"{ticker} Fiyat Grafiği")
        if fig:
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        if USE_5M_STRATEGY:
            with st.expander("ℹ️ Strateji Şartları — 5M Açılış Mumu", expanded=False):
                st.markdown(f"""
**Yön:** **LONG ve SHORT** — iki yönlü işlem.

**Mantık (prediksiyon yok, kural tabanlı):**
1. Her günün **ilk 5 dk mumu** alınır → gövde = |Kapanış - Açılış|, aralık = Yüksek - Düşük
2. Gövde / Aralık **{doji_threshold_ui:.2f}** eşiğinin altındaysa → **DOJI, işlem yok**
3. Gövde pozitifse → 2. mumun açılışında **LONG**; negatifse → 2. mumun açılışında **SHORT**

**Stop ({stop_mode_ui}):** {"İlk mumun karşı ucu (LONG'da dip, SHORT'ta tavan)" if stop_mode_param == "base" else (f"Girişe göre mesafe × {stop_mult_ui:.1f}" if stop_mode_param == "mult" else f"14 günlük ATR × %{atr_frac_ui*100:.0f}")}

**Çıkış:** Aynı günün son 5 dk mumunun kapanışı (gün sonu).

**Risk:** Kaldıraç, likidasyon ve compounding (kâr sonraki işleme eklenir) ML moduyla aynı matematik motorunu kullanır.

**Doji Eşiği:** {doji_threshold_ui:.2f} — eşik düşerse daha fazla işlem, yükselirse daha seçici.
                """)
        else:
            with st.expander("ℹ️ Strateji Şartları ve ML Mantığı", expanded=False):
                st.markdown(f"""
**Yön:** Sadece **LONG (Alım)** — short/satış yönü yok.

**ML Modeli:** RandomForestClassifier ({train_size}% eğitim / {100-train_size}% test)

**Özellikler (Features):**

| Özellik | Açıklama |
|---|---|
| Return | Bir önceki mumun fiyat değişimi (%) |
| RSI (14) | Relative Strength Index — 0–100 arası momentum |
| Price_to_SMA20 | Fiyatın 20 periyotluk ortalamaya oranı |

**Hedef (Label):** Sonraki mumun kapanışı, şimdiki kapanıştan yüksekse **1 (AL)**, değilse **0 (BEKLE)**.

**İşlem Kuralları:**
- Model **1** dönerse ve pozisyonda değilse → 🟢 **ALIM** (tüm bakiye ile, %0.02 taker fee)
- Model **0** dönerse ve pozisyondaysa → 🔴 **SATIM** (nakde geçiş, %0.02 taker fee)
- Pozisyon yokken 0 gelirse → bekleme (işlem yapılmaz)

**Zamanlama:** Sinyaller ve canlı alarmlar **yalnızca mum kapandığında** değerlendirilir (1 Gün → 00:00 UTC, 1 Saat → saat başı). Kapanmam mumla işlem yapılmaz.

**Sinyal Kaynağı:** RSI aşırı alım/satım bölgeleri, fiyat-SMA ilişkisi ve getiri momentumu birleştirilerek RandomForest ile sınıflandırma yapılır.
                """)

        st.markdown("---")

        if processed_df is not None and len(processed_df) > 0:
            disp_df = processed_df[['Open', 'High', 'Low', 'Close', 'Signal_Target', 'Predicted_Signal']].iloc[-50:].copy()
            disp_df['Trend'] = disp_df['Predicted_Signal'].map({0: '🔻 Down', 1: '🔺 Up'})
            is_correct = disp_df['Predicted_Signal'] == disp_df['Signal_Target']
            disp_df['Sonuç'] = is_correct.map({True: '✅ Başarılı', False: '❌ Başarısız'})

            total_preds = len(disp_df)
            success_count = int(is_correct.sum())
            fail_count = total_preds - success_count
            success_rate = (success_count / total_preds * 100) if total_preds > 0 else 0.0

            st.subheader(
                f"🎯 Tahmin Edilen Sinyaller — "
                f"✅ {success_count} başarılı | ❌ {fail_count} başarısız | "
                f"Başarı Oranı: %{success_rate:.1f} | Kazanç: {total_net_return_pct:+.2f}%"
            )

            disp_df = disp_df.drop(columns=['Signal_Target', 'Predicted_Signal'])

            def color_row(row):
                if row['Sonuç'] == '✅ Başarılı':
                    return ['background-color: #e8f5e9'] * len(row)
                elif row['Sonuç'] == '❌ Başarısız':
                    return ['background-color: #ffebee'] * len(row)
                return [''] * len(row)

            st.dataframe(disp_df.style.apply(color_row, axis=1), width="stretch")

        st.markdown("---")

        if not backtest_logs.empty:
            st.subheader("📋 İşlem Geçmişi")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("🪙 Parite", ticker)
            with c2:
                st.metric("⏱ Aralık", time_period)
            with c3:
                st.metric("📅 Mum Tipi", interval_label)
            with c4:
                st.metric("🤖 Eğitim Oranı", f"%{train_size}" if not USE_5M_STRATEGY else "—")
            with c5:
                st.metric("💵 Toplam Kasa", f"${total_cash_manual:,.0f}")
            with c6:
                st.metric("⚡ Kaldıraç", f"{leverage}x")

            def trade_color(row):
                if row['Sonuç'] == '💥 Likide':
                    return ['background-color: #ffebee'] * len(row)
                if row['Sonuç'] == '✅ Başarılı':
                    return ['background-color: #e8f5e9'] * len(row)
                elif row['Sonuç'] == '❌ Başarısız':
                    return ['background-color: #ffebee'] * len(row)
                return [''] * len(row)

            st.dataframe(backtest_logs.style.apply(trade_color, axis=1), width="stretch")

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                backtest_logs.to_excel(writer, index=False, sheet_name='İşlem Geçmişi')
                raw_df.to_excel(writer, index=True, sheet_name='Ham Veri')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Excel'e İndir",
                data=excel_buffer,
                file_name="backtest_sonuclari.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- SEKME 2: CANLI ALARM HAVUZU & EXCEL ---
with tab2:
    st.markdown("### 🚨 Canlı Alarm Havuzu")

    if not st.session_state.alarms:
        st.info("Henüz eklenmiş alarm bulunmamaktadır. Sol tarafta alarm ekleyin.")
    else:
        for i, alarm in enumerate(st.session_state.alarms):
            with st.expander(f"{alarm['ticker']} ({alarm['interval']}) - {'✅ Aktif' if alarm['is_active'] else '⏸️ Pasif'}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bakiye ($)", f"${alarm['balance']:,.2f}")
                with col2:
                    st.metric("Kripto Miktarı", f"{alarm['crypto_amount']:.4f}")
                with col3:
                    st.metric("Son Fiyat", f"${alarm['last_price']:.4f}" if alarm['last_price'] > 0 else "Belirtilmedi")

                is_active = st.checkbox("Aktif", value=alarm["is_active"], key=f"active_{i}")
                alarm["is_active"] = is_active

                if st.button("🗑️ Sil", key=f"delete_{i}"):
                    st.session_state.alarms.pop(i)
                    _cfg = _load_alarm_config()
                    _cfg["alarms"] = st.session_state.alarms
                    _save_alarm_config(_cfg)
                    st.rerun()

            _cfg = _load_alarm_config()
            _cfg["alarms"] = st.session_state.alarms
            _cfg["telegram_bot_token"] = bot_token
            _cfg["telegram_chat_id"] = chat_id
            _cfg["leverage"] = leverage
            _save_alarm_config(_cfg)

        active_count = sum(1 for a in st.session_state.alarms if a["is_active"])
        total_alarm_cash = sum(a.get("balance", 0.0) for a in st.session_state.alarms)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("💰 Toplam Kasa (Alarmlar)", f"${total_alarm_cash:,.2f}")
        with m2:
            st.metric("📊 Aktif Alarm", f"{active_count} / {len(st.session_state.alarms)}")

        if bot_token and chat_id and len(st.session_state.alarms) > 0:
            st.info("💡 Not: Alarm sinyalleri Telegram üzerinden gönderilecek.")

        if st.session_state.alarms:
            alarm_export_df = pd.DataFrame([{
                "ID": a["id"],
                "Kripto": a["ticker"],
                "Aralık": a["interval"],
                "Periyot": a["period"],
                "Bakiye ($)": a["balance"],
                "Kripto Miktarı": a["crypto_amount"],
                "Son Fiyat": a["last_price"],
                "Son Sinyal": a.get("last_signal", 0),
                "Aktif": "✅" if a["is_active"] else "⏸️"
            } for a in st.session_state.alarms])

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                alarm_export_df.to_excel(writer, index=False, sheet_name='Alarmlar')
                if not backtest_logs.empty:
                    backtest_logs.to_excel(writer, index=False, sheet_name='İşlem Geçmişi')
            excel_buf.seek(0)
            st.download_button(
                label="📥 Alarm ve İşlem Excel'i İndir",
                data=excel_buf,
                file_name="alarm_verileri.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- SEKME 3: GLOBAL İŞLEM GÜNLÜĞÜ ---
with tab3:
    st.markdown("### 🕒 Global İşlem Günlüğü")

    if not st.session_state.global_trade_history:
        st.info("Henüz kayıt edilmiş işlem bulunmamaktadır.")
    else:
        for entry in reversed(st.session_state.global_trade_history[-50:]):
            if isinstance(entry, str):
                st.markdown(f"- {entry}")
            else:
                st.markdown(f"- {entry}")

    if st.button("🗑️ Geçmişi Temizle"):
        st.session_state.global_trade_history = []
        st.rerun()
