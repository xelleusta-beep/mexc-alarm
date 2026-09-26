import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

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

def compute_strategy_performance(df_input, train_ratio, init_cash=10000.0, walk_forward=True, leverage=1, trade_margin=None):
    working_df = df_input.copy()
    working_df["Return"] = working_df["Close"].pct_change()
    delta = working_df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    working_df["RSI"] = 100 - (100 / (1 + rs))
    working_df["SMA_20"] = working_df["Close"].rolling(window=20).mean()
    working_df["Price_to_SMA"] = working_df["Close"] / working_df["SMA_20"]
    working_df["Signal_Target"] = (working_df["Close"].shift(-1) > working_df["Close"]).astype(int)
    working_df.dropna(inplace=True)
    if len(working_df) < 30:
        return None, 0.0, init_cash, pd.DataFrame(), 0

    features = ["Return", "RSI", "Price_to_SMA"]
    X = working_df[features]
    y = working_df["Signal_Target"]

    if walk_forward and len(working_df) >= 60:
        n_splits = min(10, len(working_df) // 30)
        min_train = int(len(X) * (train_ratio / 100))
        min_train = max(min_train, 30)
        test_size = (len(X) - min_train) // n_splits
        test_size = max(test_size, 5)
        working_df["Predicted_Signal"] = 0
        working_df["Confidence"] = 0.0
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
            proba = model.predict_proba(X.iloc[test_start:test_end])
            conf = proba.max(axis=1)
            preds = model.predict(X.iloc[test_start:test_end])
            working_df.iloc[test_start:test_end, working_df.columns.get_loc("Predicted_Signal")] = preds
            working_df.iloc[test_start:test_end, working_df.columns.get_loc("Confidence")] = conf
    else:
        split_idx = int(len(X) * (train_ratio / 100))
        if split_idx == 0 or split_idx >= len(X):
            split_idx = int(len(X) * 0.8)
        model = RandomForestClassifier(
            random_state=42, n_estimators=200, max_depth=10,
            min_samples_split=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1,
        )
        model.fit(X[:split_idx], y[:split_idx])
        preds = model.predict(X)
        working_df["Predicted_Signal"] = preds
        min_train = split_idx

    trade_logs = []
    in_pos = False
    ent_price = 0.0
    ent_date = None
    cash = init_cash
    margin_size = float(trade_margin) if trade_margin and trade_margin > 0 else cash
    units = 0.0
    pos_low = float("inf")
    trade_cash_in = 0.0
    cash_at_entry = 0.0
    start_idx = min_train

    for i in range(start_idx, len(working_df)):
        c_date = working_df.index[i]
        c_price = float(working_df["Close"].iloc[i])
        c_low = float(working_df["Low"].iloc[i])
        c_sig = int(working_df["Predicted_Signal"].iloc[i])

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
                        "ID": len(trade_logs) + 1,
                        "Yon": "AL->LIQ",
                        "Giris": ent_date.strftime("%Y-%m-%d"),
                        "Cikis": c_date.strftime("%Y-%m-%d"),
                        "GirisFiyat": round(ent_price, 2),
                        "CikisFiyat": round(liq_price, 2),
                        "ToplamKasa": 0.0,
                        "Miktar": round(trade_size, 2),
                        "NetPnL": round(-cash_at_entry, 2),
                        "Getiri%": round(ret_pct * leverage, 2),
                        "MaxDusus%": round(max_dd_pct, 2),
                        "MaxDususLev%": round(max_dd_lev_pct, 2),
                        "MaxDusus$": round(max_dd_usd, 2),
                        "Sonuc": "LIKIDE",
                    })
                    units = 0.0
                    in_pos = False
                    pos_low = float("inf")
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
                "ID": len(trade_logs) + 1,
                "Yon": "AL->SAT",
                "Giris": ent_date.strftime("%Y-%m-%d"),
                "Cikis": c_date.strftime("%Y-%m-%d"),
                "GirisFiyat": round(ent_price, 2),
                "CikisFiyat": round(c_price, 2),
                "ToplamKasa": round(cash, 2),
                "Miktar": round(trade_size, 2),
                "NetPnL": round(pnl, 2),
                "Getiri%": round(ret_pct * leverage, 2),
                "MaxDusus%": round(max_dd_pct, 2),
                "MaxDususLev%": round(max_dd_lev_pct, 2),
                "MaxDusus$": round(max_dd_usd, 2),
                "Sonuc": "LIKIDE" if is_liq else ("BASARILI" if ret_pct > 0 else "BASARISIZ"),
            })
            units = 0.0
            in_pos = False
            pos_low = float("inf")
            if is_liq:
                margin_size = 0.0

    c_last_price = float(working_df["Close"].iloc[-1])
    if in_pos:
        unreal_raw = (units * c_last_price) * 0.9998 - trade_cash_in
        final_val = max(cash_at_entry + unreal_raw * leverage, 0.0)
    else:
        final_val = cash
    total_ret_pct = ((final_val - init_cash) / init_cash) * 100
    latest_signal = int(working_df["Predicted_Signal"].iloc[-1])
    return working_df, total_ret_pct, final_val, pd.DataFrame(trade_logs), latest_signal

def run(symbol):
    print(f"\n{'='*70}")
    print(f"  {symbol} | 1 Gun | 1 Yil | %90 Egitim | 10x | Kasa $20,000 | Giris $1,000")
    print(f"{'='*70}")
    df = get_crypto_data(symbol, 365, "1d")
    if df.empty or len(df) < 30:
        print("Veri yok!")
        return
    print(f"Veri: {len(df)} mum | {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Baslangic fiyati: ${float(df['Close'].iloc[0]):,.2f} | Son: ${float(df['Close'].iloc[-1]):,.2f}")
    hodl_ret = ((float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0])) * 100
    print(f"HODL getirisi: {hodl_ret:+.2f}%")

    processed, ret, final_val, logs, last_sig = compute_strategy_performance(
        df, 90, init_cash=20000.0, walk_forward=True, leverage=10, trade_margin=1000.0
    )
    if logs is None or logs.empty:
        print("Islem yok!")
        return

    print(f"\nSonuc: {ret:+.2f}% | Son Kasa: ${final_val:,.2f} | Son Sinyal: {'AL' if last_sig==1 else 'SAT'}")
    print(f"Islem sayisi: {len(logs)}")
    wins = logs[logs["NetPnL"] > 0]
    losses = logs[logs["NetPnL"] <= 0]
    print(f"Kazanma: {len(wins)}/{len(logs)} ({100*len(wins)/len(logs):.1f}%)")
    print(f"Toplam Kar: ${wins['NetPnL'].sum():,.2f} | Toplam Zarar: ${losses['NetPnL'].sum():,.2f}")
    print(f"Ort Islem: ${logs['NetPnL'].mean():,.2f} | En Iyi: ${logs['NetPnL'].max():,.2f} | En Kotu: ${logs['NetPnL'].min():,.2f}")
    liqs = logs[logs["Sonuc"] == "LIKIDE"]
    if len(liqs):
        print(f"LIQ: {len(liqs)}")
    print(f"\nIslemler:")
    print(logs.to_string(index=False))

print("Backtest basliyor...")
run("BTC/USDT")
run("ETH/USDT")
