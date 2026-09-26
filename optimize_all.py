import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import itertools
import json
import os
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
    if not all_candles:
        return pd.DataFrame()
    all_candles = all_candles[-limit:]
    df = pd.DataFrame(all_candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df[~df.index.duplicated(keep="last")][["Open", "High", "Low", "Close", "Volume"]]

def backtest(df, train_ratio=90, init_cash=20000.0, leverage=1, trade_margin=20000.0,
             walk_forward=True, n_estimators=200, max_depth=None, min_samples_leaf=5):
    working_df = df.copy()
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
        return None

    features = ["Return", "RSI", "Price_to_SMA"]
    X = working_df[features]
    y = working_df["Signal_Target"]

    def make_model():
        return RandomForestClassifier(
            random_state=42, n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=10, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", n_jobs=-1,
        )

    if walk_forward and len(working_df) >= 60:
        n_splits = min(10, len(working_df) // 30)
        min_train = max(int(len(X) * (train_ratio / 100)), 30)
        test_size = max((len(X) - min_train) // n_splits, 5)
        working_df["Predicted_Signal"] = 0
        for fold in range(n_splits):
            train_end = min_train + fold * test_size
            test_start = train_end
            test_end = min(test_start + test_size, len(X))
            if test_start >= len(X):
                break
            model = make_model()
            model.fit(X[:train_end], y[:train_end])
            preds = model.predict(X.iloc[test_start:test_end])
            working_df.iloc[test_start:test_end, working_df.columns.get_loc("Predicted_Signal")] = preds
    else:
        split_idx = int(len(X) * (train_ratio / 100))
        if split_idx == 0 or split_idx >= len(X):
            split_idx = int(len(X) * 0.8)
        model = make_model()
        model.fit(X[:split_idx], y[:split_idx])
        working_df["Predicted_Signal"] = model.predict(X)
        min_train = split_idx

    in_pos = False
    ent_price = 0.0
    cash = init_cash
    margin_size = float(trade_margin)
    units = 0.0
    pos_low = float("inf")
    trade_cash_in = 0.0
    cash_at_entry = 0.0
    n_trades = 0
    n_wins = 0

    for i in range(min_train, len(working_df)):
        c_price = float(working_df["Close"].iloc[i])
        c_low = float(working_df["Low"].iloc[i])
        c_sig = int(working_df["Predicted_Signal"].iloc[i])

        if c_sig == 1 and not in_pos and cash > 0.01 and margin_size > 0.01:
            trade_cash_in = min(margin_size, cash)
            cash_at_entry = cash
            units = (trade_cash_in / c_price) * 0.9998
            ent_price = c_price
            in_pos = True
            pos_low = c_low

        if in_pos:
            if c_low < pos_low:
                pos_low = c_low
            if trade_cash_in > 0 and leverage > 0:
                liq_drop = cash_at_entry / (trade_cash_in * leverage)
                if liq_drop < 1.0:
                    liq_price = ent_price * (1.0 - liq_drop)
                    if pos_low <= liq_price:
                        cash = 0.0
                        n_trades += 1
                        units = 0.0
                        in_pos = False
                        pos_low = float("inf")
                        margin_size = 0.0
                        continue

        if c_sig == 0 and in_pos:
            cash_raw = (units * c_price) * 0.9998
            pnl = (cash_raw - trade_cash_in) * leverage
            cash = max(cash_at_entry + pnl, 0.0)
            margin_size = max(margin_size + pnl, 0.0)
            n_trades += 1
            if pnl > 0:
                n_wins += 1
            units = 0.0
            in_pos = False
            pos_low = float("inf")
            if cash <= 0.0 and pnl < 0:
                margin_size = 0.0

    c_last = float(working_df["Close"].iloc[-1])
    if in_pos:
        unreal = ((units * c_last) * 0.9998 - trade_cash_in) * leverage
        final_val = max(cash_at_entry + unreal, 0.0)
    else:
        final_val = cash
    ret = ((final_val - init_cash) / init_cash) * 100
    return {"ret": round(ret, 2), "final": round(final_val, 2), "trades": n_trades, "wins": n_wins}

COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "ZEC/USDT",
    "DOGE/USDT", "SHIB/USDT", "NEAR/USDT", "SUI/USDT", "LTC/USDT",
    "TRX/USDT", "ONE/USDT", "ONDO/USDT", "MUBARAK/USDT", "BROCCOLI/USDT",
]

margins = [1000, 20000]
walk_flags = [True, False]
n_est_list = [200]
depth_list = [None, 10]
leaf_list = [1, 5]

results = {}
for coin in COINS:
    print(f"=== {coin} ===", flush=True)
    try:
        df = get_crypto_data(coin, 365, "1d")
    except Exception as e:
        print(f"  veri hatasi: {e}", flush=True)
        continue
    if df.empty or len(df) < 50:
        print("  veri yok", flush=True)
        continue
    hodl = ((float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0])) * 100
    best = None
    for margin, wf, ne, depth, leaf in itertools.product(margins, walk_flags, n_est_list, depth_list, leaf_list):
        try:
            r = backtest(df, trade_margin=float(margin), walk_forward=wf,
                         n_estimators=ne, max_depth=depth, min_samples_leaf=leaf)
        except Exception:
            continue
        if r is None:
            continue
        if best is None or r["ret"] > best["ret"]:
            best = {
                "ret": r["ret"], "final": r["final"], "trades": r["trades"], "wins": r["wins"],
                "giris": margin, "wf": wf, "est": ne,
                "depth": depth if depth else 0, "leaf": leaf,
            }
    if best:
        best["hodl"] = round(hodl, 2)
        results[coin] = best
        print(f"  BEST: {best['ret']:+.2f}% giris={best['giris']} wf={best['wf']} est={best['est']} depth={best['depth']} leaf={best['leaf']} | HODL {hodl:+.2f}%", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_settings.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nKaydedildi: {out}")
