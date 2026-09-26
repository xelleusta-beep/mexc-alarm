import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import itertools
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
    all_candles = all_candles[-limit:]
    df = pd.DataFrame(all_candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df[~df.index.duplicated(keep="last")][["Open", "High", "Low", "Close", "Volume"]]

def backtest(df, train_ratio=90, init_cash=20000.0, leverage=1, trade_margin=20000.0,
             walk_forward=True, n_estimators=200, max_depth=10,
             min_samples_split=10, min_samples_leaf=5):
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
        return None, 0.0, init_cash, 0, 0

    features = ["Return", "RSI", "Price_to_SMA"]
    X = working_df[features]
    y = working_df["Signal_Target"]

    def make_model():
        return RandomForestClassifier(
            random_state=42, n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
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
    margin_size = float(trade_margin) if trade_margin and trade_margin > 0 else cash
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
    return ret, final_val, n_trades, n_wins

def run(symbol):
    print(f"\n{'='*80}")
    print(f"  OPTIMIZASYON: {symbol} | 1G | 1Y | %90 | 1x | Kasa $20,000")
    print(f"{'='*80}")
    df = get_crypto_data(symbol, 365, "1d")
    hodl = ((float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0])) * 100
    print(f"Veri: {len(df)} mum | HODL: {hodl:+.2f}%")

    results = []
    # Daraltılmış grid
    margins = [1000, 5000, 10000, 20000]
    walk_flags = [True, False]
    n_est_list = [200, 400]
    depth_list = [10, None]
    leaf_list = [1, 5]

    count = 0
    for margin, wf, ne, depth, leaf in itertools.product(margins, walk_flags, n_est_list, depth_list, leaf_list):
        try:
            ret, final_val, nt, nw = backtest(
                df, train_ratio=90, init_cash=20000.0, leverage=1,
                trade_margin=float(margin), walk_forward=wf,
                n_estimators=ne, max_depth=depth, min_samples_leaf=leaf,
            )
        except Exception:
            continue
        if ret is None:
            continue
        count += 1
        results.append({
            "Giris": margin,
            "WF": "Y" if wf else "N",
            "Est": ne,
            "Depth": str(depth),
            "Leaf": leaf,
            "Ret%": round(ret, 2),
            "Kasa": round(final_val, 2),
            "Islem": nt,
            "Win": nw,
        })

    results_df = pd.DataFrame(results).sort_values("Ret%", ascending=False)
    print(f"Toplam denenen: {count}")
    print(f"\n--- EN IYI 15 ---")
    print(results_df.head(15).to_string(index=False))
    print(f"\n--- EN KOTU 5 ---")
    print(results_df.tail(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n*** EN IYI: Giris=${best['Giris']} | WF={best['WF']} | Est={best['Est']} | Depth={best['Depth']} | Leaf={best['Leaf']}")
    print(f"    Getiri: {best['Ret%']:+.2f}% | Kasa: ${best['Kasa']:,.2f} | Islem: {best['Islem']} (Win {best['Win']})")

run("BTC/USDT")
run("ETH/USDT")
